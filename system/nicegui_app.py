"""
system/nicegui_app.py
=====================
ExhibitionBench 策展辅助系统 — NiceGUI 版本（v3）

三个任务 Tab（按难度升序排列）：
  Tab 1 · MEIP  — 展品补全（10-way 排序，中等）
  Tab 2 · ECD   — 展览一致性检测（二分类，最简单）
  Tab 3 · TES   — 主题策展检索（50-way 排序，最难）

优化点：
  - BM25 索引启动时预建，搜索不重建
  - 所有 IO 在线程池执行，不阻塞事件循环
  - 真实 benchmark 样本下拉选择（MEIP / ECD）
  - 图片懒加载

用法：
    pip install nicegui rank_bm25 openai
    python system/nicegui_app.py [--port 7861]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from nicegui import ui, app as ng_app, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE         = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE / "data"
LOG_DIR      = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)
FEEDBACK_LOG = LOG_DIR / "feedback_nicegui.jsonl"

API_KEY  = "sk-TpK0g832p8LbMXTdI_pjkQ"
API_BASE = "http://csig.litellm.prod.sgpolaris/v1"
MODEL    = "gpt-5.2"

# ─────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────
_objects: dict[str, dict]  = {}
_exhibitions: list[dict]   = []
_bm25_index                = None
_bm25_ids: list[str]       = []
_meip_samples: list[dict]  = []
_ecd_samples: list[dict]   = []
_tes_samples: list[dict]   = []


def _pick_file(stem: str) -> Optional[Path]:
    """选最新版本: _v4 > _v3_fixed > _v3 > _v2 > bare"""
    for suffix in ["_v4", "_v3_fixed", "_v3", "_v2", ""]:
        p = DATA_DIR / f"{stem}{suffix}.jsonl"
        if p.exists():
            return p
    return None


def load_data():
    global _objects, _exhibitions, _bm25_index, _bm25_ids
    global _meip_samples, _ecd_samples, _tes_samples

    # objects / exhibitions（供 BM25 + TES）
    obj_path = DATA_DIR / "objects.jsonl"
    exh_path = DATA_DIR / "exhibitions.jsonl"
    if obj_path.exists():
        with open(obj_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _objects[r["id"]] = r
    if exh_path.exists():
        with open(exh_path, encoding="utf-8") as f:
            for line in f:
                _exhibitions.append(json.loads(line))
    log.info(f"加载 {len(_objects):,} 件展品, {len(_exhibitions):,} 个展览")

    # BM25 索引（预建）
    try:
        from rank_bm25 import BM25Okapi
        corpus, ids = [], []
        for oid, obj in _objects.items():
            text = (
                f"{obj.get('title','')} {obj.get('description','')} "
                f"{obj.get('culture','')} {obj.get('medium','')}"
            ).lower().split()
            corpus.append(text)
            ids.append(oid)
        _bm25_index = BM25Okapi(corpus)
        _bm25_ids   = ids
        log.info(f"BM25 索引预建完成，共 {len(ids):,} 条")
    except ImportError:
        log.warning("rank_bm25 未安装，使用关键词回退")

    # MEIP benchmark samples
    meip_f = _pick_file("meip_samples")
    if meip_f:
        with open(meip_f, encoding="utf-8") as f:
            _meip_samples = [json.loads(l) for l in f]
        log.info(f"MEIP 样本: {len(_meip_samples)} 条 (from {meip_f.name})")

    # ECD benchmark samples
    ecd_f = _pick_file("ecd_samples")
    if ecd_f:
        with open(ecd_f, encoding="utf-8") as f:
            _ecd_samples = [json.loads(l) for l in f]
        log.info(f"ECD 样本: {len(_ecd_samples)} 条 (from {ecd_f.name})")

    # TES benchmark samples
    tes_f = _pick_file("tes_samples")
    if tes_f:
        with open(tes_f, encoding="utf-8") as f:
            _tes_samples = [json.loads(l) for l in f]
        log.info(f"TES 样本: {len(_tes_samples)} 条 (from {tes_f.name})")


# ─────────────────────────────────────────────────────────────
# 检索后端
# ─────────────────────────────────────────────────────────────
def _bm25_search(query: str, top_k: int = 20) -> list[dict]:
    if _bm25_index is not None:
        tokens  = query.lower().split()
        scores  = _bm25_index.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [_objects[_bm25_ids[i]] for i in top_idx if _bm25_ids[i] in _objects]
    # 回退
    tokens = query.lower().split()
    scored = [
        (sum(1 for t in tokens if t in
             f"{o.get('title','')} {o.get('description','')} {o.get('culture','')} {o.get('medium','')}".lower()),
         o)
        for o in _objects.values()
    ]
    scored.sort(key=lambda x: -x[0])
    return [o for _, o in scored[:top_k]]


def _llm_rerank_tes(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """TES: 从 50 个候选主题中排序，返回 top_k。candidates 是 TES 候选列表。"""
    try:
        from openai import OpenAI
        client     = OpenAI(api_key=API_KEY, base_url=API_BASE)
        # TES candidates 是 theme-level，每个候选有 theme / title / description / sample_objects
        cand_block = "\n".join(
            f"ID:{i+1} | {c.get('title', c.get('theme',''))} | "
            f"{str(c.get('description',''))[:120]}"
            for i, c in enumerate(candidates[:50])
        )
        prompt = (
            f"You are an expert museum curator. Given the query theme: '{query}'\n\n"
            f"Rank the following {len(candidates)} exhibition themes by relevance.\n\n"
            f"Candidates:\n{cand_block}\n\n"
            f"Return ONLY a JSON array of the top {top_k} IDs (1-based), e.g. [3, 1, 7, ...]"
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw       = resp.choices[0].message.content
        arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if arr_match:
            nums   = json.loads(arr_match.group())
            ranked = [candidates[int(n)-1] for n in nums if 1 <= int(n) <= len(candidates)]
            seen   = {c["id"] for c in ranked}
            for c in candidates:
                if c["id"] not in seen:
                    ranked.append(c)
            return ranked[:top_k]
    except Exception as e:
        log.warning(f"TES LLM 重排失败: {e}")
    return candidates[:top_k]


def _llm_predict_meip(sample: dict) -> tuple[str, Optional[dict]]:
    """MEIP：给定 context + candidates，预测 gold。"""
    context    = sample.get("context", [])
    candidates = sample.get("candidates", [])
    gold_id    = sample.get("gold_id", "")

    if not candidates:
        return "无候选展品", None

    ctx_block  = "\n".join(
        f"- {i.get('title','')} | {i.get('culture','')} | {i.get('date','')} | {i.get('medium','')}"
        for i in context
    )
    cand_block = "\n".join(
        f"ID:{i+1} | {c.get('title','')} | {c.get('culture','')} | "
        f"{c.get('date','')} | {c.get('medium','')}"
        for i, c in enumerate(candidates)
    )
    prompt = (
        "You are an expert museum curator. Given the following artworks already in an exhibition, "
        "identify the single best artwork from the candidates to complete it.\n\n"
        f"Exhibition theme: {sample.get('exhibition_theme','(unknown)')}\n\n"
        f"Existing artworks:\n{ctx_block}\n\n"
        f"Candidates (10 options):\n{cand_block}\n\n"
        "Return ONLY the ID number (1–10) of the best match, then briefly explain why in one sentence."
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        resp   = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw       = resp.choices[0].message.content.strip()
        num_match = re.search(r'\b([1-9]|10)\b', raw)
        obj       = None
        if num_match:
            idx = int(num_match.group(1)) - 1
            if 0 <= idx < len(candidates):
                obj = candidates[idx]
        correct = (obj["id"] == gold_id) if obj else False
        return raw, obj, correct, gold_id, candidates
    except Exception as e:
        return f"LLM 不可用: {e}", None, False, gold_id, candidates


def _llm_classify_ecd(sample: dict) -> tuple[str, int, bool]:
    """ECD：判断 positive 和 negative 哪个更连贯（0=positive 更好，1=negative 更好）。
    label=0 表示 positive 是原始真实序列，negative 是被扰动的。
    模型应判断 positive 更连贯 → 预测 0 = 正确。
    """
    pos_items  = sample.get("positive", {}).get("items", [])
    neg_items  = sample.get("negative", {}).get("items", [])
    theme      = sample.get("positive", {}).get("theme", "(unknown theme)")
    level      = sample.get("level", "?")
    perturb    = sample.get("perturbation_type", "unknown")
    gold_label = sample.get("label", 0)  # 0: positive 是真实的（大多数情况）

    def fmt_items(items):
        return "\n".join(
            f"  {i+1}. {it.get('title','')} | {it.get('culture','')} | "
            f"{it.get('date','')} | {it.get('medium','')}"
            for i, it in enumerate(items)
        )

    prompt = (
        f"You are an expert museum curator evaluating exhibition coherence.\n"
        f"Theme: \"{theme}\"\n\n"
        f"Sequence A:\n{fmt_items(pos_items)}\n\n"
        f"Sequence B:\n{fmt_items(neg_items)}\n\n"
        "Which sequence is more coherent and thematically consistent? "
        "Answer ONLY 'A' or 'B', then explain in one sentence."
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        resp   = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw        = resp.choices[0].message.content.strip()
        choice     = 0 if re.search(r'\bA\b', raw) else 1
        correct    = (choice == gold_label)
        return raw, choice, correct
    except Exception as e:
        return f"LLM 不可用: {e}", -1, False


# ─────────────────────────────────────────────────────────────
# 工具：展品卡片 HTML
# ─────────────────────────────────────────────────────────────
def _obj_card_html(obj: dict, rank: int = 0, highlight: bool = False) -> str:
    img     = obj.get("image_url", "")
    title   = obj.get("title", "Unknown")
    border  = "2px solid #16a34a" if highlight else "1px solid #e0e0e0"
    badge   = f'<div style="position:absolute;top:6px;left:6px;background:#1a1a2e;color:white;border-radius:4px;padding:1px 6px;font-size:0.75em;font-weight:bold">#{rank}</div>' if rank else ""
    gold_badge = '<div style="position:absolute;top:6px;right:6px;background:#16a34a;color:white;border-radius:4px;padding:1px 6px;font-size:0.75em">✓ Gold</div>' if highlight else ""
    img_tag = (
        f'<img src="{img}" loading="lazy" onerror="this.style.display=\'none\'" '
        f'style="width:100%;height:150px;object-fit:cover;display:block;">'
        if img else
        '<div style="width:100%;height:150px;background:#e8e8e8;display:flex;align-items:center;'
        'justify-content:center;color:#aaa;font-size:0.8em">No Image</div>'
    )
    return f"""
    <div style="border:{border};border-radius:8px;overflow:hidden;background:white;
                box-shadow:0 1px 4px rgba(0,0,0,0.08);position:relative;
                transition:box-shadow .2s;">
      {badge}{gold_badge}
      {img_tag}
      <div style="padding:8px 10px;font-size:0.8em;color:#555;">
        <div style="font-weight:700;color:#1a1a2e;margin-bottom:4px;line-height:1.3;font-size:0.9em">{title}</div>
        <div>📍 {obj.get('culture','—')}</div>
        <div>📅 {obj.get('date','—')}</div>
        <div>🖼️ {obj.get('medium','—')[:38]}</div>
      </div>
    </div>"""


def _cards_grid(items: list[dict], gold_id: str = "", ranks: bool = True) -> str:
    cards = [
        _obj_card_html(obj, rank=i+1 if ranks else 0, highlight=(obj.get("id","") == gold_id))
        for i, obj in enumerate(items)
    ]
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));'
        'gap:10px;padding:8px 0;">' + "".join(cards) + "</div>"
    )


# ─────────────────────────────────────────────────────────────
# NiceGUI 界面
# ─────────────────────────────────────────────────────────────
GLOBAL_CSS = """
body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          color: white; padding: 18px 32px; }
.header h1 { margin: 0; font-size: 1.55em; letter-spacing: 0.02em; }
.header p  { margin: 4px 0 0; opacity: 0.7; font-size: 0.88em; }
.task-badge { display:inline-block; padding:2px 10px; border-radius:12px;
              font-size:0.78em; font-weight:600; margin-right:6px; }
.badge-meip { background:#dbeafe; color:#1d4ed8; }
.badge-ecd  { background:#dcfce7; color:#15803d; }
.badge-tes  { background:#fef3c7; color:#b45309; }
"""


def build_ui():
    ui.add_head_html(f"<style>{GLOBAL_CSS}</style>")

    # ── Header ──
    with ui.element("div").classes("header"):
        ui.html("<h1>🏛️ ExhibitionBench — 策展辅助系统</h1>")
        ui.html(
            f"<p>数据: {len(_objects):,} 件展品 · {len(_exhibitions):,} 个展览 · "
            f"MEIP {len(_meip_samples)} / ECD {len(_ecd_samples)} / TES {len(_tes_samples)} 样本 · "
            "Metropolitan Museum of Art + Europeana</p>"
        )

    # ── Tabs ──
    with ui.tabs().classes("w-full bg-white shadow-sm px-4") as tabs:
        tab_meip = ui.tab("🔮 MEIP — 展品补全")
        tab_ecd  = ui.tab("🔍 ECD — 一致性检测")
        tab_tes  = ui.tab("🎨 TES — 主题策展")

    with ui.tab_panels(tabs, value=tab_meip).classes("w-full"):

        # ════════════════════════════════════════════════════════
        # Tab 1 · MEIP  (10-way ranking，中等难度)
        # ════════════════════════════════════════════════════════
        with ui.tab_panel(tab_meip):
            ui.html(
                '<span class="task-badge badge-meip">MEIP</span>'
                '<b>展品补全预测</b> — 给定展览中已有的若干展品，从 10 个候选中预测最佳补充件（10-way ranking）'
            ).classes("text-sm text-gray-600 mb-3 mt-1")

            # 样本选择
            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                meip_opts = [
                    f"[{s['exhibition_theme']}] #{i+1} — {s['id']}"
                    for i, s in enumerate(_meip_samples[:100])
                ] if _meip_samples else ["(无样本)"]
                meip_select = ui.select(
                    label="选择 MEIP 样本",
                    options=meip_opts,
                    value=meip_opts[0],
                ).classes("flex-grow min-w-80")
                rand_meip_btn = ui.button("🎲 随机", color="grey").classes("self-end")
                run_meip_btn  = ui.button("🔮 预测补充展品", color="purple").classes("self-end")

            # 当前样本预览
            meip_ctx_html   = ui.html("").classes("w-full mt-2")
            meip_spinner    = ui.spinner("dots", size="lg").classes("hidden mt-4")

            # 结果区
            with ui.card().classes("w-full mt-3 hidden") as meip_result_card:
                meip_verdict  = ui.html("").classes("w-full mb-2")
                meip_grid     = ui.html("").classes("w-full")
                meip_expl     = ui.markdown("").classes("text-sm text-gray-600 mt-2")

            def _refresh_meip_preview(idx: int):
                if idx >= len(_meip_samples):
                    return
                s       = _meip_samples[idx]
                ctx     = s.get("context", [])
                gold_id = s.get("gold_id", "")
                theme   = s.get("exhibition_theme", "")

                ctx_rows = "".join(
                    f"<tr><td style='padding:3px 8px'><b>{i+1}</b></td>"
                    f"<td style='padding:3px 8px'>{it.get('title','')}</td>"
                    f"<td style='padding:3px 8px;color:#888'>{it.get('culture','')}</td>"
                    f"<td style='padding:3px 8px;color:#888'>{it.get('date','')}</td></tr>"
                    for i, it in enumerate(ctx)
                )
                meip_ctx_html.set_content(
                    f"<div style='background:#f0f4ff;border-radius:8px;padding:12px;font-size:0.88em'>"
                    f"<b>🎨 展览主题:</b> {theme} &nbsp;|&nbsp; "
                    f"<b>上下文展品:</b> {len(ctx)} 件 &nbsp;|&nbsp; "
                    f"<b>候选数量:</b> {len(s.get('candidates',[]))} 件<br><br>"
                    f"<table style='border-collapse:collapse;width:100%'>"
                    f"<thead><tr style='background:#e8edf8'>"
                    f"<th style='padding:3px 8px;text-align:left'>#</th>"
                    f"<th style='padding:3px 8px;text-align:left'>展品</th>"
                    f"<th style='padding:3px 8px;text-align:left'>文化</th>"
                    f"<th style='padding:3px 8px;text-align:left'>年代</th></tr></thead>"
                    f"<tbody>{ctx_rows}</tbody></table></div>"
                )
                meip_result_card.classes(add="hidden")

            def _meip_idx_from_select() -> int:
                val = meip_select.value or ""
                for i, opt in enumerate(meip_opts):
                    if opt == val:
                        return i
                return 0

            def on_meip_select_change():
                _refresh_meip_preview(_meip_idx_from_select())
            meip_select.on("update:model-value", lambda: on_meip_select_change())

            def on_rand_meip():
                idx = random.randrange(min(100, len(_meip_samples)))
                meip_select.set_value(meip_opts[idx])
                _refresh_meip_preview(idx)
            rand_meip_btn.on_click(on_rand_meip)

            async def do_meip_predict():
                idx = _meip_idx_from_select()
                if idx >= len(_meip_samples):
                    ui.notify("无样本", type="warning"); return
                sample = _meip_samples[idx]

                run_meip_btn.disable()
                meip_spinner.classes(remove="hidden")
                meip_result_card.classes(add="hidden")

                try:
                    result = await run.io_bound(_llm_predict_meip, sample)
                    raw, obj, correct, gold_id, candidates = result

                    color   = "#16a34a" if correct else "#dc2626"
                    icon    = "✅" if correct else "❌"
                    verdict = (
                        f"<div style='padding:10px 14px;background:{'#f0fdf4' if correct else '#fef2f2'};"
                        f"border-left:4px solid {color};border-radius:4px;font-size:0.9em'>"
                        f"<b>{icon} {'预测正确！' if correct else '预测错误'}</b> &nbsp;|&nbsp; "
                        f"模型选: <b>{obj.get('title','—') if obj else '—'}</b> &nbsp;|&nbsp; "
                        f"Gold: <b>{gold_id}</b></div>"
                    )
                    meip_verdict.set_content(verdict)
                    meip_grid.set_content(_cards_grid(candidates, gold_id=gold_id, ranks=True))
                    meip_expl.set_content(f"**模型解释:**\n\n{raw}")
                    meip_result_card.classes(remove="hidden")
                except Exception as e:
                    ui.notify(f"错误: {e}", type="negative")
                    log.error(f"MEIP 预测失败: {e}")
                finally:
                    meip_spinner.classes(add="hidden")
                    run_meip_btn.enable()

            run_meip_btn.on_click(do_meip_predict)

            # 初始预览
            if _meip_samples:
                _refresh_meip_preview(0)

        # ════════════════════════════════════════════════════════
        # Tab 2 · ECD  (二分类，最简单)
        # ════════════════════════════════════════════════════════
        with ui.tab_panel(tab_ecd):
            ui.html(
                '<span class="task-badge badge-ecd">ECD</span>'
                '<b>展览一致性检测</b> — 给定两个展品序列（A/B），判断哪个更连贯（二分类）'
            ).classes("text-sm text-gray-600 mb-3 mt-1")

            LEVEL_LABELS = {1: "L1 时代错位", 2: "L2 文化漂移", 3: "L3 主题偏差", 4: "L4 微妙不一致"}
            PERTURB_LABELS = {
                "temporal_anachronism": "时代错位 (L1)",
                "cultural_drift":       "文化漂移 (L2)",
                "thematic_deviation":   "主题偏差 (L3)",
                "subtle_incoherence":   "微妙不一致 (L4)",
            }

            # 过滤 + 样本选择
            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                ecd_level_filter = ui.select(
                    label="扰动级别",
                    options=["全部"] + [f"L{i}" for i in range(1, 5)],
                    value="全部",
                ).classes("w-36")
                ecd_opts = [
                    f"[L{s['level']}·{s['perturbation_type'][:8]}] {s['id']}"
                    for s in _ecd_samples[:200]
                ] if _ecd_samples else ["(无样本)"]
                ecd_select = ui.select(
                    label="选择 ECD 样本",
                    options=ecd_opts,
                    value=ecd_opts[0],
                ).classes("flex-grow min-w-80")
                rand_ecd_btn = ui.button("🎲 随机", color="grey").classes("self-end")
                run_ecd_btn  = ui.button("🔍 判断一致性", color="green").classes("self-end")

            ecd_preview_html = ui.html("").classes("w-full mt-2")
            ecd_spinner      = ui.spinner("dots", size="lg").classes("hidden mt-4")

            with ui.card().classes("w-full mt-3 hidden") as ecd_result_card:
                ecd_verdict_html = ui.html("").classes("w-full mb-3")
                with ui.row().classes("w-full gap-4"):
                    ecd_seq_a = ui.html("").classes("flex-1")
                    ecd_seq_b = ui.html("").classes("flex-1")
                ecd_expl_md = ui.markdown("").classes("text-sm text-gray-600 mt-2")

            _ecd_filtered: list[dict] = list(_ecd_samples[:200])

            def _ecd_idx_from_select() -> int:
                val = ecd_select.value or ""
                for i, opt in enumerate(ecd_opts):
                    if opt == val:
                        return i
                return 0

            def _fmt_ecd_seq(items: list[dict], label: str, highlight_color: str = "") -> str:
                rows = "".join(
                    f"<tr><td style='padding:3px 6px'><b>{i+1}</b></td>"
                    f"<td style='padding:3px 6px'>{it.get('title','')}</td>"
                    f"<td style='padding:3px 6px;color:#888;font-size:0.88em'>{it.get('culture','')}</td>"
                    f"<td style='padding:3px 6px;color:#888;font-size:0.88em'>{it.get('date','')}</td></tr>"
                    for i, it in enumerate(items)
                )
                bg = highlight_color or "#f9fafb"
                return (
                    f"<div style='background:{bg};border-radius:8px;padding:10px 14px;font-size:0.86em'>"
                    f"<div style='font-weight:700;margin-bottom:6px'>{label}</div>"
                    f"<table style='border-collapse:collapse;width:100%'>"
                    f"<thead><tr style='background:#e5e7eb'>"
                    f"<th style='padding:2px 6px;text-align:left'>#</th>"
                    f"<th style='padding:2px 6px;text-align:left'>展品</th>"
                    f"<th style='padding:2px 6px;text-align:left'>文化</th>"
                    f"<th style='padding:2px 6px;text-align:left'>年代</th></tr></thead>"
                    f"<tbody>{rows}</tbody></table></div>"
                )

            def _refresh_ecd_preview(idx: int):
                if idx >= len(_ecd_filtered):
                    return
                s     = _ecd_filtered[idx]
                level = s.get("level", "?")
                pt    = s.get("perturbation_type", "")
                theme = s.get("positive", {}).get("theme", "(unknown)")
                ecd_preview_html.set_content(
                    f"<div style='background:#f0fdf4;border-radius:8px;padding:10px 14px;font-size:0.88em'>"
                    f"<b>🎨 展览主题:</b> {theme} &nbsp;|&nbsp; "
                    f"<b>扰动类型:</b> L{level} · {PERTURB_LABELS.get(pt, pt)} &nbsp;|&nbsp; "
                    f"<b>ID:</b> {s.get('id','')}<br>"
                    f"<span style='color:#666;font-size:0.9em'>A=原始连贯序列，B=被扰动序列（label=0 表示 A 更好）</span></div>"
                )
                ecd_result_card.classes(add="hidden")

            def on_ecd_select_change():
                _refresh_ecd_preview(_ecd_idx_from_select())
            ecd_select.on("update:model-value", lambda: on_ecd_select_change())

            def on_rand_ecd():
                if not _ecd_filtered:
                    return
                idx = random.randrange(len(_ecd_filtered))
                ecd_select.set_value(ecd_opts[idx])
                _refresh_ecd_preview(idx)
            rand_ecd_btn.on_click(on_rand_ecd)

            async def do_ecd_classify():
                idx = _ecd_idx_from_select()
                if idx >= len(_ecd_filtered):
                    ui.notify("无样本", type="warning"); return
                sample = _ecd_filtered[idx]

                run_ecd_btn.disable()
                ecd_spinner.classes(remove="hidden")
                ecd_result_card.classes(add="hidden")

                try:
                    raw, choice, correct = await run.io_bound(_llm_classify_ecd, sample)

                    gold  = sample.get("label", 0)
                    pos_items = sample.get("positive", {}).get("items", [])
                    neg_items = sample.get("negative", {}).get("items", [])

                    color  = "#16a34a" if correct else "#dc2626"
                    icon   = "✅" if correct else "❌"
                    chosen = "A（序列A）" if choice == 0 else "B（序列B）"
                    gold_s = "A（序列A）" if gold == 0 else "B（序列B）"

                    ecd_verdict_html.set_content(
                        f"<div style='padding:10px 14px;background:{'#f0fdf4' if correct else '#fef2f2'};"
                        f"border-left:4px solid {color};border-radius:4px;font-size:0.9em'>"
                        f"<b>{icon} {'判断正确！' if correct else '判断错误'}</b> &nbsp;|&nbsp; "
                        f"模型选: <b>{chosen}</b> &nbsp;|&nbsp; "
                        f"正确答案: <b>{gold_s}</b> （原始序列更连贯）</div>"
                    )
                    # 显示两个序列，正确答案背景高亮
                    a_bg = "#f0fdf4" if gold == 0 else "#fef9f0"
                    b_bg = "#fef9f0" if gold == 0 else "#f0fdf4"
                    ecd_seq_a.set_content(_fmt_ecd_seq(pos_items, "序列 A（Positive）", a_bg))
                    ecd_seq_b.set_content(_fmt_ecd_seq(neg_items, "序列 B（Negative / 扰动）", b_bg))
                    ecd_expl_md.set_content(f"**模型解释:**\n\n{raw}")
                    ecd_result_card.classes(remove="hidden")
                except Exception as e:
                    ui.notify(f"错误: {e}", type="negative")
                    log.error(f"ECD 分类失败: {e}")
                finally:
                    ecd_spinner.classes(add="hidden")
                    run_ecd_btn.enable()

            run_ecd_btn.on_click(do_ecd_classify)

            # 初始预览
            if _ecd_samples:
                _refresh_ecd_preview(0)

        # ════════════════════════════════════════════════════════
        # Tab 3 · TES  (50-way ranking，最难)
        # ════════════════════════════════════════════════════════
        with ui.tab_panel(tab_tes):
            ui.html(
                '<span class="task-badge badge-tes">TES</span>'
                '<b>主题策展检索</b> — 给定展览主题描述，从 50 个候选主题中排序，找到最相关的（50-way ranking）'
            ).classes("text-sm text-gray-600 mb-3 mt-1")

            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                # 既可以选 TES benchmark 样本，也可以自由输入
                tes_opts = [
                    f"[{s['query_theme']}] #{i+1}"
                    for i, s in enumerate(_tes_samples[:80])
                ] if _tes_samples else []
                tes_opts_all = ["✏️ 自定义输入"] + tes_opts
                tes_select = ui.select(
                    label="选择 TES 样本",
                    options=tes_opts_all,
                    value=tes_opts_all[0],
                ).classes("flex-grow min-w-80")
                rand_tes_btn = ui.button("🎲 随机", color="grey").classes("self-end")
                n_slider     = ui.number(label="展示 Top-K", value=10, min=5, max=20, step=1).classes("w-28")
                run_tes_btn  = ui.button("🔍 策展推荐", color="amber").classes("self-end")

            # 自定义输入（仅当选"自定义"时显示）
            tes_custom_row = ui.row().classes("w-full gap-3 hidden")
            with tes_custom_row:
                tes_theme_input = ui.input(
                    label="自定义展览主题",
                    placeholder="例: French Impressionism / Ancient Chinese Bronzes",
                ).classes("flex-grow")

            tes_preview_html = ui.html("").classes("w-full mt-2")
            tes_spinner      = ui.spinner("dots", size="lg").classes("hidden mt-4")

            with ui.card().classes("w-full mt-3 hidden") as tes_result_card:
                tes_status_html = ui.html("").classes("w-full mb-2")
                tes_grid        = ui.html("").classes("w-full")

            def _tes_idx_from_select() -> int:
                val = tes_select.value or ""
                for i, opt in enumerate(tes_opts_all):
                    if opt == val and i > 0:
                        return i - 1  # offset for "自定义" at index 0
                return -1

            def _refresh_tes_preview():
                idx = _tes_idx_from_select()
                is_custom = (idx < 0)
                if is_custom:
                    tes_custom_row.classes(remove="hidden")
                    tes_preview_html.set_content("")
                    return
                tes_custom_row.classes(add="hidden")
                s = _tes_samples[idx]
                theme = s.get("query_theme", "")
                desc  = s.get("query_description", "")
                gold  = s.get("gold_id", s.get("gold_ids", []))
                n_cands = len(s.get("candidates", []))
                tes_preview_html.set_content(
                    f"<div style='background:#fffbeb;border-radius:8px;padding:10px 14px;font-size:0.88em'>"
                    f"<b>🎨 主题:</b> {theme}<br>"
                    f"<b>描述:</b> {str(desc)[:200]}<br>"
                    f"<b>候选池:</b> {n_cands} 个主题 &nbsp;|&nbsp; "
                    f"<b>Gold:</b> {gold if isinstance(gold,str) else ', '.join(gold[:3])}</div>"
                )
                tes_result_card.classes(add="hidden")

            tes_select.on("update:model-value", lambda: _refresh_tes_preview())

            def on_rand_tes():
                if not _tes_samples:
                    return
                idx = random.randrange(min(80, len(_tes_samples)))
                tes_select.set_value(tes_opts_all[idx + 1])
                _refresh_tes_preview()
            rand_tes_btn.on_click(on_rand_tes)

            async def do_tes_search():
                idx  = _tes_idx_from_select()
                k    = int(n_slider.value)

                if idx < 0:
                    # 自定义输入 → BM25 + LLM rerank on objects pool
                    q = tes_theme_input.value.strip()
                    if not q:
                        ui.notify("请输入展览主题", type="warning"); return
                    run_tes_btn.disable()
                    tes_spinner.classes(remove="hidden")
                    tes_result_card.classes(add="hidden")
                    try:
                        candidates_obj = await run.io_bound(_bm25_search, q, 50)
                        results        = await run.io_bound(_llm_rerank_tes, q, candidates_obj, k)
                        gold_id        = ""
                        tes_status_html.set_content(
                            f"<div style='padding:8px 12px;background:#fffbeb;border-radius:4px;font-size:0.9em'>"
                            f"✅ 主题: <b>{q}</b> | 展示 Top-{len(results)} (BM25+LLM)</div>"
                        )
                        tes_grid.set_content(_cards_grid(results, gold_id=gold_id, ranks=True))
                        tes_result_card.classes(remove="hidden")
                    except Exception as e:
                        ui.notify(f"错误: {e}", type="negative")
                    finally:
                        tes_spinner.classes(add="hidden")
                        run_tes_btn.enable()
                    return

                # Benchmark 样本 → 对 TES 候选主题排序
                sample = _tes_samples[idx]
                q      = sample.get("query_theme", "") + " " + sample.get("query_description", "")
                cands  = sample.get("candidates", [])
                gold_id = sample.get("gold_id", "")
                gold_ids = sample.get("gold_ids", [gold_id] if gold_id else [])

                run_tes_btn.disable()
                tes_spinner.classes(remove="hidden")
                tes_result_card.classes(add="hidden")
                try:
                    ranked = await run.io_bound(_llm_rerank_tes, q, cands, k)

                    # 检查 gold 是否在 top-k
                    ranked_ids = [c.get("id","") for c in ranked]
                    hit        = any(gid in ranked_ids for gid in gold_ids)
                    hit_rank   = next((i+1 for i, rid in enumerate(ranked_ids) if rid in gold_ids), None)

                    color = "#16a34a" if hit else "#dc2626"
                    icon  = "✅" if hit else "❌"
                    tes_status_html.set_content(
                        f"<div style='padding:8px 12px;background:{'#f0fdf4' if hit else '#fef2f2'};"
                        f"border-left:4px solid {color};border-radius:4px;font-size:0.9em'>"
                        f"<b>{icon} {'Gold 命中！' if hit else 'Gold 未命中'}</b>"
                        + (f" (排名 #{hit_rank})" if hit_rank else "") +
                        f" &nbsp;|&nbsp; 主题: <b>{sample.get('query_theme','')}</b></div>"
                    )
                    # 构建 TES 候选卡片（theme-level，无图片）
                    cards_html = []
                    for i, c in enumerate(ranked):
                        cid       = c.get("id","")
                        is_gold   = cid in gold_ids
                        border    = "2px solid #16a34a" if is_gold else "1px solid #e0e0e0"
                        badge     = f'<div style="position:absolute;top:6px;left:6px;background:#1a1a2e;color:white;border-radius:4px;padding:1px 6px;font-size:0.75em;font-weight:bold">#{i+1}</div>'
                        gold_b    = '<div style="position:absolute;top:6px;right:6px;background:#16a34a;color:white;border-radius:4px;padding:1px 6px;font-size:0.75em">✓ Gold</div>' if is_gold else ""
                        desc_text = str(c.get("description",""))[:100]
                        cards_html.append(
                            f'<div style="border:{border};border-radius:8px;background:white;'
                            f'padding:12px 14px;position:relative;box-shadow:0 1px 4px rgba(0,0,0,0.07);">'
                            f'{badge}{gold_b}'
                            f'<div style="font-weight:700;color:#1a1a2e;margin:20px 0 6px;font-size:0.92em">'
                            f'{c.get("title", c.get("theme",""))}</div>'
                            f'<div style="font-size:0.8em;color:#666">{desc_text}</div></div>'
                        )
                    tes_grid.set_content(
                        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));'
                        'gap:10px;padding:8px 0;">' + "".join(cards_html) + "</div>"
                    )
                    tes_result_card.classes(remove="hidden")
                except Exception as e:
                    ui.notify(f"错误: {e}", type="negative")
                    log.error(f"TES 搜索失败: {e}")
                finally:
                    tes_spinner.classes(add="hidden")
                    run_tes_btn.enable()

            run_tes_btn.on_click(do_tes_search)

            # 示例快捷按钮（自定义输入）
            ui.label("快速示例（自定义模式）:").classes("text-xs text-gray-400 mt-4")
            with ui.row().classes("gap-2 flex-wrap"):
                for ex in ["French Impressionism", "Ancient Chinese Bronzes",
                           "Japanese Woodblock Prints", "Islamic Calligraphy", "Medieval Tapestry"]:
                    def _set_custom(e=ex):
                        tes_select.set_value("✏️ 自定义输入")
                        tes_custom_row.classes(remove="hidden")
                        tes_theme_input.set_value(e)
                    ui.button(ex, on_click=_set_custom).props("flat dense").classes("text-xs")

            # 初始预览
            _refresh_tes_preview()


# ─────────────────────────────────────────────────────────────
# 反馈记录
# ─────────────────────────────────────────────────────────────
def _log_feedback(item: Optional[str], rating: str, status_label):
    if not item:
        ui.notify("请先选择展品", type="warning"); return
    record = {"timestamp": datetime.now().isoformat(), "item": item, "rating": rating}
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    status_label.set_text(f"✅ 已记录: {rating}")
    ui.notify(f"反馈已保存 {rating}", type="positive")


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    log.info("加载数据 + 预建 BM25 索引...")
    load_data()

    log.info("构建 NiceGUI 界面...")
    build_ui()

    log.info(f"启动 NiceGUI Demo (port={args.port})...")
    ui.run(
        host="0.0.0.0",
        port=args.port,
        title="ExhibitionBench 策展辅助",
        favicon="🏛️",
        reload=False,
        show=False,
    )


if __name__ == "__main__":
    main()
