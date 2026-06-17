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
import os
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

API_KEY = os.environ.get("LLM_API_KEY", "").strip()
API_BASE = os.environ.get("LLM_API_BASE", "http://YOUR_LLM_API_BASE").rstrip("/") + "/v1"
MODEL    = "gpt-5.2"


def _require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError("Missing LLM_API_KEY environment variable")
    return API_KEY

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
_he_data: dict[str, list[dict]] = {}    # task → pre-generated human eval entries
_HE_ANNOTATIONS = LOG_DIR / "human_eval_annotations.jsonl"


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

    # Human Eval pre-generated outputs
    he_dir = BASE / "human_eval"
    for task in ("meip", "ecd", "tes"):
        he_f = he_dir / f"human_eval_{task}.jsonl"
        if he_f.exists():
            with open(he_f, encoding="utf-8") as f:
                _he_data[task] = [json.loads(l) for l in f]
            log.info(f"Human Eval [{task.upper()}]: {len(_he_data[task])} 条 pre-generated outputs")
        else:
            _he_data[task] = []
    log.info(f"Human Eval 总计: {sum(len(v) for v in _he_data.values())} 条")


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
        client     = OpenAI(api_key=_require_api_key(), base_url=API_BASE)
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
        client = OpenAI(api_key=_require_api_key(), base_url=API_BASE)
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
        client = OpenAI(api_key=_require_api_key(), base_url=API_BASE)
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
.badge-he   { background:#f3e8ff; color:#7c3aed; }
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
        tab_he   = ui.tab("👁️ Human Eval")

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

        # ════════════════════════════════════════════════════════
        # Tab 4 · Human Eval  — A/B 盲评
        # ════════════════════════════════════════════════════════
        with ui.tab_panel(tab_he):
            ui.html(
                '<span class="task-badge badge-he">Human Eval</span>'
                '<b>人工盲评</b> — 对三个任务的模型输出进行 A/B 盲比较，发现新 insight'
            ).classes("text-sm text-gray-600 mb-3 mt-1")

            # ── 顶部工具栏：任务切换 + 进度 ──
            with ui.row().classes("w-full items-center gap-4 flex-wrap mb-2"):
                he_task_select = ui.select(
                    label="评测任务",
                    options=["MEIP", "ECD", "TES"],
                    value="MEIP",
                ).classes("w-32")
                he_progress_label = ui.label("进度: 0/0 完成").classes("text-sm text-gray-500")
                he_prev_btn = ui.button("◀ 上一条", color="grey").props("flat dense")
                he_next_btn = ui.button("▶ 下一条", color="grey").props("flat dense")
                he_jump_num = ui.number(label="跳转到第 N 条", value=1, min=1, step=1).classes("w-32")
                he_jump_btn = ui.button("Go", color="grey").props("flat dense")
                with ui.element("div").classes("flex-grow"):
                    pass
                he_export_btn = ui.button("📥 导出注释", color="purple").props("flat dense")

            # ── 提示：如果 human_eval/ 目录为空 ──
            he_no_data_msg = ui.html(
                "<div style='padding:20px;background:#fef3c7;border-radius:8px;color:#92400e;text-align:center'>"
                "⚠️ 未找到预生成的模型输出。请先运行：<br>"
                "<code style='background:#fde68a;padding:2px 8px;border-radius:4px'>"
                "python scripts/generate_human_eval_outputs.py</code>"
                "</div>"
            ).classes("w-full")

            # ── 样本展示区 ──
            he_sample_card = ui.card().classes("w-full hidden")
            with he_sample_card:
                he_sample_meta = ui.html("").classes("w-full mb-3")

                # 上下文区（MEIP/ECD/TES 不同布局）
                he_context_html = ui.html("").classes("w-full mb-3")

                # A/B 两列输出
                with ui.row().classes("w-full gap-4"):
                    with ui.card().classes("flex-1 bg-blue-50"):
                        ui.label("模型 A").classes("text-lg font-bold text-blue-700 mb-2")
                        he_out_a = ui.html("").classes("w-full")
                    with ui.card().classes("flex-1 bg-orange-50"):
                        ui.label("模型 B").classes("text-lg font-bold text-orange-700 mb-2")
                        he_out_b = ui.html("").classes("w-full")

                ui.separator().classes("my-3")

                # 评分区
                with ui.row().classes("w-full items-center gap-4 flex-wrap"):
                    ui.label("你的评判:").classes("font-semibold")
                    he_pref = ui.radio(
                        options=["A 更好", "B 更好", "差不多 / 平局", "两者都很差"],
                        value=None,
                    ).props("inline")

                with ui.row().classes("w-full items-start gap-3 mt-2"):
                    he_comment = ui.textarea(
                        label="💬 评语（为什么？发现了什么 pattern / insight？）",
                        placeholder="例：A 的理由更准确地抓住了主题；B 选了错误的文化背景；"
                                    "两个模型都忽视了时代线索……",
                    ).classes("flex-grow").props("rows=3 outlined")
                    he_save_btn = ui.button("💾 保存评注", color="purple").classes("self-end")

                he_save_status = ui.label("").classes("text-sm text-gray-400 mt-1")

            # ── 汇总统计面板（折叠） ──
            with ui.expansion("📊 已有评注汇总", icon="bar_chart").classes("w-full mt-4"):
                he_stats_html = ui.html("").classes("w-full")
                he_refresh_stats_btn = ui.button("🔄 刷新统计", color="grey").props("flat dense")

            # ──────────────────────────────────────────────────
            # Human Eval 逻辑
            # ──────────────────────────────────────────────────
            _he_state: dict = {
                "task":       "meip",
                "idx":        0,       # 当前第几条
                "entries":    [],      # 当前任务的所有 entries
                "model_a":    "",      # 匿名化前的真实模型名
                "model_b":    "",
                "rng_seed":   42,      # 每个 entry 固定 AB 顺序，用 seed 重现
            }

            def _he_load_annotations() -> list[dict]:
                if not _HE_ANNOTATIONS.exists():
                    return []
                rows = []
                with open(_HE_ANNOTATIONS, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            pass
                return rows

            def _he_annotation_key(task: str, sample_id: str, model_a: str, model_b: str) -> str:
                return f"{task}|{sample_id}|{model_a}|{model_b}"

            def _he_load_annotation_map() -> dict[str, dict]:
                """sample_key → annotation dict"""
                ann_map: dict[str, dict] = {}
                for row in _he_load_annotations():
                    k = _he_annotation_key(
                        row.get("task",""), row.get("sample_id",""),
                        row.get("model_a",""), row.get("model_b","")
                    )
                    ann_map[k] = row
                return ann_map

            def _he_models_for_entry(entry: dict) -> list[str]:
                """从 entry 中找出所有非元数据 key（即模型输出 key）。"""
                skip = {"sample_id", "task", "sample"}
                return [k for k in entry.keys() if k not in skip]

            def _he_ab_pair(entry: dict, rng_seed: int) -> tuple[str, str]:
                """给定一个 entry，随机（但可重现）分配 A/B。"""
                models = _he_models_for_entry(entry)
                if len(models) < 2:
                    return models[0] if models else "", ""
                rng = random.Random(rng_seed)
                pair = rng.sample(models, 2)
                return pair[0], pair[1]

            def _fmt_he_output(entry: dict, model_key: str, task: str) -> str:
                """将某个模型的输出格式化为 HTML。"""
                out = entry.get(model_key, {})
                if not out:
                    return "<i style='color:#aaa'>无输出</i>"
                if "error" in out:
                    return f"<span style='color:#dc2626'>错误: {out['error']}</span>"

                raw = out.get("raw_response", "")
                lines = []

                if task == "meip":
                    choice_num = out.get("choice_num", "?")
                    pred_title = out.get("predicted_title", "—")
                    correct    = out.get("correct")
                    icon       = "✅" if correct else ("❌" if correct is not None else "")
                    lines.append(
                        f"<div style='margin-bottom:8px'>"
                        f"<b>选择:</b> #{choice_num} — <em>{pred_title}</em> {icon}"
                        f"</div>"
                    )
                    # 直接展示 justification（去掉选号部分，截取正文）
                    justification = raw.strip() if raw else ""
                    # 跳过首行如果只是数字/选号
                    import re as _re
                    just_lines = justification.splitlines()
                    if just_lines and _re.match(r'^\s*\d+\.?\s*$', just_lines[0]):
                        justification = "\n".join(just_lines[1:]).strip()
                    if justification:
                        lines.append(
                            f"<div style='color:#374151;font-size:0.85em;"
                            f"background:#f9fafb;border-radius:4px;padding:6px 10px;margin:4px 0;"
                            f"border-left:3px solid #d1d5db;white-space:pre-wrap'>"
                            f"{justification[:400]}"
                            + ("..." if len(justification) > 400 else "")
                            + f"</div>"
                        )
                elif task == "ecd":
                    choice_str = "A" if out.get("choice") == 0 else ("B" if out.get("choice") == 1 else "?")
                    correct    = out.get("correct")
                    icon       = "✅" if correct else ("❌" if correct is not None else "")
                    perturb    = out.get("perturbation", "")
                    lines.append(
                        f"<div style='margin-bottom:8px'>"
                        f"<b>判断:</b> 序列 {choice_str} 更连贯 {icon}"
                        + (f" <span style='color:#888;font-size:0.85em'>扰动: {perturb}</span>" if perturb else "")
                        + "</div>"
                    )
                elif task == "tes":
                    ranked_titles = out.get("ranked_titles", [])
                    hit        = out.get("hit")
                    hit_rank   = out.get("hit_rank")
                    icon       = "✅" if hit else ("❌" if hit is not None else "")
                    lines.append(
                        f"<div style='margin-bottom:8px'>"
                        f"<b>Gold 命中:</b> {icon}"
                        + (f" 排名 #{hit_rank}" if hit_rank else "")
                        + "</div>"
                    )
                    if ranked_titles:
                        items_html = "".join(
                            f"<li style='margin:2px 0'><b>#{i+1}</b> {t}</li>"
                            for i, t in enumerate(ranked_titles[:5])
                        )
                        lines.append(f"<ol style='margin:0;padding-left:18px;font-size:0.88em'>{items_html}</ol>")

                lines.append(
                    f"<details style='margin-top:8px'>"
                    f"<summary style='cursor:pointer;color:#6b7280;font-size:0.85em'>🔍 原始响应</summary>"
                    f"<div style='background:#f9fafb;border-radius:4px;padding:8px;margin-top:4px;"
                    f"font-size:0.82em;white-space:pre-wrap;max-height:200px;overflow-y:auto'>{raw}</div>"
                    f"</details>"
                )
                return "\n".join(lines)

            def _fmt_he_context(entry: dict, task: str) -> str:
                """将样本 context 格式化为 HTML。"""
                sample = entry.get("sample", {})
                if task == "meip":
                    ctx     = sample.get("context", [])
                    cands   = sample.get("candidates", [])
                    theme   = sample.get("exhibition_theme", "")
                    gold_id = sample.get("gold_id", "")

                    def _meip_obj_row(it, label, is_gold=False):
                        img_url = it.get("image_url", "")
                        img_td = (
                            f"<td style='padding:2px 4px'>"
                            f"<img src='{img_url}' "
                            f"style='height:52px;width:52px;object-fit:cover;border-radius:4px' "
                            f"loading='lazy' onerror='this.style.display=\"none\"'>"
                            f"</td>"
                            if img_url else "<td style='width:56px'></td>"
                        )
                        desc = it.get("description", "") or ""
                        desc_html = (
                            f"<details style='display:inline'>"
                            f"<summary style='color:#9ca3af;font-size:0.78em;cursor:pointer'>描述</summary>"
                            f"<div style='font-size:0.8em;color:#4b5563;margin-top:2px'>{desc[:300]}</div>"
                            f"</details>"
                            if desc else ""
                        )
                        gold_badge = " <b style='color:#16a34a;font-size:0.85em'>✓ GOLD</b>" if is_gold else ""
                        medium = str(it.get("medium", "") or "")[:45]
                        row_bg = "background:#f0fdf4;" if is_gold else ""
                        return (
                            f"<tr style='{row_bg}'>"
                            f"{img_td}"
                            f"<td style='padding:2px 8px'><b>{label}</b></td>"
                            f"<td style='padding:2px 8px'>{it.get('title','')}{gold_badge} {desc_html}</td>"
                            f"<td style='padding:2px 8px;color:#555;font-size:0.85em'>{it.get('culture','')}</td>"
                            f"<td style='padding:2px 8px;color:#555;font-size:0.85em'>{it.get('date','')}</td>"
                            f"<td style='padding:2px 8px;color:#888;font-size:0.82em'>{medium}</td>"
                            f"</tr>"
                        )

                    hdr = (
                        "<tr style='border-bottom:1px solid #e5e7eb'>"
                        "<th style='padding:2px 4px;width:56px'>图</th>"
                        "<th style='padding:2px 8px'>#</th>"
                        "<th style='padding:2px 8px;text-align:left'>作品</th>"
                        "<th style='padding:2px 8px;text-align:left'>文化</th>"
                        "<th style='padding:2px 8px;text-align:left'>年代</th>"
                        "<th style='padding:2px 8px;text-align:left'>材质/类型</th>"
                        "</tr>"
                    )

                    ctx_rows = "".join(
                        _meip_obj_row(it, str(i+1)) for i, it in enumerate(ctx)
                    )
                    cand_rows = "".join(
                        _meip_obj_row(c, f"[{i+1}]", is_gold=(c.get("id","") == gold_id))
                        for i, c in enumerate(cands)
                    )
                    return (
                        f"<div style='font-size:0.85em'>"
                        f"<b>🎨 主题:</b> {theme}<br><br>"
                        f"<b>已有展品 ({len(ctx)} 件):</b>"
                        f"<table style='border-collapse:collapse;width:100%;margin:4px 0 10px'>"
                        f"<thead>{hdr}</thead><tbody>{ctx_rows}</tbody></table>"
                        f"<b>候选 ({len(cands)} 件):</b>"
                        f"<table style='border-collapse:collapse;width:100%;margin:4px 0'>"
                        f"<thead>{hdr}</thead><tbody>{cand_rows}</tbody></table></div>"
                    )

                elif task == "ecd":
                    pos   = sample.get("positive", {})
                    neg   = sample.get("negative", {})
                    theme = pos.get("theme", "")
                    level = sample.get("level", "?")
                    perturb = sample.get("perturbation_type", "")
                    gold  = sample.get("label", 0)

                    def _fmt_seq(items, label):
                        rows = "".join(
                            f"<tr><td style='padding:2px 6px'><b>{i+1}</b></td>"
                            f"<td style='padding:2px 6px'>{it.get('title','')}</td>"
                            f"<td style='padding:2px 6px;color:#888;font-size:0.88em'>{it.get('culture','')}</td>"
                            f"<td style='padding:2px 6px;color:#888;font-size:0.88em'>{it.get('date','')}</td></tr>"
                            for i, it in enumerate(items)
                        )
                        return (
                            f"<div style='background:#f9fafb;border-radius:6px;padding:8px;margin-bottom:6px'>"
                            f"<b>{label}</b>"
                            f"<table style='border-collapse:collapse;width:100%;margin-top:4px'>"
                            f"<tbody>{rows}</tbody></table></div>"
                        )
                    gold_label = "序列 A 正确" if gold == 0 else "序列 B 正确"
                    return (
                        f"<div style='font-size:0.85em'>"
                        f"<b>🎨 主题:</b> {theme} &nbsp;|&nbsp; "
                        f"<b>L{level}</b> · {perturb} &nbsp;|&nbsp; "
                        f"<b style='color:#16a34a'>答案: {gold_label}</b><br><br>"
                        + _fmt_seq(pos.get("items",[]), "序列 A（Positive / 真实）")
                        + _fmt_seq(neg.get("items",[]), "序列 B（Negative / 扰动）")
                        + "</div>"
                    )

                elif task == "tes":
                    qt    = sample.get("query_theme", "")
                    qdesc = sample.get("query_description", "")
                    gids  = sample.get("gold_ids", [sample.get("gold_id","")])
                    cands = sample.get("candidates", [])
                    top5  = "".join(
                        f"<li style='font-size:0.85em'><b>#{i+1}</b> {c.get('title', c.get('theme',''))}</li>"
                        for i, c in enumerate(cands[:5])
                    )
                    return (
                        f"<div style='font-size:0.85em'>"
                        f"<b>🎨 查询主题:</b> {qt}<br>"
                        f"<b>描述:</b> {str(qdesc)[:200]}<br>"
                        f"<b>候选池:</b> {len(cands)} 个主题 &nbsp;|&nbsp; "
                        f"<b style='color:#16a34a'>Gold ID(s): {', '.join(str(g) for g in gids)}</b><br>"
                        f"<b>前 5 候选（示例）:</b><ol style='margin:4px 0;padding-left:18px'>{top5}</ol></div>"
                    )
                return ""

            def _he_compute_stats() -> str:
                """计算已有评注的统计汇总 HTML。"""
                rows = _he_load_annotations()
                if not rows:
                    return "<i style='color:#aaa'>暂无评注数据</i>"
                from collections import Counter
                total        = len(rows)
                by_task: Counter = Counter(r.get("task","?") for r in rows)
                by_pref: Counter = Counter(r.get("preference","?") for r in rows)
                # 统计各模型"被选为更好"的次数（去匿名化）
                win_counter: Counter = Counter()
                for r in rows:
                    pref  = r.get("preference","")
                    ma, mb = r.get("model_a",""), r.get("model_b","")
                    if pref == "A 更好" and ma:
                        win_counter[ma] += 1
                    elif pref == "B 更好" and mb:
                        win_counter[mb] += 1
                task_rows = "".join(
                    f"<tr><td style='padding:3px 8px'>{t}</td><td style='padding:3px 8px'>{n}</td></tr>"
                    for t, n in sorted(by_task.items())
                )
                pref_rows = "".join(
                    f"<tr><td style='padding:3px 8px'>{p}</td><td style='padding:3px 8px'>{n}</td></tr>"
                    for p, n in by_pref.most_common()
                )
                win_rows = "".join(
                    f"<tr><td style='padding:3px 8px'>{m}</td><td style='padding:3px 8px'>{n}</td></tr>"
                    for m, n in win_counter.most_common()
                )
                has_comment = sum(1 for r in rows if r.get("comment","").strip())
                return (
                    f"<div style='font-size:0.88em'>"
                    f"<b>总评注数:</b> {total} &nbsp;|&nbsp; <b>有评语:</b> {has_comment}<br><br>"
                    f"<div style='display:flex;gap:24px;flex-wrap:wrap'>"
                    f"<div><b>按任务:</b><table style='border-collapse:collapse'><tbody>{task_rows}</tbody></table></div>"
                    f"<div><b>偏好分布:</b><table style='border-collapse:collapse'><tbody>{pref_rows}</tbody></table></div>"
                    f"<div><b>模型获胜次数:</b><table style='border-collapse:collapse'><tbody>{win_rows}</tbody></table></div>"
                    f"</div></div>"
                )

            def _he_render_current():
                """根据 _he_state 渲染当前样本。"""
                task    = _he_state["task"]
                entries = _he_state["entries"]
                idx     = _he_state["idx"]

                # 进度标签
                total   = len(entries)
                ann_map = _he_load_annotation_map()
                done    = sum(
                    1 for i, e in enumerate(entries)
                    for ma, mb in [_he_ab_pair(e, _he_state["rng_seed"] + i)]
                    if _he_annotation_key(task, e.get("sample_id",""), ma, mb) in ann_map
                )
                he_progress_label.set_text(f"进度: {done}/{total} 完成")

                if not entries:
                    he_sample_card.classes(add="hidden")
                    he_no_data_msg.classes(remove="hidden")
                    return

                he_no_data_msg.classes(add="hidden")
                he_sample_card.classes(remove="hidden")

                entry   = entries[idx]
                sid     = entry.get("sample_id", f"{task}_{idx}")
                ma, mb  = _he_ab_pair(entry, _he_state["rng_seed"] + idx)
                _he_state["model_a"] = ma
                _he_state["model_b"] = mb

                # 检查是否已有评注
                key = _he_annotation_key(task, sid, ma, mb)
                existing = ann_map.get(key, {})

                # 元信息
                he_sample_meta.set_content(
                    f"<div style='background:#f5f3ff;border-radius:8px;padding:8px 14px;"
                    f"font-size:0.85em;display:flex;gap:16px;flex-wrap:wrap'>"
                    f"<span><b>任务:</b> {task.upper()}</span>"
                    f"<span><b>样本:</b> {sid}</span>"
                    f"<span><b>条数:</b> {idx+1}/{total}</span>"
                    + (f"<span style='color:#16a34a'><b>✓ 已评注</b></span>" if existing else "")
                    + "</div>"
                )

                # 上下文
                he_context_html.set_content(_fmt_he_context(entry, task))

                # A/B 输出
                he_out_a.set_content(_fmt_he_output(entry, ma, task))
                he_out_b.set_content(_fmt_he_output(entry, mb, task))

                # 恢复已有评注
                if existing:
                    he_pref.set_value(existing.get("preference"))
                    he_comment.set_value(existing.get("comment", ""))
                    he_save_status.set_text(f"✅ 已于 {existing.get('ts','?')} 保存")
                else:
                    he_pref.set_value(None)
                    he_comment.set_value("")
                    he_save_status.set_text("")

            def _he_reload_task(task_upper: str):
                task = task_upper.lower()
                entries = _he_data.get(task, [])
                _he_state["task"]    = task
                _he_state["entries"] = entries
                _he_state["idx"]     = 0
                he_jump_num.set_value(1)
                he_jump_num.props(f"max={max(1, len(entries))}")
                _he_render_current()

            he_task_select.on("update:model-value", lambda: _he_reload_task(he_task_select.value))

            def he_go_prev():
                if _he_state["idx"] > 0:
                    _he_state["idx"] -= 1
                    he_jump_num.set_value(_he_state["idx"] + 1)
                    _he_render_current()
                else:
                    ui.notify("已是第一条", type="info")
            he_prev_btn.on_click(he_go_prev)

            def he_go_next():
                if _he_state["idx"] < len(_he_state["entries"]) - 1:
                    _he_state["idx"] += 1
                    he_jump_num.set_value(_he_state["idx"] + 1)
                    _he_render_current()
                else:
                    ui.notify("已是最后一条", type="info")
            he_next_btn.on_click(he_go_next)

            def he_go_jump():
                n = int(he_jump_num.value or 1)
                total = len(_he_state["entries"])
                idx   = max(0, min(n - 1, total - 1))
                _he_state["idx"] = idx
                he_jump_num.set_value(idx + 1)
                _he_render_current()
            he_jump_btn.on_click(he_go_jump)

            def he_save_annotation():
                task   = _he_state["task"]
                idx    = _he_state["idx"]
                entries = _he_state["entries"]
                if not entries or idx >= len(entries):
                    ui.notify("无样本", type="warning"); return
                entry  = entries[idx]
                sid    = entry.get("sample_id", f"{task}_{idx}")
                ma     = _he_state["model_a"]
                mb     = _he_state["model_b"]
                pref   = he_pref.value
                comment = he_comment.value.strip()

                if not pref:
                    ui.notify("请先选择偏好 (A/B/平局)", type="warning"); return

                # 从 entry 提取各模型的 gold 准确率字段
                out_a = entry.get(ma, {})
                out_b = entry.get(mb, {})
                # MEIP/ECD: "correct" bool; TES: "hit" bool
                def _gold_correct(out_dict):
                    if "correct" in out_dict:
                        return out_dict["correct"]
                    if "hit" in out_dict:
                        return out_dict["hit"]
                    return None

                record = {
                    "ts":             datetime.now().isoformat(),
                    "task":           task,
                    "sample_id":      sid,
                    "model_a":        ma,
                    "model_b":        mb,
                    "preference":     pref,
                    "comment":        comment,
                    "idx":            idx,
                    "gold_correct_a": _gold_correct(out_a),
                    "gold_correct_b": _gold_correct(out_b),
                }
                # 追加到日志（允许重复，后处理时取最新）
                with open(_HE_ANNOTATIONS, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                he_save_status.set_text(f"✅ 已于 {record['ts'][:19]} 保存")
                ui.notify("✅ 评注已保存", type="positive")
                # 自动跳到下一条
                if idx < len(entries) - 1:
                    _he_state["idx"] += 1
                    he_jump_num.set_value(_he_state["idx"] + 1)
                    _he_render_current()
            he_save_btn.on_click(he_save_annotation)

            def he_refresh_stats():
                he_stats_html.set_content(_he_compute_stats())
            he_refresh_stats_btn.on_click(he_refresh_stats)

            def he_export():
                rows = _he_load_annotations()
                if not rows:
                    ui.notify("暂无评注", type="warning"); return
                # 导出为 TSV 到剪贴板（通过 ui.notify 显示文件路径）
                out_path = LOG_DIR / f"human_eval_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                ui.notify(f"✅ 已导出 {len(rows)} 条 → {out_path}", type="positive", timeout=6000)
            he_export_btn.on_click(he_export)

            # 初始化
            _he_reload_task("MEIP")


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
