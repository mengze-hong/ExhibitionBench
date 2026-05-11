"""
system/nicegui_app.py
=====================
ExhibitionBench 策展辅助系统 — NiceGUI 版本
基于 FastAPI + Vue.js WebSocket，非阻塞 async 架构。

优化点（v2）:
  - BM25 索引启动时预建，搜索只做评分（快 10-50x）
  - 所有 IO（BM25 + LLM）在线程池执行，不阻塞事件循环
  - 图片懒加载 (loading=lazy)
  - 搜索过程显示 spinner，按钮状态正确

用法：
    pip install nicegui rank_bm25 openai
    python system/nicegui_app.py [--port 7861]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
# 数据 + BM25 索引（启动时预建，只建一次）
# ─────────────────────────────────────────────────────────────
_objects: dict[str, dict] = {}
_exhibitions: list[dict]  = []
_bm25_index = None          # BM25Okapi 实例
_bm25_ids: list[str] = []   # 与 _bm25_index 对齐的 id 列表


def load_data():
    global _objects, _exhibitions, _bm25_index, _bm25_ids

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

    # 预建 BM25 索引
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
        log.warning("rank_bm25 未安装，将使用关键词回退")


# ─────────────────────────────────────────────────────────────
# 检索后端（索引已预建，直接评分）
# ─────────────────────────────────────────────────────────────
def _bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """使用预建索引直接评分，无需重建，速度快 10-50x。"""
    if _bm25_index is not None:
        tokens  = query.lower().split()
        scores  = _bm25_index.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [_objects[_bm25_ids[i]] for i in top_idx if _bm25_ids[i] in _objects]

    # 回退：简单关键词匹配
    tokens = query.lower().split()
    scored = []
    for obj in _objects.values():
        text  = f"{obj.get('title','')} {obj.get('description','')} {obj.get('culture','')} {obj.get('medium','')}".lower()
        score = sum(1 for t in tokens if t in text)
        scored.append((score, obj))
    scored.sort(key=lambda x: -x[0])
    return [o for _, o in scored[:top_k]]


def _llm_rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """同步 LLM 重排（在线程池里调用，不阻塞事件循环）。"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        cand_block = "\n".join(
            f"ID:{i+1} | {c.get('title','')} | Culture: {c.get('culture','')} | "
            f"Medium: {c.get('medium','')} | Date: {c.get('date','')}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            f"You are an expert museum curator. Select the top {top_k} artworks "
            f"most suitable for an exhibition on the theme: '{query}'.\n\n"
            f"Candidates:\n{cand_block}\n\n"
            "Return ONLY a JSON array of IDs (1-based), e.g. [3, 1, 7, ...]"
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        import re
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
        log.warning(f"LLM 重排失败: {e}")
    return candidates[:top_k]


def _llm_predict_next(context_lines: list[str]) -> tuple[str, Optional[dict]]:
    """MEIP：同步 LLM 预测（在线程池里调用）。"""
    combined   = " ".join(context_lines)
    candidates = _bm25_search(combined, top_k=50)
    if not candidates:
        return "未找到相关展品", None
    try:
        from openai import OpenAI
        client     = OpenAI(api_key=API_KEY, base_url=API_BASE)
        ctx_block  = "\n".join(f"- {l}" for l in context_lines)
        cand_block = "\n".join(
            f"ID:{i+1} | {c.get('title','')} | Culture: {c.get('culture','')} | "
            f"Medium: {c.get('medium','')} | Date: {c.get('date','')}"
            for i, c in enumerate(candidates[:20])
        )
        prompt = (
            "You are an expert museum curator. Given the following artworks already in an "
            "exhibition, identify the single best artwork from the candidates to complete it.\n\n"
            f"Exhibition artworks:\n{ctx_block}\n\n"
            f"Candidates:\n{cand_block}\n\n"
            "Return ONLY the ID number of the best match, e.g. '5'. Then briefly explain why."
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        import re
        raw       = resp.choices[0].message.content.strip()
        num_match = re.search(r'\b(\d+)\b', raw)
        obj       = None
        if num_match:
            idx = int(num_match.group(1)) - 1
            if 0 <= idx < len(candidates):
                obj = candidates[idx]
        return raw, obj
    except Exception as e:
        obj = candidates[0]
        return f"BM25 fallback (LLM不可用: {e})", obj


# ─────────────────────────────────────────────────────────────
# 工具：构建展品卡片 HTML
# ─────────────────────────────────────────────────────────────
def _cards_html(results: list[dict]) -> str:
    cards = []
    for i, obj in enumerate(results, 1):
        img  = obj.get("image_url", "")
        t    = obj.get("title", "Unknown")
        img_tag = (
            f'<img src="{img}" loading="lazy" '
            f'onerror="this.style.display=\'none\'" '
            f'style="width:100%;height:160px;object-fit:cover;display:block;">'
            if img else
            '<div style="width:100%;height:160px;background:#e8e8e8;'
            'display:flex;align-items:center;justify-content:center;'
            'color:#aaa;font-size:0.8em">No Image</div>'
        )
        cards.append(f"""
        <div class="obj-card">
          {img_tag}
          <div class="meta">
            <div class="title">#{i} {t}</div>
            <div>📍 {obj.get('culture','—')}</div>
            <div>📅 {obj.get('date','—')}</div>
            <div>🖼️ {obj.get('medium','—')[:40]}</div>
          </div>
        </div>""")
    return f'<div class="grid">{"".join(cards)}</div>'


# ─────────────────────────────────────────────────────────────
# NiceGUI 界面
# ─────────────────────────────────────────────────────────────
def build_ui():
    ui.add_head_html("""
    <style>
      body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; }
      .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white; padding: 20px 32px; }
      .header h1 { margin: 0; font-size: 1.6em; }
      .header p  { margin: 4px 0 0; opacity: 0.75; font-size: 0.9em; }
      .obj-card  { border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;
                   background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                   transition: box-shadow .2s; }
      .obj-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
      .obj-card .meta  { padding: 10px 12px; font-size: 0.82em; color: #555; }
      .obj-card .title { font-weight: 700; color: #1a1a2e; margin-bottom: 6px;
                         font-size: 0.95em; line-height: 1.3; }
      .grid { display: grid;
              grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
              gap: 12px; padding: 12px 0; }
    </style>
    """)

    with ui.element("div").classes("header"):
        ui.html("<h1>🏛️ ExhibitionBench — 策展辅助系统</h1>")
        ui.html(
            f"<p>数据: {len(_objects):,} 件展品 · {len(_exhibitions):,} 个展览 · "
            "Metropolitan Museum of Art + Europeana</p>"
        )

    with ui.tabs().classes("w-full bg-white shadow-sm px-4") as tabs:
        tab_tes  = ui.tab("🎨 TES — 主题策展")
        tab_meip = ui.tab("🔮 MEIP — 展品补全")
        tab_stat = ui.tab("📊 数据统计")

    with ui.tab_panels(tabs, value=tab_tes).classes("w-full"):

        # ════════════════════════════════
        # Tab 1: TES
        # ════════════════════════════════
        with ui.tab_panel(tab_tes):
            ui.markdown(
                "**任务**: 输入展览主题，系统从藏品中检索推荐展品。  \n"
                "**适用**: 策展人快速筛选候选展品。"
            ).classes("text-sm text-gray-500 mb-2")

            with ui.row().classes("w-full items-end gap-4 flex-wrap"):
                theme_input = ui.input(
                    label="展览主题",
                    placeholder="例: French Impressionism / Chinese Ceramics",
                ).classes("flex-grow min-w-64")
                n_slider   = ui.number(label="展示数量", value=10, min=5, max=20, step=1).classes("w-28")
                use_llm    = ui.checkbox("GPT-5.2 重排").classes("self-center")
                search_btn = ui.button("🔍 策展推荐", color="blue").classes("self-end")

            with ui.row().classes("items-center gap-2 mt-1"):
                status_label = ui.label("等待输入...").classes("text-sm text-gray-500")
                spinner      = ui.spinner(size="sm").classes("hidden")

            results_area = ui.html("").classes("w-full")

            feedback_row = ui.row().classes("w-full items-center gap-3 mt-2 hidden")
            with feedback_row:
                selected_item = ui.select(label="选择展品反馈", options=[], value=None).classes("flex-grow")
                ui.button("👍 相关",   color="positive",
                          on_click=lambda: _log_feedback(selected_item.value, "👍 relevant",   feedback_status))
                ui.button("👎 不相关", color="negative",
                          on_click=lambda: _log_feedback(selected_item.value, "👎 irrelevant", feedback_status))
            feedback_status = ui.label("").classes("text-sm text-green-600")

            async def do_search():
                q = theme_input.value.strip()
                if not q:
                    ui.notify("请输入展览主题", type="warning")
                    return
                k = int(n_slider.value)

                # 显示加载状态
                search_btn.disable()
                spinner.classes(remove="hidden")
                status_label.set_text("🔍 检索中...")
                results_area.set_content("")

                try:
                    # BM25 在线程池执行（非阻塞）
                    candidates = await run.io_bound(_bm25_search, q, 50)

                    if use_llm.value and candidates:
                        status_label.set_text("🤖 LLM 重排中...")
                        results = await run.io_bound(_llm_rerank, q, candidates, k)
                        method  = "LLM重排"
                    else:
                        results = candidates[:k]
                        method  = "BM25"

                    results_area.set_content(_cards_html(results))
                    status_label.set_text(
                        f"✅ 主题: {q} | 找到 {len(results)} 件推荐展品 ({method})"
                    )
                    item_opts = [
                        f"{obj.get('title','')} ({obj.get('culture','')}, {obj.get('date','')})"
                        for obj in results
                    ]
                    selected_item.set_options(item_opts, value=item_opts[0] if item_opts else None)
                    feedback_row.classes(remove="hidden")
                except Exception as e:
                    status_label.set_text(f"❌ 错误: {e}")
                    log.error(f"搜索失败: {e}")
                finally:
                    spinner.classes(add="hidden")
                    search_btn.enable()

            search_btn.on_click(do_search)

            ui.label("快速示例:").classes("text-xs text-gray-400 mt-3")
            with ui.row().classes("gap-2 flex-wrap"):
                for ex in ["French Impressionism", "Chinese Ceramics",
                           "Japanese Woodblock Prints", "Islamic Calligraphy"]:
                    ui.button(ex, on_click=lambda e=ex: (
                        theme_input.set_value(e),
                        asyncio.ensure_future(do_search()),
                    )).props("flat dense").classes("text-xs")

        # ════════════════════════════════
        # Tab 2: MEIP
        # ════════════════════════════════
        with ui.tab_panel(tab_meip):
            ui.markdown(
                "**任务**: 输入展览中已有的展品，预测最适合补充的下一件。  \n"
                "**格式**: 每行一件，`标题 | 文化 | 年代`"
            ).classes("text-sm text-gray-500 mb-2")

            ctx_input = ui.textarea(
                label="已有展品（每行一件）",
                placeholder=(
                    "Water Lilies | French | 1906\n"
                    "Impression, Sunrise | French | 1872\n"
                    "Haystacks | French | 1891"
                ),
            ).classes("w-full").props("rows=6 outlined")

            with ui.row().classes("items-center gap-2"):
                predict_btn    = ui.button("🔮 预测补充展品", color="purple")
                predict_spinner = ui.spinner(size="sm").classes("hidden")

            with ui.card().classes("w-full mt-3 hidden") as result_card:
                result_md   = ui.markdown("").classes("w-full")
                with ui.row().classes("w-full gap-4 mt-2"):
                    rec_card = ui.card().classes("flex-1 hidden")
                    with rec_card:
                        ui.label("推荐展品").classes("font-bold text-purple-700")
                        rec_title   = ui.label("").classes("font-semibold")
                        rec_details = ui.html("")

            async def do_predict():
                lines = [l.strip() for l in ctx_input.value.strip().split("\n") if l.strip()]
                if not lines:
                    ui.notify("请输入已有展品", type="warning")
                    return
                predict_btn.disable()
                predict_spinner.classes(remove="hidden")
                result_card.classes(remove="hidden")
                rec_card.classes(add="hidden")
                result_md.set_content("⏳ 分析中...")

                try:
                    explanation, obj = await run.io_bound(_llm_predict_next, lines)
                    result_md.set_content(f"**模型分析:**\n\n{explanation}")
                    if obj:
                        rec_card.classes(remove="hidden")
                        rec_title.set_text(obj.get("title", "Unknown"))
                        rec_details.set_content(
                            f"<div style='font-size:0.85em;color:#555;line-height:1.8'>"
                            f"📍 文化: {obj.get('culture','—')}<br>"
                            f"📅 年代: {obj.get('date','—')}<br>"
                            f"🖼️ 媒介: {obj.get('medium','—')}<br>"
                            f"🏛️ 来源: {obj.get('source','—')}<br>"
                            f"<code style='font-size:0.8em'>{obj.get('id','')}</code>"
                            f"</div>"
                        )
                        # 显示图片（懒加载）
                        img_url = obj.get("image_url", "")
                        if img_url:
                            rec_details.set_content(
                                rec_details.content +
                                f'<img src="{img_url}" loading="lazy" '
                                f'style="width:100%;max-width:280px;margin-top:8px;border-radius:4px;" '
                                f'onerror="this.style.display=\'none\'">'
                            )
                except Exception as e:
                    result_md.set_content(f"❌ 错误: {e}")
                    log.error(f"预测失败: {e}")
                finally:
                    predict_spinner.classes(add="hidden")
                    predict_btn.enable()

            predict_btn.on_click(do_predict)

            ui.label("快速示例:").classes("text-xs text-gray-400 mt-3")
            examples_meip = [
                ("莫奈系列",   "Water Lilies | French | 1906\nImpression, Sunrise | French | 1872"),
                ("中国古代",   "Terracotta Warrior | Chinese | 210 BCE\nBronze Ritual Vessel | Chinese | 1200 BCE"),
                ("日本武士",   "Samurai Armor | Japanese | 1600\nKatana Sword | Japanese | 1700"),
            ]
            with ui.row().classes("gap-2 flex-wrap"):
                for label, val in examples_meip:
                    ui.button(label, on_click=lambda v=val: (
                        ctx_input.set_value(v),
                        asyncio.ensure_future(do_predict()),
                    )).props("flat dense").classes("text-xs")

        # ════════════════════════════════
        # Tab 3: 统计
        # ════════════════════════════════
        with ui.tab_panel(tab_stat):
            from collections import Counter
            sources  = Counter(obj.get("source", "unknown") for obj in _objects.values())
            has_img  = sum(1 for obj in _objects.values() if obj.get("image_url"))
            n_themes = len(set(e.get("theme", "") for e in _exhibitions))

            with ui.grid(columns=3).classes("w-full gap-4 mb-6"):
                for icon, lbl, val in [
                    ("🏛️", "展览总数",  f"{len(_exhibitions)}"),
                    ("🖼️", "展品总数",  f"{len(_objects):,}"),
                    ("🎨", "有图片",    f"{has_img:,} ({has_img/max(len(_objects),1)*100:.0f}%)"),
                    ("📌", "主题数",    f"{n_themes}"),
                    ("🌍", "文化覆盖",  "Western / Asian / Islamic / African"),
                    ("📦", "数据来源",  " / ".join(f"{s}:{n}" for s, n in sources.most_common(3))),
                ]:
                    with ui.card().classes("p-4 text-center"):
                        ui.label(icon).classes("text-3xl")
                        ui.label(val).classes("text-xl font-bold text-blue-700 mt-1")
                        ui.label(lbl).classes("text-sm text-gray-500")

            ui.separator()
            ui.label("评测指标").classes("font-bold mt-4")
            ui.markdown("""
| 任务 | 指标 |
|------|------|
| MEIP | Hit@1, Hits@3, Hits@5, MRR |
| TES  | P@k, R@k, F1@k, NDCG@k (k=5,10) |
| ECD  | PairAcc per level (L1–L4) |
            """)


# ─────────────────────────────────────────────────────────────
# 反馈记录
# ─────────────────────────────────────────────────────────────
def _log_feedback(item: Optional[str], rating: str, status_label):
    if not item:
        ui.notify("请先选择展品", type="warning")
        return
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
