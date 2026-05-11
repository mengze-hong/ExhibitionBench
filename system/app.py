"""
system/app.py
=============
ExhibitionBench 策展辅助系统 Gradio Demo (Week 5)。

功能:
  1. TES 模式：输入主题 → 从候选池检索并展示推荐展品（网格），支持 LLM 排序
  2. MEIP 模式：输入已有展品 → 预测最佳补充展品
  3. 人在回路：用户可对推荐结果点赞/踩，反馈记录到 logs/feedback.jsonl

使用方法:
  pip install gradio sentence-transformers openai
  python system/app.py [--port 7860]
"""

from __future__ import annotations
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime

import gradio as gr

log = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)

FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

_objects: dict[str, dict] = {}
_exhibitions: list[dict] = []


def load_data():
    global _objects, _exhibitions
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
    log.info(f"加载 {len(_objects)} 件展品, {len(_exhibitions)} 个展览")


# ─────────────────────────────────────────────────────────────────────────────
# 检索后端（BM25 + 可选 LLM 重排）
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """BM25 简化实现：基于 title + description + culture 的 TF-IDF 近似。"""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        # Fallback: simple keyword matching
        tokens = query.lower().split()
        scored = []
        for obj in _objects.values():
            text = f"{obj.get('title','')} {obj.get('description','')} {obj.get('culture','')} {obj.get('medium','')}".lower()
            score = sum(1 for t in tokens if t in text)
            scored.append((score, obj))
        scored.sort(key=lambda x: -x[0])
        return [obj for _, obj in scored[:top_k]]

    corpus = []
    ids = []
    for oid, obj in _objects.items():
        text = f"{obj.get('title','')} {obj.get('description','')} {obj.get('culture','')} {obj.get('medium','')}"
        corpus.append(text.lower().split())
        ids.append(oid)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [_objects[ids[i]] for i in top_indices]


def _llm_rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """调用内部 LiteLLM API 对候选展品重排。"""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key="sk-TpK0g832p8LbMXTdI_pjkQ",
            base_url="http://csig.litellm.prod.sgpolaris/v1"
        )
        cand_block = "\n".join(
            f"ID:{i+1} | {c.get('title','')} | Culture: {c.get('culture','')} | "
            f"Medium: {c.get('medium','')} | Date: {c.get('date','')}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            f"You are an expert museum curator. Select the top {top_k} artworks "
            f"most suitable for an exhibition on the theme: '{query}'.\n\n"
            f"Candidates:\n{cand_block}\n\n"
            f"Return ONLY a JSON array of IDs (1-based), e.g. [3, 1, 7, ...]"
        )
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        import re
        raw = resp.choices[0].message.content
        arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if arr_match:
            ranked_nums = json.loads(arr_match.group())
            ranked_nums = [int(n) - 1 for n in ranked_nums if 1 <= int(n) <= len(candidates)]
            result = [candidates[i] for i in ranked_nums if i < len(candidates)]
            seen = {c["id"] for c in result}
            for c in candidates:
                if c["id"] not in seen:
                    result.append(c)
            return result[:top_k]
    except Exception as e:
        log.warning(f"LLM 重排失败: {e}")
    return candidates[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI 逻辑
# ─────────────────────────────────────────────────────────────────────────────

_last_results: list[dict] = []


def search_exhibitions(theme: str, use_llm: bool, n_results: int) -> tuple:
    """TES 模式：根据主题检索推荐展品。"""
    global _last_results
    if not theme.strip():
        return "请输入展览主题", "", []

    candidates = _bm25_search(theme, top_k=50)
    if use_llm and candidates:
        results = _llm_rerank(theme, candidates, top_k=n_results)
    else:
        results = candidates[:n_results]

    _last_results = results

    # 构建 HTML 展示
    cards = []
    for i, obj in enumerate(results, 1):
        img_url = obj.get("image_url", "")
        title = obj.get("title", "Unknown")
        culture = obj.get("culture", "")
        date = obj.get("date", "")
        medium = obj.get("medium", "")
        source = obj.get("source", "")
        obj_id = obj.get("id", "")

        img_html = (
            f'<img src="{img_url}" style="width:100%;height:180px;object-fit:cover;border-radius:4px;" '
            f'onerror="this.style.display=\'none\'">'
            if img_url else
            '<div style="width:100%;height:180px;background:#ddd;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#888">No Image</div>'
        )
        card = f"""
<div style="border:1px solid #e0e0e0;border-radius:8px;padding:12px;margin:8px;background:white;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
  <div style="font-weight:bold;color:#1a1a2e;margin-bottom:8px">#{i} {title}</div>
  {img_html}
  <div style="font-size:0.85em;color:#555;margin-top:8px">
    <b>文化:</b> {culture}<br>
    <b>年代:</b> {date}<br>
    <b>媒介:</b> {medium}<br>
    <b>来源:</b> {source}<br>
    <b>ID:</b> <code style="font-size:0.8em">{obj_id}</code>
  </div>
</div>"""
        cards.append(card)

    grid_html = f"""
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;padding:8px;">
  {''.join(cards)}
</div>"""

    summary = (
        f"✅ 主题: **{theme}** | "
        f"找到 **{len(results)}** 件推荐展品"
        + (" (LLM重排)" if use_llm else " (BM25检索)")
    )
    return summary, grid_html, [f"{obj.get('title','')} ({obj.get('culture','')}, {obj.get('date','')})" for obj in results]


def predict_next_item(context_text: str) -> str:
    """MEIP 模式：根据已有展品描述预测补充展品。"""
    if not context_text.strip():
        return "请输入已有展品描述（每行一件）"

    context_lines = [l.strip() for l in context_text.strip().split("\n") if l.strip()]

    # 用上下文描述搜索相似展品
    combined_query = " ".join(context_lines)
    candidates = _bm25_search(combined_query, top_k=50)

    if not candidates:
        return "未找到相关展品"

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key="sk-TpK0g832p8LbMXTdI_pjkQ",
            base_url="http://csig.litellm.prod.sgpolaris/v1"
        )
        ctx_block = "\n".join(f"- {l}" for l in context_lines)
        cand_block = "\n".join(
            f"ID:{i+1} | {c.get('title','')} | Culture: {c.get('culture','')} | "
            f"Medium: {c.get('medium','')} | Date: {c.get('date','')}"
            for i, c in enumerate(candidates[:20])
        )
        prompt = (
            "You are an expert museum curator. Given the following artworks already in an exhibition, "
            "identify the single best artwork from the candidates to complete the exhibition.\n\n"
            f"Exhibition artworks:\n{ctx_block}\n\n"
            f"Candidates:\n{cand_block}\n\n"
            "Return ONLY the ID number of the best match, e.g. '5'. Then briefly explain why."
        )
        resp = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        import re
        num_match = re.search(r'\b(\d+)\b', raw)
        result_lines = [f"**模型响应:**\n{raw}\n"]
        if num_match:
            idx = int(num_match.group(1)) - 1
            if 0 <= idx < len(candidates):
                obj = candidates[idx]
                result_lines.append(
                    f"\n**推荐展品:**\n"
                    f"- 标题: {obj.get('title','')}\n"
                    f"- 文化: {obj.get('culture','')}\n"
                    f"- 年代: {obj.get('date','')}\n"
                    f"- 媒介: {obj.get('medium','')}\n"
                    f"- ID: `{obj.get('id','')}`"
                )
        return "\n".join(result_lines)
    except Exception as e:
        # Fallback: return top BM25 result
        obj = candidates[0]
        return (
            f"**BM25 推荐 (LLM不可用: {e}):**\n"
            f"- 标题: {obj.get('title','')}\n"
            f"- 文化: {obj.get('culture','')}\n"
            f"- 年代: {obj.get('date','')}\n"
            f"- 媒介: {obj.get('medium','')}"
        )


def log_feedback(item_info: str, rating: str) -> str:
    """记录用户反馈到日志。"""
    if not item_info:
        return "请先搜索展品"
    record = {
        "timestamp": datetime.now().isoformat(),
        "item": item_info,
        "rating": rating,
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"✅ 已记录反馈: {rating} for '{item_info[:50]}...'"


# ─────────────────────────────────────────────────────────────────────────────
# 构建 Gradio 界面
# ─────────────────────────────────────────────────────────────────────────────

_CUSTOM_CSS = """
.header-title { font-size: 1.8em; font-weight: bold; color: #1a1a2e; text-align: center; padding: 16px 0; }
.task-description { color: #555; font-size: 0.95em; margin-bottom: 12px; }
"""

def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="ExhibitionBench — 策展辅助系统",
    ) as demo:
        gr.HTML('<div class="header-title">🏛️ ExhibitionBench — 博物馆策展辅助系统</div>')
        gr.Markdown(
            "**ExhibitionBench** 演示系统 | 数据来源: Metropolitan Museum of Art + Europeana"
        )

        with gr.Tabs():
            # ── Tab 1: TES (Theme-based Exhibition Selection) ──
            with gr.TabItem("🎨 TES — 主题策展"):
                gr.Markdown(
                    "**任务**: 输入展览主题，系统从 1500+ 件藏品中检索并推荐最合适的展品组合。\n\n"
                    "**适用场景**: 策展人快速筛选某一主题的候选展品。",
                    elem_classes="task-description"
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        theme_input = gr.Textbox(
                            label="展览主题",
                            placeholder="例如: French Impressionism / Chinese Ceramics / Medieval Art",
                            lines=1,
                        )
                    with gr.Column(scale=1):
                        n_results = gr.Slider(5, 20, value=10, step=1, label="展示数量")
                        use_llm = gr.Checkbox(label="启用 GPT-5.2 重排（更准确，需联网）", value=False)

                search_btn = gr.Button("🔍 开始策展推荐", variant="primary", size="lg")

                status_md = gr.Markdown("等待输入...")
                results_html = gr.HTML()
                result_list = gr.Dropdown(
                    label="选择展品以反馈",
                    choices=[],
                    interactive=True,
                )

                with gr.Row():
                    thumbs_up = gr.Button("👍 相关", variant="secondary")
                    thumbs_down = gr.Button("👎 不相关", variant="secondary")
                feedback_status = gr.Textbox(label="反馈状态", interactive=False)

                # 事件绑定
                search_btn.click(
                    fn=search_exhibitions,
                    inputs=[theme_input, use_llm, n_results],
                    outputs=[status_md, results_html, result_list],
                )
                thumbs_up.click(
                    fn=lambda item: log_feedback(item, "👍 relevant"),
                    inputs=[result_list],
                    outputs=[feedback_status],
                )
                thumbs_down.click(
                    fn=lambda item: log_feedback(item, "👎 irrelevant"),
                    inputs=[result_list],
                    outputs=[feedback_status],
                )

                # 示例
                gr.Examples(
                    examples=[
                        ["French Impressionism", False, 10],
                        ["Ancient Chinese Ceramics", False, 8],
                        ["Medieval Tapestry", True, 6],
                        ["Japanese Woodblock Prints", False, 10],
                        ["Islamic Calligraphy", True, 8],
                    ],
                    inputs=[theme_input, use_llm, n_results],
                    label="示例主题",
                )

            # ── Tab 2: MEIP (Masked Exhibition Item Prediction) ──
            with gr.TabItem("🔮 MEIP — 展品补全"):
                gr.Markdown(
                    "**任务**: 输入展览中已有的几件展品，系统预测最适合补充的下一件展品。\n\n"
                    "**适用场景**: 策展人在已有部分展品时，寻找最佳配套展品。",
                    elem_classes="task-description"
                )

                context_input = gr.Textbox(
                    label="已有展品（每行一件，格式: 标题 | 文化 | 年代）",
                    placeholder=(
                        "Water Lilies | French | 1906\n"
                        "Impression, Sunrise | French | 1872\n"
                        "Haystacks | French | 1891"
                    ),
                    lines=6,
                )
                predict_btn = gr.Button("🔮 预测补充展品", variant="primary", size="lg")
                prediction_output = gr.Markdown("等待输入...")

                predict_btn.click(
                    fn=predict_next_item,
                    inputs=[context_input],
                    outputs=[prediction_output],
                )

                gr.Examples(
                    examples=[
                        ["Water Lilies | French | 1906\nImpression, Sunrise | French | 1872\nHaystacks | French | 1891"],
                        ["Terracotta Warrior | Chinese | 210 BCE\nBronze Ritual Vessel | Chinese | 1200 BCE\nJade Burial Suit | Chinese | 100 BCE"],
                        ["Samurai Armor | Japanese | 1600\nKatana Sword | Japanese | 1700\nWoodblock Print | Japanese | 1830"],
                    ],
                    inputs=[context_input],
                    label="示例上下文",
                )

            # ── Tab 3: 数据集概览 ──
            with gr.TabItem("📊 数据集概览"):
                gr.Markdown("### ExhibitionBench 数据统计")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            f"""
**数据来源**: Met Museum Open Access API + Europeana Record Search API

| 指标 | 数量 |
|------|------|
| 展览数量 | {len(_exhibitions)} 个 |
| 展品总数 | {len(_objects)} 件 |
| MEIP 样本 | ~500 个 |
| TES 样本 | ~54 个 |

**文化覆盖**: 西方 / 亚洲 / 非洲 / 中东 / 前哥伦布 / 其他

**评测指标**:
- MEIP: Acc@1, Hits@3, Hits@5, MRR
- TES: P@k, R@k, F1@k, NDCG@k (k=5,10)
                            """
                        )

                refresh_btn = gr.Button("🔄 刷新统计")
                stats_output = gr.Markdown()

                def refresh_stats():
                    from collections import Counter
                    sources = Counter(obj.get("source", "unknown") for obj in _objects.values())
                    cultures_raw = [obj.get("culture", "") for obj in _objects.values()]
                    has_image = sum(1 for obj in _objects.values() if obj.get("image_url"))
                    return (
                        f"**展品来源**: " + " | ".join(f"{s}: {n}" for s, n in sources.items()) + "\n\n"
                        f"**有图片**: {has_image}/{len(_objects)} ({has_image/len(_objects)*100:.1f}%)\n\n"
                        f"**展览主题数**: {len(set(e.get('theme','') for e in _exhibitions))}"
                    )

                refresh_btn.click(fn=refresh_stats, outputs=stats_output)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="ExhibitionBench Gradio Demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公共分享链接")
    args = parser.parse_args()

    log.info("加载数据...")
    load_data()

    log.info(f"启动 Gradio Demo (port={args.port})...")
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
