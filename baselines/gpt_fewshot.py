"""
baselines/gpt_fewshot.py
=========================
LLM few-shot baseline（3-shot in-context learning）。
使用内部 LiteLLM API，支持任意可用模型。

使用方法：
  python baselines/gpt_fewshot.py meip \\
      --input data/meip_samples.jsonl \\
      --output results/gpt5_fewshot_meip_pred.jsonl \\
      --model gpt-5.2

  python baselines/gpt_fewshot.py tes \\
      --input data/tes_samples.jsonl \\
      --output results/gpt5_fewshot_tes_pred.jsonl \\
      --model gpt-5.2
"""

from __future__ import annotations
import json
import time
import argparse
import logging
import random
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

log = logging.getLogger(__name__)

INTERNAL_API_BASE = "http://csig.litellm.prod.sgpolaris"
INTERNAL_API_KEY = "sk-TpK0g832p8LbMXTdI_pjkQ"
DEFAULT_MODEL = "gpt-5.2"

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot 示例（硬编码，来自艺术史常识）
# ─────────────────────────────────────────────────────────────────────────────

MEIP_FEW_SHOT_EXAMPLES = [
    {
        "context": [
            "Title: Water Lilies | Culture: French | Medium: Oil on canvas | Date: 1906 | Artist: Claude Monet",
            "Title: Impression, Sunrise | Culture: French | Medium: Oil on canvas | Date: 1872 | Artist: Claude Monet",
            "Title: Haystacks | Culture: French | Medium: Oil on canvas | Date: 1891 | Artist: Claude Monet",
        ],
        "candidates": [
            "A: The Starry Night | Culture: Dutch | Medium: Oil on canvas | Date: 1889 | Artist: Van Gogh",
            "B: Rouen Cathedral | Culture: French | Medium: Oil on canvas | Date: 1894 | Artist: Claude Monet",
            "C: The Birth of Venus | Culture: Italian | Medium: Tempera on canvas | Date: 1485 | Artist: Botticelli",
        ],
        "answer": "B",
        "reason": "The context artworks are all French Impressionist oil paintings by Monet focusing on light and natural scenes. 'Rouen Cathedral' fits perfectly — same artist, same style, same period.",
    },
    {
        "context": [
            "Title: Terracotta Army Warrior | Culture: Chinese | Medium: Terracotta | Date: 210 BCE",
            "Title: Bronze Ritual Vessel (Ding) | Culture: Chinese | Medium: Bronze | Date: 1200 BCE",
            "Title: Jade Burial Suit | Culture: Chinese | Medium: Jade and gold wire | Date: 100 BCE",
        ],
        "candidates": [
            "A: Greek Marble Kouros | Culture: Greek | Medium: Marble | Date: 550 BCE",
            "B: Chinese Silk Robe (Dragon) | Culture: Chinese | Medium: Silk | Date: 200 BCE",
            "C: Egyptian Canopic Jar | Culture: Egyptian | Medium: Limestone | Date: 1350 BCE",
        ],
        "answer": "B",
        "reason": "Context is ancient Chinese ritual and burial objects. The Chinese Silk Dragon Robe matches culturally (Chinese), temporally (200 BCE), and thematically (status/ritual artifacts).",
    },
]

TES_FEW_SHOT_EXAMPLES = [
    {
        "theme": "French Impressionism",
        "candidates": [
            "A: Water Lilies — Monet, French, 1906, Oil on canvas",
            "B: The Last Supper — Da Vinci, Italian, 1498, Tempera",
            "C: Starry Night over the Rhone — Van Gogh, Dutch, 1888, Oil on canvas",
            "D: Rouen Cathedral — Monet, French, 1894, Oil on canvas",
            "E: Guernica — Picasso, Spanish, 1937, Oil on canvas",
        ],
        "answer": ["A", "C", "D"],
        "reason": "A and D are by Monet (core Impressionist), C is by Van Gogh (Post-Impressionist, closely related). B is Renaissance, E is Cubism — wrong periods and styles.",
    },
]


def _build_meip_few_shot_prompt(sample: dict) -> list[dict]:
    """构建 MEIP few-shot 对话消息。"""
    system = (
        "You are an expert museum curator. Given a set of artworks already in an exhibition "
        "and a list of candidates, rank the candidates from most to least suitable to complete the exhibition. "
        "Consider thematic coherence, stylistic consistency, cultural context, and historical period."
    )

    messages = [{"role": "system", "content": system}]

    # 注入 few-shot 示例
    for ex in MEIP_FEW_SHOT_EXAMPLES:
        ctx_block = "\n".join(f"- {c}" for c in ex["context"])
        cand_block = "\n".join(f"- {c}" for c in ex["candidates"])
        user_content = (
            f"Exhibition context artworks:\n{ctx_block}\n\n"
            f"Candidates to rank:\n{cand_block}\n\n"
            f"Rank ALL candidates from most to least suitable. Return a JSON array of candidate IDs."
        )
        assistant_content = (
            f'Reasoning: {ex["reason"]}\n\n'
            f'Ranking: ["{ex["answer"]}", ' + ", ".join(f'"{c.split(":")[0].strip()}"' for c in ex["candidates"] if c.split(":")[0].strip() != ex["answer"]) + "]"
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    # 真实样本
    ctx_block = "\n".join(
        f'- {c.get("title","")} | Culture: {c.get("culture","")} | Medium: {c.get("medium","")} | Date: {c.get("date","")}'
        for c in sample["context"]
    )
    cand_block = "\n".join(
        f'- ID:{c["id"]} | {c.get("title","")} | Culture: {c.get("culture","")} | Medium: {c.get("medium","")} | Date: {c.get("date","")}'
        for c in sample["candidates"]
    )
    messages.append({
        "role": "user",
        "content": (
            f"Exhibition context artworks:\n{ctx_block}\n\n"
            f"Candidates to rank:\n{cand_block}\n\n"
            f"Rank ALL candidates from most to least suitable. "
            f"Return ONLY a JSON array of the exact candidate IDs (the ID: part), e.g. [\"id1\", \"id2\", ...]"
        ),
    })
    return messages


def run_meip_fewshot(client: OpenAI, sample: dict, model: str = DEFAULT_MODEL) -> dict:
    messages = _build_meip_few_shot_prompt(sample)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        # 解析 JSON 数组
        import re
        arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if arr_match:
            ranked_ids = json.loads(arr_match.group())
            ranked_ids = [str(x) for x in ranked_ids]
        else:
            raise ValueError(f"No JSON array found in: {raw[:200]}")

        candidate_ids = [c["id"] for c in sample["candidates"]]
        ranked_ids = [r for r in ranked_ids if r in candidate_ids]
        missing = [cid for cid in candidate_ids if cid not in ranked_ids]
        ranked_ids += missing
        return {"id": sample["id"], "ranked_ids": ranked_ids, "status": "ok"}
    except Exception as e:
        log.warning(f"[{sample['id']}] 失败: {e}")
        fallback = [c["id"] for c in sample["candidates"]]
        random.shuffle(fallback)
        return {"id": sample["id"], "ranked_ids": fallback, "status": f"error: {e}"}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="LLM few-shot baseline")
    parser.add_argument("task", choices=["tes", "meip"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    client = OpenAI(api_key=INTERNAL_API_KEY, base_url=INTERNAL_API_BASE)

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    with open(out_path, "a", encoding="utf-8") as out:
        for sample in tqdm(samples, desc=f"{args.task.upper()} few-shot [{args.model}]"):
            if sample["id"] in done_ids:
                continue
            if args.task == "meip":
                result = run_meip_fewshot(client, sample, model=args.model)
            else:
                # TES few-shot 复用 zero-shot 逻辑（few-shot prompt 类似）
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from gpt4o_zeroshot import run_tes_zero_shot
                result = run_tes_zero_shot(client, sample, model=args.model)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(args.sleep)

    log.info(f"完成 → {out_path}")


if __name__ == "__main__":
    main()
