"""
baselines/gpt4o_zeroshot.py
============================
GPT-4o 零样本 baseline — Week 2 可行性验证。

支持两个任务：
  --task tes   : 给定主题 + 候选列表，让 GPT-4o 排序
  --task meip  : 给定上下文展品 + 候选列表，让 GPT-4o 选最合适的 gold

里程碑判断（可行性）：
  MEIP Acc@1 > 30%（随机基线 = 10%）→ 任务可行，继续推进

使用方法：
  export OPENAI_API_KEY=sk-...

  # MEIP 可行性验证（取前 100 个样本）
  python baselines/gpt4o_zeroshot.py \\
      --task meip \\
      --input data/meip_samples.jsonl \\
      --output results/gpt4o_zeroshot_meip_pred.jsonl \\
      --max-samples 100

  # TES 评测
  python baselines/gpt4o_zeroshot.py \\
      --task tes \\
      --input data/tes_samples.jsonl \\
      --output results/gpt4o_zeroshot_tes_pred.jsonl \\
      --max-samples 50

  # 然后用 benchmark 评测脚本计算指标：
  python benchmark/meip_eval.py eval \\
      --gold data/meip_samples.jsonl \\
      --pred results/gpt4o_zeroshot_meip_pred.jsonl
"""

from __future__ import annotations
import json
import argparse
import logging
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 内部 LiteLLM API 配置
# ─────────────────────────────────────────────────────────────────────────────
INTERNAL_API_BASE = "http://csig.litellm.prod.sgpolaris"
INTERNAL_API_KEY = "sk-TpK0g832p8LbMXTdI_pjkQ"
DEFAULT_MODEL = "gpt-5.2"  # 主力模型
FAST_MODEL = "gemini-3-flash-preview"  # 快速验证用

# ─────────────────────────────────────────────────────────────────────────────
# Prompt 模板
# ─────────────────────────────────────────────────────────────────────────────

MEIP_SYSTEM = """\
You are an expert museum curator with deep knowledge of art history, cultural heritage, and exhibition design.
Your task is to analyze a set of artworks already chosen for an exhibition and identify which additional work \
best fits the exhibition's theme, style, and curatorial logic.
"""

MEIP_USER_TEMPLATE = """\
## Exhibition Context
The following {n_context} artworks have been selected for an exhibition:

{context_block}

## Your Task
Below are {n_candidates} candidate artworks. Rank ALL of them from most to least suitable for inclusion in this exhibition.
Consider: thematic coherence, stylistic consistency, cultural context, historical period, and overall curatorial logic.

## Candidates
{candidates_block}

## Output Format
Return ONLY a JSON array of candidate IDs in order from most to least suitable.
Example: ["id_A", "id_C", "id_B", ...]
"""

TES_SYSTEM = """\
You are an expert museum curator. Your task is to select a cohesive set of artworks for an exhibition based on a given theme.
"""

TES_USER_TEMPLATE = """\
## Exhibition Theme
"{query}"

{description_block}

## Candidate Artworks
Below are {n_candidates} artworks. Select the {k} most suitable ones for this exhibition.
Consider: relevance to the theme, stylistic coherence, cultural diversity, and overall exhibition quality.

{candidates_block}

## Output Format
Return ONLY a JSON array of the selected artwork IDs (exactly {k} IDs).
Example: ["id_A", "id_C", "id_B"]
"""


def _obj_to_text(obj: dict, idx: int | None = None) -> str:
    """将展品 dict 格式化为文字描述。"""
    prefix = f"[{idx}] " if idx is not None else ""
    parts = [f"{prefix}ID: {obj['id']}"]
    if obj.get("title"):
        parts.append(f"Title: {obj['title']}")
    if obj.get("date"):
        parts.append(f"Date: {obj['date']}")
    if obj.get("culture"):
        parts.append(f"Culture: {obj['culture']}")
    if obj.get("medium"):
        parts.append(f"Medium: {obj['medium']}")
    if obj.get("description"):
        desc = obj["description"][:300]  # 截断避免 token 爆炸
        parts.append(f"Description: {desc}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# MEIP baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_meip_zero_shot(
    client: OpenAI,
    sample: dict,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> dict:
    """
    对单个 MEIP 样本调用 GPT-4o，返回 ranked_ids。
    出错时返回随机排序（保证输出完整）。
    """
    context_block = "\n\n".join(
        _obj_to_text(obj, i + 1) for i, obj in enumerate(sample["context"])
    )
    candidates_block = "\n\n".join(
        _obj_to_text(obj) for obj in sample["candidates"]
    )

    user_msg = MEIP_USER_TEMPLATE.format(
        n_context=len(sample["context"]),
        context_block=context_block,
        n_candidates=len(sample["candidates"]),
        candidates_block=candidates_block,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MEIP_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=512,
            # 注意：不传 response_format，Gemini/LiteLLM 不支持此参数
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("模型返回 content 为 None")
        raw = content.strip()

        # 优先用 regex 提取 JSON 数组（兼容各种模型输出格式）
        import re
        arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if arr_match:
            ranked_ids = [str(x) for x in json.loads(arr_match.group())]
        else:
            # 尝试整体解析（模型可能返回 {"ranked": [...]} 或直接 [...]）
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                ranked_ids = [str(x) for x in parsed]
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        ranked_ids = [str(x) for x in v]
                        break
                else:
                    raise ValueError(f"Cannot find list in response: {raw[:200]}")
            else:
                raise ValueError(f"Unexpected response type: {type(parsed)}")

        # 确保所有候选都在结果中（处理模型漏掉的情况）
        candidate_ids = [c["id"] for c in sample["candidates"]]
        ranked_ids = [r for r in ranked_ids if r in candidate_ids]
        missing = [cid for cid in candidate_ids if cid not in ranked_ids]
        ranked_ids += missing  # 漏掉的排到最后

        return {"id": sample["id"], "ranked_ids": ranked_ids, "status": "ok"}

    except Exception as e:
        log.warning(f"[{sample['id']}] GPT-4o 调用失败: {e}")
        # fallback: 随机排序
        import random
        candidate_ids = [c["id"] for c in sample["candidates"]]
        random.shuffle(candidate_ids)
        return {"id": sample["id"], "ranked_ids": candidate_ids, "status": f"error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# TES baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_tes_zero_shot(
    client: OpenAI,
    sample: dict,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> dict:
    """对单个 TES 样本调用 GPT-4o，返回 pred_ids（top-k 选择）。"""
    k = sample.get("k", 10)
    desc_block = f'Description: "{sample["description"]}"' if sample.get("description") else ""
    candidates_block = "\n\n".join(
        _obj_to_text(obj) for obj in sample["candidates"]
    )

    user_msg = TES_USER_TEMPLATE.format(
        query=sample["query"],
        description_block=desc_block,
        n_candidates=len(sample["candidates"]),
        k=k,
        candidates_block=candidates_block,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TES_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=256,
            # 注意：不传 response_format，Gemini/LiteLLM 不支持此参数
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("模型返回 content 为 None")
        raw = content.strip()

        import re
        arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if arr_match:
            pred_ids = [str(x) for x in json.loads(arr_match.group())]
        else:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                pred_ids = [str(x) for x in parsed]
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        pred_ids = [str(x) for x in v]
                        break
                else:
                    raise ValueError(f"Cannot find list in response: {raw[:200]}")
            else:
                raise ValueError(f"Unexpected type: {type(parsed)}")

        candidate_ids = {c["id"] for c in sample["candidates"]}
        pred_ids = [p for p in pred_ids if p in candidate_ids][:k]
        return {"id": sample["id"], "pred_ids": pred_ids, "status": "ok"}

    except Exception as e:
        log.warning(f"[{sample['id']}] GPT-4o TES 调用失败: {e}")
        import random
        fallback = [c["id"] for c in sample["candidates"]]
        random.shuffle(fallback)
        return {"id": sample["id"], "pred_ids": fallback[:k], "status": f"error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="GPT-4o 零样本 baseline")
    parser.add_argument("--task", choices=["tes", "meip"], required=True)
    parser.add_argument("--input", required=True, help="输入样本 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出预测 JSONL 路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--max-samples", type=int, default=None, help="最多处理样本数（调试用）")
    parser.add_argument("--sleep", type=float, default=0.2, help="每次 API 调用后的等待秒数")
    parser.add_argument("--api-key", default=None, help="API key（默认使用内置 key）")
    parser.add_argument("--api-base", default=None, help="API base URL（默认使用内部代理）")
    args = parser.parse_args()

    api_key = args.api_key or INTERNAL_API_KEY
    api_base = args.api_base or INTERNAL_API_BASE

    client = OpenAI(api_key=api_key, base_url=api_base)

    # 读取输入
    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    if args.max_samples:
        samples = samples[:args.max_samples]

    log.info(f"任务: {args.task.upper()}, 样本数: {len(samples)}, 模型: {args.model}")

    # 断点续跑：读取已有输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        log.info(f"已有 {len(done_ids)} 个样本，跳过")

    # 运行
    error_count = 0
    with open(output_path, "a", encoding="utf-8") as out:
        for sample in tqdm(samples, desc=f"{args.task.upper()} zero-shot"):
            if sample["id"] in done_ids:
                continue

            if args.task == "meip":
                result = run_meip_zero_shot(client, sample, model=args.model)
            else:
                result = run_tes_zero_shot(client, sample, model=args.model)

            if "error" in result.get("status", ""):
                error_count += 1

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(args.sleep)

    total = len(samples) - len(done_ids)
    log.info(f"完成: {total} 个样本, 错误: {error_count}")
    log.info(f"输出: {output_path}")
    log.info(f"")
    log.info(f"下一步 — 评测指标：")
    if args.task == "meip":
        log.info(f"  python benchmark/meip_eval.py eval --gold {args.input} --pred {args.output}")
    else:
        log.info(f"  python benchmark/tes_eval.py eval --gold {args.input} --pred {args.output} --k 5 10")
    log.info(f"")
    log.info(f"里程碑判断（MEIP）: Acc@1 > 0.30 → 任务可行 ✓")


if __name__ == "__main__":
    main()
