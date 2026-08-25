"""
baselines/bm25_baseline.py
===========================
BM25 检索 baseline。

TES 任务：用展品 title + description 建倒排索引，query → BM25 top-k。
MEIP 任务：将上下文展品拼接成 pseudo-query，BM25 在候选中排序。

使用方法：
  # TES
  python baselines/bm25_baseline.py tes \\
      --input data/tes_samples.jsonl \\
      --output results/bm25_tes_pred.jsonl

  # MEIP
  python baselines/bm25_baseline.py meip \\
      --input data/meip_samples.jsonl \\
      --output results/bm25_meip_pred.jsonl
"""

from __future__ import annotations
import json
import re
import argparse
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm

try:
    from .data_utils import exhibition_to_text, load_objects, meip_candidates, meip_context, tes_query
except ImportError:  # Direct execution: python baselines/bm25_baseline.py
    from data_utils import exhibition_to_text, load_objects, meip_candidates, meip_context, tes_query

log = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """简单 tokenize：小写 + 按非字母数字切分。"""
    return re.findall(r"[a-z0-9]+", text.lower())


def obj_to_doc(obj: dict) -> str:
    """将展品字典拼成检索文档字符串。"""
    parts = [
        obj.get("title", ""),
        obj.get("description", ""),
        obj.get("culture", ""),
        obj.get("medium", ""),
        obj.get("date", ""),
    ]
    return " ".join(p for p in parts if p)


def run_tes_bm25(samples: list[dict]) -> list[dict]:
    """BM25 TES baseline：对每个样本单独建索引（candidates 各不相同）。"""
    results = []
    for s in tqdm(samples, desc="BM25 TES"):
        candidates = s["candidates"]
        docs = [tokenize(exhibition_to_text(c)) for c in candidates]
        bm25 = BM25Okapi(docs)
        query_tokens = tokenize(tes_query(s))
        scores = bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        pred_ids = [candidates[i]["id"] for i in ranked_idx]
        k = s.get("k", 10)
        results.append({"id": s["id"], "pred_ids": pred_ids[:k]})
    return results


def run_meip_bm25(samples: list[dict], objects: dict[str, dict]) -> list[dict]:
    """
    BM25 MEIP baseline：将上下文展品拼成 pseudo-query，在候选中排序。
    伪查询 = 所有上下文展品的 title 拼接。
    """
    results = []
    for s in tqdm(samples, desc="BM25 MEIP"):
        context = meip_context(s, objects)
        context_text = " ".join(c.get("title", "") + " " + c.get("culture", "") for c in context)
        candidates = meip_candidates(s, objects)
        docs = [tokenize(obj_to_doc(c)) for c in candidates]
        bm25 = BM25Okapi(docs)
        query_tokens = tokenize(context_text)
        scores = bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        ranked_ids = [candidates[i]["id"] for i in ranked_idx]
        results.append({"id": s["id"], "ranked_ids": ranked_ids})
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="BM25 baseline")
    parser.add_argument("task", choices=["tes", "meip"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--objects", default="data/objects.jsonl",
                        help="Object metadata used to resolve ID-only MEIP samples")
    args = parser.parse_args()

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    log.info(f"加载 {len(samples)} 个样本")

    if args.task == "tes":
        results = run_tes_bm25(samples)
    else:
        objects = load_objects(Path(args.objects))
        results = run_meip_bm25(samples, objects)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"输出: {out_path}")


if __name__ == "__main__":
    main()
