"""
baselines/embedding_baseline.py
================================
语义嵌入 baseline（SBERT + 可选 CLIP）。

TES 任务：query embedding vs. 候选展品 description embedding，cosine 排序。
MEIP 任务：上下文展品平均 embedding vs. 候选展品 embedding，最近邻。

使用方法：
  pip install sentence-transformers

  python baselines/embedding_baseline.py tes \\
      --input data/tes_samples.jsonl \\
      --output results/baselines_pred/sbert_tes_pred.jsonl

  python baselines/embedding_baseline.py meip \\
      --input data/meip_samples.jsonl \\
      --output results/baselines_pred/sbert_meip_pred.jsonl
"""

from __future__ import annotations
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from .data_utils import exhibition_to_text, load_objects, meip_candidates, meip_context, tes_query
except ImportError:  # Direct execution: python baselines/embedding_baseline.py
    from data_utils import exhibition_to_text, load_objects, meip_candidates, meip_context, tes_query

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def obj_to_text(obj: dict) -> str:
    parts = [
        obj.get("title", ""),
        obj.get("culture", ""),
        obj.get("medium", ""),
        obj.get("date", ""),
        obj.get("description", "")[:200],
    ]
    return " | ".join(p for p in parts if p)


def load_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    log.info(f"加载模型: {model_name}")
    return SentenceTransformer(model_name)


def run_tes_sbert(samples: list[dict], model) -> list[dict]:
    """TES: query → 最相似候选 top-k。"""
    results = []
    for s in tqdm(samples, desc="SBERT TES"):
        query_text = tes_query(s)
        cand_texts = [exhibition_to_text(c) for c in s["candidates"]]

        query_emb = model.encode(query_text, normalize_embeddings=True)
        cand_embs = model.encode(cand_texts, normalize_embeddings=True, batch_size=32)

        scores = cand_embs @ query_emb  # cosine（已归一化）
        ranked_idx = np.argsort(-scores).tolist()
        k = s.get("k", 10)
        pred_ids = [s["candidates"][i]["id"] for i in ranked_idx[:k]]
        results.append({"id": s["id"], "pred_ids": pred_ids})
    return results


def run_meip_sbert(samples: list[dict], model, objects: dict[str, dict]) -> list[dict]:
    """MEIP: 上下文平均 embedding → 最近邻候选。"""
    results = []
    for s in tqdm(samples, desc="SBERT MEIP"):
        context = meip_context(s, objects)
        candidates = meip_candidates(s, objects)
        ctx_texts = [obj_to_text(c) for c in context]
        cand_texts = [obj_to_text(c) for c in candidates]

        ctx_embs = model.encode(ctx_texts, normalize_embeddings=True, batch_size=32)
        cand_embs = model.encode(cand_texts, normalize_embeddings=True, batch_size=32)

        # 上下文平均嵌入
        query_emb = ctx_embs.mean(axis=0)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

        scores = cand_embs @ query_emb
        ranked_idx = np.argsort(-scores).tolist()
        ranked_ids = [candidates[i]["id"] for i in ranked_idx]
        results.append({"id": s["id"], "ranked_ids": ranked_ids})
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="SBERT 语义嵌入 baseline")
    parser.add_argument("task", choices=["tes", "meip"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--objects", default="data/objects.jsonl",
                        help="Object metadata used to resolve ID-only MEIP samples")
    args = parser.parse_args()

    samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    log.info(f"加载 {len(samples)} 个样本")

    model = load_model(args.model)

    if args.task == "tes":
        results = run_tes_sbert(samples, model)
    else:
        objects = load_objects(Path(args.objects))
        results = run_meip_sbert(samples, model, objects)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"输出: {out_path}")


if __name__ == "__main__":
    main()
