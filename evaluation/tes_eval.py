"""
benchmark/tes_eval.py
=====================
TES (Theme-based Exhibition Selection) 评测 harness。

任务定义：
  给定主题查询 q，从候选展品池 C（|C|=50）中选出 top-k 展品作为展览。
  gold set G ⊆ C 为真实策展人选择的展品。

评估指标：
  - Precision@k     精确率（命中 gold 的比例）
  - Recall@k        召回率
  - F1@k
  - NDCG@k          归一化折损累计增益
  - ILS (Intra-List Similarity)  集合内嵌入相似度均值（多样性反指标，越低越多样）
  - Diversity@k = 1 - ILS@k
  - Coherence@k     集合内平均余弦相似度（主题连贯性正指标）

数据格式（JSONL，每行一个样本）：
  {
    "id": "tes_001",
    "query": "Impressionism landscapes",
    "candidates": [{"id": "obj1", "title": "...", "description": "..."}, ...],  // len=50
    "gold_ids": ["obj3", "obj17", "obj29"],   // k 不固定，通常 5-20
    "k": 5   // 评测时取 top-k，可覆盖
  }

使用方法：
  # 从文件评测某个 baseline 的预测结果
  python benchmark/tes_eval.py \\
      --gold data/tes_samples.jsonl \\
      --pred results/bm25_tes_pred.jsonl \\
      --k 5 10
"""

from __future__ import annotations
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算（纯数值，不依赖 embeddings）
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(pred_ids: list[str], gold_ids: list[str], k: int) -> float:
    """P@k：前 k 个预测中命中 gold 的比例。"""
    top_k = set(pred_ids[:k])
    hits = len(top_k & set(gold_ids))
    return hits / k


def recall_at_k(pred_ids: list[str], gold_ids: list[str], k: int) -> float:
    """R@k：前 k 个预测中命中 gold 的占 gold 总数的比例。"""
    if not gold_ids:
        return 0.0
    top_k = set(pred_ids[:k])
    hits = len(top_k & set(gold_ids))
    return hits / len(gold_ids)


def f1_at_k(pred_ids: list[str], gold_ids: list[str], k: int) -> float:
    p = precision_at_k(pred_ids, gold_ids, k)
    r = recall_at_k(pred_ids, gold_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(pred_ids: list[str], gold_ids: list[str], k: int) -> float:
    """NDCG@k：二值相关性（在 gold 中=1，否则=0）。"""
    gold_set = set(gold_ids)
    dcg = 0.0
    for i, pid in enumerate(pred_ids[:k], start=1):
        if pid in gold_set:
            dcg += 1.0 / math.log2(i + 1)
    # 理想 DCG：gold 排在最前
    ideal_len = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_set_metrics(
    pred_ids: list[str],
    gold_ids: list[str],
    k: int,
    embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """
    计算单个样本在 top-k 下的所有指标。

    embeddings: {obj_id: np.ndarray(dim,)} 可选。
      若提供，额外计算 ILS（多样性）和 Coherence。
    """
    result = {
        f"P@{k}": precision_at_k(pred_ids, gold_ids, k),
        f"R@{k}": recall_at_k(pred_ids, gold_ids, k),
        f"F1@{k}": f1_at_k(pred_ids, gold_ids, k),
        f"NDCG@{k}": ndcg_at_k(pred_ids, gold_ids, k),
    }

    if embeddings is not None:
        top_k_ids = pred_ids[:k]
        vecs = [embeddings[i] for i in top_k_ids if i in embeddings]
        if len(vecs) >= 2:
            vecs_arr = np.stack(vecs)
            # 归一化
            norms = np.linalg.norm(vecs_arr, axis=1, keepdims=True)
            vecs_norm = vecs_arr / (norms + 1e-9)
            sim_matrix = vecs_norm @ vecs_norm.T
            # 取上三角（不含对角线）
            n = len(vecs)
            idx = np.triu_indices(n, k=1)
            pairwise_sims = sim_matrix[idx]
            ils = float(pairwise_sims.mean())
            result[f"ILS@{k}"] = ils
            result[f"Diversity@{k}"] = 1.0 - ils
        else:
            result[f"ILS@{k}"] = 0.0
            result[f"Diversity@{k}"] = 1.0

    return result


def aggregate_metrics(per_sample: list[dict[str, float]]) -> dict[str, float]:
    """对多个样本的指标取均值。"""
    if not per_sample:
        return {}
    all_keys = set().union(*per_sample)
    return {k: float(np.mean([s.get(k, 0.0) for s in per_sample])) for k in all_keys}


# ─────────────────────────────────────────────────────────────────────────────
# 数据构建工具：从 exhibitions.jsonl 生成 TES 样本
# ─────────────────────────────────────────────────────────────────────────────

def build_tes_samples(
    exhibitions_path: Path,
    objects_path: Path,
    output_path: Path,
    neg_pool_size: int = 50,
    min_gold: int = 5,
    max_gold: int = 30,
    seed: int = 42,
) -> int:
    """
    从 exhibitions.jsonl 构建 TES 评测样本。

    策略：
    - 每个展览作为一个 TES 样本：theme = 展览标题，gold = 展览所有展品
    - 从其他展览随机抽取展品填满候选池到 neg_pool_size
    - 过滤展品数量 < min_gold 或 > max_gold 的展览

    输出到 output_path（JSONL）。
    返回生成样本数。
    """
    import random
    random.seed(seed)
    rng = np.random.default_rng(seed)

    # 读取所有展览和展品
    exhibitions: list[dict] = []
    with open(exhibitions_path, encoding="utf-8") as f:
        for line in f:
            exhibitions.append(json.loads(line))

    objects: dict[str, dict] = {}
    with open(objects_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            objects[obj["id"]] = obj

    # 所有展品 id 池
    all_obj_ids = list(objects.keys())

    # 过滤展览
    valid = [
        e for e in exhibitions
        if min_gold <= len(e.get("object_ids", [])) <= max_gold
    ]
    log.info(f"TES 样本构建：有效展览 {len(valid)}/{len(exhibitions)}")

    n = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for exh in valid:
            gold_ids = exh["object_ids"]
            gold_set = set(gold_ids)

            # 构建负样本候选
            neg_pool = [i for i in all_obj_ids if i not in gold_set]
            n_neg = max(0, neg_pool_size - len(gold_ids))
            if n_neg > len(neg_pool):
                n_neg = len(neg_pool)
            neg_sample = random.sample(neg_pool, n_neg)

            candidates = gold_ids + neg_sample
            random.shuffle(candidates)

            # 只保留有完整信息的展品
            cand_objs = [
                {
                    "id": oid,
                    "title": objects[oid].get("title", ""),
                    "description": objects[oid].get("description", ""),
                    "culture": objects[oid].get("culture", ""),
                    "medium": objects[oid].get("medium", ""),
                    "date": objects[oid].get("date", ""),
                }
                for oid in candidates
                if oid in objects
            ]

            sample = {
                "id": f"tes_{n:04d}",
                "source_exhibition_id": exh["id"],
                "query": exh.get("theme") or exh.get("title", ""),
                "description": exh.get("description", ""),
                "candidates": cand_objs,
                "gold_ids": gold_ids,
                "k": min(len(gold_ids), 10),
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1

    log.info(f"TES 样本已写入 {output_path}（共 {n} 条）")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 评测入口：比较 gold vs. pred
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tes(
    gold_path: Path,
    pred_path: Path,
    k_list: list[int] = (5, 10),
    embeddings_path: Path | None = None,
) -> dict[str, float]:
    """
    读取 gold（TES 样本 JSONL）和 pred（预测结果 JSONL），计算平均指标。

    pred 格式：每行 {"id": "tes_001", "pred_ids": ["obj3", "obj17", ...]}
    """
    # 加载 gold
    gold_map: dict[str, dict] = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            gold_map[s["id"]] = s

    # 加载 pred
    pred_map: dict[str, list[str]] = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            pred_map[p["id"]] = p["pred_ids"]

    # 加载 embeddings（可选）
    embeddings = None
    if embeddings_path and embeddings_path.exists():
        embeddings = np.load(str(embeddings_path), allow_pickle=True).item()

    # 计算
    all_results: dict[int, list[dict]] = {k: [] for k in k_list}
    matched = 0
    for sid, gold in gold_map.items():
        if sid not in pred_map:
            continue
        matched += 1
        for k in k_list:
            metrics = compute_set_metrics(
                pred_map[sid], gold["gold_ids"], k=k, embeddings=embeddings
            )
            all_results[k].append(metrics)

    log.info(f"评测: {matched}/{len(gold_map)} 样本匹配")

    # 汇总
    aggregated: dict[str, float] = {}
    for k in k_list:
        agg = aggregate_metrics(all_results[k])
        aggregated.update(agg)
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="TES 评测工具")
    subparsers = parser.add_subparsers(dest="cmd")

    # 构建 TES 样本
    build_p = subparsers.add_parser("build", help="从 exhibitions.jsonl 构建 TES 样本")
    build_p.add_argument("--exhibitions", default="data/exhibitions.jsonl")
    build_p.add_argument("--objects", default="data/objects.jsonl")
    build_p.add_argument("--output", default="data/tes_samples.jsonl")
    build_p.add_argument("--neg-pool-size", type=int, default=50)
    build_p.add_argument("--min-gold", type=int, default=5)
    build_p.add_argument("--max-gold", type=int, default=30)

    # 评测
    eval_p = subparsers.add_parser("eval", help="评测 baseline 预测结果")
    eval_p.add_argument("--gold", required=True)
    eval_p.add_argument("--pred", required=True)
    eval_p.add_argument("--k", nargs="+", type=int, default=[5, 10])
    eval_p.add_argument("--embeddings", default=None)

    args = parser.parse_args()
    base = Path(__file__).resolve().parent.parent  # 项目根目录

    if args.cmd == "build":
        build_tes_samples(
            exhibitions_path=base / args.exhibitions,
            objects_path=base / args.objects,
            output_path=base / args.output,
            neg_pool_size=args.neg_pool_size,
            min_gold=args.min_gold,
            max_gold=args.max_gold,
        )
    elif args.cmd == "eval":
        results = evaluate_tes(
            gold_path=base / args.gold,
            pred_path=base / args.pred,
            k_list=args.k,
            embeddings_path=base / args.embeddings if args.embeddings else None,
        )
        print("\n=== TES 评测结果 ===")
        for metric, val in sorted(results.items()):
            print(f"  {metric:20s}: {val:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
