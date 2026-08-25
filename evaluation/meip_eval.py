"""
evaluation/meip_eval.py
======================
MEIP (Masked Exhibition Item Prediction) 评测 harness。

任务定义：
  给定展览中 n-1 件展品作为上下文，从 10 个候选（1 gold + 9 hard negatives）
  中预测被遮蔽的第 n 件展品。类比完形填空（cloze），测试 LLM 对策展逻辑的理解。

评估指标：
  - Acc@1     模型排第 1 的候选 = gold 的比例
  - MRR       Mean Reciprocal Rank（gold 排名的倒数均值）
  - Hits@3    gold 在前 3 中的比例
  - Hits@5    gold 在前 5 中的比例

数据格式（JSONL，每行一个样本）：
  {
    "id": "meip_001",
    "source_exhibition_id": "europeana_xxx",
    "context": [  // n-1 件已知展品
      {"id": "obj1", "title": "...", "description": "...", ...},
      ...
    ],
    "candidates": [  // 10 个候选（顺序已打乱）
      {"id": "obj_gold", ...},
      {"id": "obj_neg1", ...},
      ...
    ],
    "gold_idx": 3,   // gold 在 candidates 中的下标
    "gold_id": "obj_gold"
  }

pred 格式：每行 {"id": "meip_001", "ranked_ids": ["obj_gold", "obj_neg2", ...]}
  ranked_ids 从最可能到最不可能排列。

使用方法：
  python evaluation/meip_eval.py build --exhibitions data/exhibitions.jsonl ...
  python evaluation/meip_eval.py eval --gold data/meip_samples.jsonl --pred results/pred.jsonl
"""

from __future__ import annotations
import json
import math
import random
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────

def acc_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_id: str) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid == gold_id:
            return 1.0 / i
    return 0.0


def compute_meip_metrics(ranked_ids: list[str], gold_id: str) -> dict[str, float]:
    return {
        "Acc@1": acc_at_k(ranked_ids, gold_id, 1),
        "Hits@3": acc_at_k(ranked_ids, gold_id, 3),
        "Hits@5": acc_at_k(ranked_ids, gold_id, 5),
        "MRR": reciprocal_rank(ranked_ids, gold_id),
    }


def aggregate_metrics(per_sample: list[dict[str, float]]) -> dict[str, float]:
    if not per_sample:
        return {}
    all_keys = set().union(*per_sample)
    return {k: float(np.mean([s.get(k, 0.0) for s in per_sample])) for k in all_keys}


# ─────────────────────────────────────────────────────────────────────────────
# 数据构建：从 exhibitions.jsonl 生成 MEIP 样本
# ─────────────────────────────────────────────────────────────────────────────

def _hard_negatives(
    gold_id: str,
    gold_obj: dict,
    all_objects: dict[str, dict],
    context_ids: list[str],
    n_neg: int = 9,
    strategy: str = "same_culture",
    rng: Any = None,
) -> list[str]:
    """
    为 MEIP 构建 hard negatives。

    策略：
    - same_culture：优先从相同文化来源的展品中采样（但不在上下文中，也不是 gold）
    - random：完全随机（baseline 兜底）

    返回 n_neg 个 id 列表。
    """
    if rng is None:
        rng = random.Random(42)

    exclude = set(context_ids) | {gold_id}
    gold_culture = gold_obj.get("culture", "")

    # 优先同文化
    if strategy == "same_culture" and gold_culture:
        pool_same = [
            oid for oid, obj in all_objects.items()
            if oid not in exclude and obj.get("culture", "") == gold_culture
        ]
        if len(pool_same) >= n_neg:
            return rng.sample(pool_same, n_neg)
        # 不够就混入随机
        pool_rest = [oid for oid in all_objects if oid not in exclude and oid not in pool_same]
        needed = n_neg - len(pool_same)
        if len(pool_rest) >= needed:
            return pool_same + rng.sample(pool_rest, needed)
        else:
            return (pool_same + pool_rest)[:n_neg]

    # fallback random
    pool = [oid for oid in all_objects if oid not in exclude]
    if len(pool) >= n_neg:
        return rng.sample(pool, n_neg)
    return pool[:n_neg]


def build_meip_samples(
    exhibitions_path: Path,
    objects_path: Path,
    output_path: Path,
    n_candidates: int = 10,
    min_context: int = 3,
    max_samples_per_exh: int = 3,
    seed: int = 42,
) -> int:
    """
    从 exhibitions.jsonl 构建 MEIP 样本。

    每个展览最多生成 max_samples_per_exh 个样本（随机遮蔽不同位置）。
    最小上下文长度 min_context，确保任务有足够条件信息。

    返回生成样本数。
    """
    rng = random.Random(seed)

    # 读取数据
    exhibitions: list[dict] = []
    with open(exhibitions_path, encoding="utf-8") as f:
        for line in f:
            exhibitions.append(json.loads(line))

    all_objects: dict[str, dict] = {}
    with open(objects_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            all_objects[obj["id"]] = obj

    def obj_snippet(oid: str) -> dict:
        obj = all_objects.get(oid, {})
        return {
            "id": oid,
            "title": obj.get("title", ""),
            "date": obj.get("date", ""),
            "culture": obj.get("culture", ""),
            "medium": obj.get("medium", ""),
            "description": obj.get("description", ""),
        }

    n = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for exh in exhibitions:
            obj_ids = [oid for oid in exh.get("object_ids", []) if oid in all_objects]
            if len(obj_ids) < min_context + 1:
                continue  # 展览太小

            # 随机选 mask 位置（可多个，最多 max_samples_per_exh）
            mask_positions = list(range(len(obj_ids)))
            rng.shuffle(mask_positions)
            mask_positions = mask_positions[:max_samples_per_exh]

            for mask_pos in mask_positions:
                gold_id = obj_ids[mask_pos]
                context_ids = obj_ids[:mask_pos] + obj_ids[mask_pos + 1:]
                if len(context_ids) < min_context:
                    continue

                # 只保留 min_context 到 min_context+2 件上下文，控制 prompt 长度
                ctx_size = min(len(context_ids), min_context + 2)
                context_ids_trimmed = context_ids[:ctx_size]

                # 构建 hard negatives
                negs = _hard_negatives(
                    gold_id,
                    all_objects.get(gold_id, {}),
                    all_objects,
                    context_ids_trimmed,
                    n_neg=n_candidates - 1,
                    rng=rng,
                )
                if len(negs) < n_candidates - 1:
                    continue  # 候选不够

                candidates = [gold_id] + negs
                rng.shuffle(candidates)
                gold_idx = candidates.index(gold_id)

                sample = {
                    "id": f"meip_{n:05d}",
                    "source_exhibition_id": exh["id"],
                    "exhibition_theme": exh.get("theme") or exh.get("title", ""),
                    "context": [obj_snippet(oid) for oid in context_ids_trimmed],
                    "candidates": [obj_snippet(oid) for oid in candidates],
                    "gold_idx": gold_idx,
                    "gold_id": gold_id,
                    "n_candidates": n_candidates,
                }
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1

    log.info(f"MEIP 样本已写入 {output_path}（共 {n} 条）")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 评测入口
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_meip(gold_path: Path, pred_path: Path) -> dict[str, float]:
    """
    读取 gold（MEIP 样本 JSONL）和 pred，计算平均指标。

    pred 格式：{"id": "meip_00001", "ranked_ids": [...10个id, 从高到低排列...]}
    """
    gold_map: dict[str, dict] = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            gold_map[s["id"]] = s

    pred_map: dict[str, list[str]] = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            pred_map[p["id"]] = p["ranked_ids"]

    per_sample = []
    matched = 0
    for sid, gold in gold_map.items():
        if sid not in pred_map:
            continue
        matched += 1
        metrics = compute_meip_metrics(pred_map[sid], gold["gold_id"])
        per_sample.append(metrics)

    log.info(f"评测: {matched}/{len(gold_map)} 样本匹配")
    agg = aggregate_metrics(per_sample)

    # 随机基线参考
    first = list(gold_map.values())[0] if gold_map else {}
    n_cand = first.get("n_candidates") or len(first.get("candidates", [])) or 10
    log.info(f"随机基线参考: Acc@1 = {1/n_cand:.3f}, MRR = {sum(1/i for i in range(1, n_cand+1))/n_cand:.3f}")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="MEIP 评测工具")
    subparsers = parser.add_subparsers(dest="cmd")

    build_p = subparsers.add_parser("build", help="构建 MEIP 样本")
    build_p.add_argument("--exhibitions", default="data/exhibitions.jsonl")
    build_p.add_argument("--objects", default="data/objects.jsonl")
    build_p.add_argument("--output", default="data/meip_samples.jsonl")
    build_p.add_argument("--n-candidates", type=int, default=10)
    build_p.add_argument("--min-context", type=int, default=3)
    build_p.add_argument("--max-per-exh", type=int, default=3)

    eval_p = subparsers.add_parser("eval", help="评测预测结果")
    eval_p.add_argument("--gold", required=True)
    eval_p.add_argument("--pred", required=True)

    args = parser.parse_args()
    base = Path(__file__).resolve().parent.parent

    if args.cmd == "build":
        build_meip_samples(
            exhibitions_path=base / args.exhibitions,
            objects_path=base / args.objects,
            output_path=base / args.output,
            n_candidates=args.n_candidates,
            min_context=args.min_context,
            max_samples_per_exh=args.max_per_exh,
        )
    elif args.cmd == "eval":
        results = evaluate_meip(base / args.gold, base / args.pred)
        print("\n=== MEIP 评测结果 ===")
        for metric, val in sorted(results.items()):
            print(f"  {metric:10s}: {val:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
