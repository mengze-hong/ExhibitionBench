"""
benchmark/build_samples.py
===========================
从 exhibitions.jsonl + objects.jsonl 构建 MEIP 和 TES 评测样本。

MEIP 样本构建策略:
  - 每个展览至少包含 5 件展品才参与采样
  - 随机选 1 件作为 gold，剩余 n-1 件作为 context
  - 从其他展览中随机采样 9 件展品作为 hard negatives（优先同主题的其他展览）
  - 每个展览最多产生 3 个 MEIP 样本（不同的 gold 选择）

TES 样本构建策略:
  - 每个展览生成 1 个 TES 样本
  - gold: 该展览所有展品（取前 k=10）
  - candidates: gold + 随机 40 件其他展品，共 50 件（随机打乱）

使用方法:
  python benchmark/build_samples.py \\
      --exhibitions data/exhibitions.jsonl \\
      --objects data/objects.jsonl \\
      --meip-output data/meip_samples.jsonl \\
      --tes-output data/tes_samples.jsonl \\
      --max-meip 500 \\
      --max-tes 200 \\
      --seed 42
"""

from __future__ import annotations
import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

log = logging.getLogger(__name__)


def load_objects(objects_path: Path) -> dict[str, dict]:
    """加载 objects.jsonl，返回 id → record 字典。"""
    objects = {}
    with open(objects_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            objects[obj["id"]] = obj
    return objects


def load_exhibitions(exhibitions_path: Path) -> list[dict]:
    """加载 exhibitions.jsonl，返回列表。"""
    exhibitions = []
    with open(exhibitions_path, encoding="utf-8") as f:
        for line in f:
            exhibitions.append(json.loads(line))
    return exhibitions


def build_meip_samples(
    exhibitions: list[dict],
    objects: dict[str, dict],
    max_samples: int = 500,
    min_exh_size: int = 5,
    n_negatives: int = 9,
    max_per_exh: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    构建 MEIP 样本列表。
    每个样本格式:
    {
        "id": "meip_XXXX",
        "exhibition_id": str,
        "context": [obj_record, ...],   # n-1 件展品
        "candidates": [obj_record, ...], # 10 件候选（1 gold + 9 negatives），已打乱
        "gold_id": str,                 # 正确答案的 obj id
    }
    """
    rng = random.Random(seed)

    # 按展览分组所有有元数据的展品
    exh_valid: list[dict] = []
    for exh in exhibitions:
        valid_ids = [oid for oid in exh["object_ids"] if oid in objects]
        if len(valid_ids) >= min_exh_size:
            exh_valid.append({**exh, "valid_ids": valid_ids})

    log.info(f"有效展览数（≥{min_exh_size}件）: {len(exh_valid)}/{len(exhibitions)}")

    # 按主题分组展览，用于 hard negative 采样
    theme_to_exh: dict[str, list[dict]] = defaultdict(list)
    for exh in exh_valid:
        theme_to_exh[exh.get("theme", "unknown")].append(exh)

    all_obj_ids = list(objects.keys())
    samples: list[dict] = []
    sample_id = 0

    for exh in exh_valid:
        if len(samples) >= max_samples:
            break

        valid_ids = exh["valid_ids"]
        # 每个展览最多产生 max_per_exh 个样本
        n_samples_here = min(max_per_exh, len(valid_ids))
        gold_candidates = rng.sample(valid_ids, n_samples_here)

        for gold_id in gold_candidates:
            if len(samples) >= max_samples:
                break

            # context: 除 gold 外的所有展品（最多取 8 件避免 token 过多）
            context_ids = [oid for oid in valid_ids if oid != gold_id]
            if len(context_ids) > 8:
                context_ids = rng.sample(context_ids, 8)

            # hard negatives: 优先从同主题其他展览采样
            theme = exh.get("theme", "unknown")
            neg_pool: list[str] = []

            # 同主题展览的展品
            for other_exh in theme_to_exh.get(theme, []):
                if other_exh["id"] != exh["id"]:
                    neg_pool.extend(other_exh["valid_ids"])

            # 去掉当前展览的展品
            current_exh_ids = set(valid_ids)
            neg_pool = [oid for oid in neg_pool if oid not in current_exh_ids]
            neg_pool = list(set(neg_pool))

            # 若同主题 negatives 不够，从全体补充
            if len(neg_pool) < n_negatives:
                global_pool = [oid for oid in all_obj_ids if oid not in current_exh_ids]
                rng.shuffle(global_pool)
                neg_pool.extend(global_pool)

            # 去重后采样
            neg_pool = list(dict.fromkeys(neg_pool))  # 保序去重
            if len(neg_pool) < n_negatives:
                log.warning(f"展览 {exh['id']} 负样本不足: {len(neg_pool)}")
                continue

            neg_ids = rng.sample(neg_pool, n_negatives)

            # 构建 candidates 列表（gold + negatives，打乱顺序）
            candidate_ids = [gold_id] + neg_ids
            rng.shuffle(candidate_ids)

            sample = {
                "id": f"meip_{sample_id:04d}",
                "exhibition_id": exh["id"],
                "exhibition_theme": exh.get("theme", ""),
                "context": [objects[oid] for oid in context_ids if oid in objects],
                "candidates": [objects[oid] for oid in candidate_ids if oid in objects],
                "gold_id": gold_id,
            }

            # 确保 candidates 中存在 gold
            if gold_id not in [c["id"] for c in sample["candidates"]]:
                log.warning(f"gold {gold_id} 不在 candidates 中，跳过")
                continue

            samples.append(sample)
            sample_id += 1

    log.info(f"MEIP 样本总数: {len(samples)}")
    return samples


def build_tes_samples(
    exhibitions: list[dict],
    objects: dict[str, dict],
    max_samples: int = 200,
    k: int = 10,
    n_candidates: int = 50,
    min_exh_size: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    构建 TES 样本列表。
    每个样本格式:
    {
        "id": "tes_XXXX",
        "exhibition_id": str,
        "query": str,          # 展览主题名
        "description": str,    # 展览描述
        "k": int,              # 需要选几件
        "candidates": [obj_record, ...],  # 50 件候选（含 gold），已打乱
        "gold_ids": [str, ...],           # 正确答案的 obj ids（即展览展品）
    }
    """
    rng = random.Random(seed)
    all_obj_ids = list(objects.keys())

    samples: list[dict] = []
    sample_id = 0

    for exh in exhibitions:
        if len(samples) >= max_samples:
            break

        valid_ids = [oid for oid in exh["object_ids"] if oid in objects]
        if len(valid_ids) < min_exh_size:
            continue

        # gold: 该展览展品（最多取 k 件）
        gold_ids = valid_ids[:k] if len(valid_ids) >= k else valid_ids

        # 从其他展品中填充到 n_candidates
        other_pool = [oid for oid in all_obj_ids if oid not in set(valid_ids)]
        n_extra = n_candidates - len(gold_ids)
        if n_extra > len(other_pool):
            n_extra = len(other_pool)

        extra_ids = rng.sample(other_pool, n_extra)
        candidate_ids = gold_ids + extra_ids
        rng.shuffle(candidate_ids)

        sample = {
            "id": f"tes_{sample_id:04d}",
            "exhibition_id": exh["id"],
            "query": exh.get("theme", exh.get("title", "")),
            "description": exh.get("description", ""),
            "k": len(gold_ids),
            "candidates": [objects[oid] for oid in candidate_ids if oid in objects],
            "gold_ids": gold_ids,
        }
        samples.append(sample)
        sample_id += 1

    log.info(f"TES 样本总数: {len(samples)}")
    return samples


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="构建 MEIP + TES 评测样本")
    parser.add_argument("--exhibitions", default="data/exhibitions.jsonl")
    parser.add_argument("--objects", default="data/objects.jsonl")
    parser.add_argument("--meip-output", default="data/meip_samples.jsonl")
    parser.add_argument("--tes-output", default="data/tes_samples.jsonl")
    parser.add_argument("--max-meip", type=int, default=500)
    parser.add_argument("--max-tes", type=int, default=200)
    parser.add_argument("--min-exh-size", type=int, default=5)
    parser.add_argument("--n-negatives", type=int, default=9,
                        help="MEIP 每个样本的负样本数（candidates = gold + negatives）")
    parser.add_argument("--max-per-exh", type=int, default=3,
                        help="每个展览最多生成多少个 MEIP 样本")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    exh_path = base / args.exhibitions
    obj_path = base / args.objects
    meip_out = base / args.meip_output
    tes_out = base / args.tes_output

    log.info("加载数据...")
    objects = load_objects(obj_path)
    exhibitions = load_exhibitions(exh_path)
    log.info(f"展览: {len(exhibitions)}, 展品: {len(objects)}")

    # 构建 MEIP 样本
    log.info("构建 MEIP 样本...")
    meip_samples = build_meip_samples(
        exhibitions, objects,
        max_samples=args.max_meip,
        min_exh_size=args.min_exh_size,
        n_negatives=args.n_negatives,
        max_per_exh=args.max_per_exh,
        seed=args.seed,
    )
    with open(meip_out, "w", encoding="utf-8") as f:
        for s in meip_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info(f"MEIP → {meip_out} ({len(meip_samples)} 样本)")

    # 构建 TES 样本
    log.info("构建 TES 样本...")
    tes_samples = build_tes_samples(
        exhibitions, objects,
        max_samples=args.max_tes,
        min_exh_size=args.min_exh_size,
        seed=args.seed,
    )
    with open(tes_out, "w", encoding="utf-8") as f:
        for s in tes_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info(f"TES → {tes_out} ({len(tes_samples)} 样本)")

    log.info("=" * 50)
    log.info(f"完成! MEIP: {len(meip_samples)} 样本, TES: {len(tes_samples)} 样本")


if __name__ == "__main__":
    main()
