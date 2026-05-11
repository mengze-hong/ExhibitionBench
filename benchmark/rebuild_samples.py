"""
benchmark/rebuild_samples.py
==============================
基于扩充后的数据集（v2）重建 TES / MEIP 样本。

TES  目标: ≥ 300 个查询（每个 TES-eligible 展览 1 个查询）
MEIP 目标: ≥ 800 个样本（保留原有 500，新增 300+）

用法:
  python benchmark/rebuild_samples.py
  python benchmark/rebuild_samples.py --tes-only
  python benchmark/rebuild_samples.py --meip-only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_objects(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in load_jsonl(path)}


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records)} records -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# BM25 hard negatives（用于 MEIP）
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25_index(objects: dict[str, dict]):
    try:
        from rank_bm25 import BM25Okapi
        obj_list = list(objects.values())
        corpus = []
        for o in obj_list:
            text = f"{o.get('title','')} {o.get('culture','')} {o.get('department','')} {o.get('medium','')} {o.get('description','')}"
            corpus.append(text.lower().split())
        bm25 = BM25Okapi(corpus)
        return bm25, obj_list
    except ImportError:
        log.warning("rank_bm25 not installed; using random negatives for MEIP")
        return None, list(objects.values())


def get_hard_negatives(
    query_text: str,
    gold_id: str,
    exclude_ids: set,
    bm25,
    obj_list: list[dict],
    k: int = 9,
    rng: random.Random = random,
) -> list[str]:
    """获取 k 个 BM25 hard negative，不含 gold 和 exclude。"""
    if bm25 is None:
        # 随机负例回退
        pool = [o["id"] for o in obj_list if o["id"] not in exclude_ids and o["id"] != gold_id]
        return rng.sample(pool, min(k, len(pool)))

    scores = bm25.get_scores(query_text.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    negatives = []
    for idx in ranked:
        oid = obj_list[idx]["id"]
        if oid != gold_id and oid not in exclude_ids:
            negatives.append(oid)
            if len(negatives) >= k:
                break
    # 补足
    if len(negatives) < k:
        pool = [o["id"] for o in obj_list if o["id"] not in exclude_ids
                and o["id"] != gold_id and o["id"] not in negatives]
        negatives.extend(rng.sample(pool, min(k - len(negatives), len(pool))))
    return negatives


# ─────────────────────────────────────────────────────────────────────────────
# TES 样本生成
# ─────────────────────────────────────────────────────────────────────────────

def build_tes_samples(
    exhibitions: list[dict],
    objects: dict[str, dict],
    n_negative_exhibitions: int = 49,
    min_objects: int = 5,
    seed: int = 42,
    max_samples: int = 400,
) -> list[dict]:
    """
    TES: 给定主题查询，从候选展览列表中找到最相关的展览。
    每个展览生成 1 个 TES 样本，50-way ranking（1 gold + 49 negatives）。
    """
    rng = random.Random(seed)

    # 过滤有效展览
    valid_exhs = [
        exh for exh in exhibitions
        if len([oid for oid in exh.get("object_ids", []) if oid in objects]) >= min_objects
    ]
    log.info(f"TES-eligible exhibitions: {len(valid_exhs)}")

    if len(valid_exhs) < n_negative_exhibitions + 1:
        log.warning(f"Not enough exhibitions for {n_negative_exhibitions} negatives")

    samples = []
    for exh in valid_exhs[:max_samples]:
        # Gold: this exhibition
        gold_id = exh["id"]

        # Negatives: random other exhibitions
        other_exhs = [e for e in valid_exhs if e["id"] != gold_id]
        neg_exhs = rng.sample(other_exhs, min(n_negative_exhibitions, len(other_exhs)))

        # Build candidate list (gold + negatives, shuffled)
        candidates = [exh] + neg_exhs
        rng.shuffle(candidates)

        # Candidate summaries for prompt
        candidate_summaries = []
        for c in candidates:
            obj_ids = [oid for oid in c.get("object_ids", []) if oid in objects][:5]
            sample_objs = [objects[oid] for oid in obj_ids]
            candidate_summaries.append({
                "id": c["id"],
                "theme": c.get("theme", ""),
                "title": c.get("title", ""),
                "description": c.get("description", ""),
                "sample_objects": [
                    {"title": o.get("title", ""), "culture": o.get("culture", ""), "date": o.get("date", "")}
                    for o in sample_objs
                ],
            })

        samples.append({
            "id": f"tes_{exh['id']}",
            "query_theme": exh.get("theme", ""),
            "query_description": exh.get("description", ""),
            "gold_id": gold_id,
            "gold_ids": [gold_id],
            "candidates": candidate_summaries,
            "source": exh.get("source", ""),
        })

    log.info(f"Built {len(samples)} TES samples")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MEIP 样本生成（增量，避免重复）
# ─────────────────────────────────────────────────────────────────────────────

def build_meip_samples(
    exhibitions: list[dict],
    objects: dict[str, dict],
    existing_sample_ids: set,
    n_candidates: int = 10,
    n_context: int = 4,
    min_objects: int = 5,
    seed: int = 42,
    max_new_samples: int = 500,
) -> list[dict]:
    """
    MEIP: 给定展览主题和部分展品（context），从候选列表中预测下一件展品。
    10-way ranking: 1 gold + 9 BM25 hard negatives。
    """
    rng = random.Random(seed)

    log.info("Building BM25 index for MEIP hard negatives...")
    bm25, obj_list = build_bm25_index(objects)

    valid_exhs = [
        exh for exh in exhibitions
        if len([oid for oid in exh.get("object_ids", []) if oid in objects]) >= min_objects + n_context
    ]
    log.info(f"MEIP-eligible exhibitions: {len(valid_exhs)}")

    samples = []
    for exh in valid_exhs:
        if len(samples) >= max_new_samples:
            break

        obj_ids = [oid for oid in exh.get("object_ids", []) if oid in objects]
        rng.shuffle(obj_ids)

        # 每个展览生成多个样本（按不同的 gold 切分）
        for gold_idx in range(n_context, min(len(obj_ids), n_context + 5)):
            sample_id = f"meip_{exh['id']}_{gold_idx}"
            if sample_id in existing_sample_ids:
                continue

            gold_id = obj_ids[gold_idx]
            context_ids = obj_ids[:n_context]
            exclude_ids = set(obj_ids)  # 不从当前展览取负例

            # 构建查询文本
            context_objs = [objects[oid] for oid in context_ids]
            query_parts = [exh.get("theme", "")]
            for o in context_objs[:2]:
                query_parts.append(f"{o.get('title','')} {o.get('culture','')}")
            query_text = " ".join(p for p in query_parts if p)

            # BM25 hard negatives
            neg_ids = get_hard_negatives(
                query_text, gold_id, exclude_ids, bm25, obj_list, k=n_candidates - 1, rng=rng
            )
            if len(neg_ids) < n_candidates - 1:
                continue

            candidate_ids = neg_ids + [gold_id]
            rng.shuffle(candidate_ids)

            samples.append({
                "id": sample_id,
                "exhibition_id": exh["id"],
                "exhibition_theme": exh.get("theme", ""),
                "source": exh.get("source", ""),
                "context": [{"id": oid} for oid in context_ids],
                "candidate_ids": candidate_ids,
                "gold_id": gold_id,
            })

            if len(samples) >= max_new_samples:
                break

    log.info(f"Built {len(samples)} new MEIP samples")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Rebuild TES/MEIP samples from v2 dataset")
    parser.add_argument("--exhibitions", default="data/exhibitions_v2.jsonl")
    parser.add_argument("--objects", default="data/objects_v2.jsonl")
    parser.add_argument("--tes-out", default="data/tes_samples_v2.jsonl")
    parser.add_argument("--meip-out", default="data/meip_samples_v2.jsonl")
    parser.add_argument("--existing-meip", default="data/meip_samples.jsonl")
    parser.add_argument("--tes-only", action="store_true")
    parser.add_argument("--meip-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Fallback paths
    exh_path = BASE / args.exhibitions
    obj_path = BASE / args.objects
    if not exh_path.exists():
        exh_path = DATA / "exhibitions.jsonl"
    if not obj_path.exists():
        obj_path = DATA / "objects.jsonl"

    exhibitions = load_jsonl(exh_path)
    objects = load_objects(obj_path)
    log.info(f"Loaded {len(exhibitions)} exhibitions, {len(objects)} objects")

    if not args.meip_only:
        tes_samples = build_tes_samples(exhibitions, objects, seed=args.seed)
        write_jsonl(BASE / args.tes_out, tes_samples)
        print(f"\nTES v2: {len(tes_samples)} samples (was 54, now {len(tes_samples)})")

    if not args.tes_only:
        # 加载现有 MEIP 样本（避免重复）
        existing_meip_path = BASE / args.existing_meip
        existing_meip = load_jsonl(existing_meip_path) if existing_meip_path.exists() else []
        existing_ids = {s["id"] for s in existing_meip}
        log.info(f"Existing MEIP samples: {len(existing_meip)}")

        new_meip = build_meip_samples(
            exhibitions, objects,
            existing_sample_ids=existing_ids,
            seed=args.seed,
            max_new_samples=500,
        )
        all_meip = existing_meip + new_meip
        write_jsonl(BASE / args.meip_out, all_meip)
        print(f"\nMEIP v2: {len(all_meip)} samples ({len(existing_meip)} existing + {len(new_meip)} new)")

    print("\nDone! Rebuild complete.")
    print(f"  TES: {BASE / args.tes_out}")
    print(f"  MEIP: {BASE / args.meip_out}")


if __name__ == "__main__":
    main()
