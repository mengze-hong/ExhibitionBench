"""
benchmark/fix_met_meip.py
==========================
修复 Met 展览数据质量问题：
1. 用 department 白名单过滤 8 个 purity < 0.5 的 Met 展览
2. 对过滤后仍有 >= 5 件展品的 6 个展览重新生成 MEIP 样本
3. 将生成的干净样本合并进 meip_samples_v3.jsonl，输出 meip_samples_v3_fixed.jsonl
   - 丢弃 5 个被完全清除的展览（ancient_greece/roman_art/african_art/american_art/oceanic_art）
   - 丢弃 6 个可修复展览的旧噪声样本，替换为新生成的干净样本

用法：
    python benchmark/fix_met_meip.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

THEME_DEPT_RULES = {
    "african_art": ["The Michael C. Rockefeller Wing"],
    "ancient_egypt": ["Egyptian Art"],
    "ancient_greece": ["Greek and Roman Art"],
    "roman_art": ["Greek and Roman Art"],
    "greek_sculpture": ["Greek and Roman Art"],
    "roman_portrait": ["Greek and Roman Art"],
    "islamic_art": ["Islamic Art"],
    "islamic_manuscript": ["Islamic Art"],
    "persian_art": ["Islamic Art"],
    "indian_art": ["Asian Art"],
    "chinese_art": ["Asian Art"],
    "japanese_art": ["Asian Art"],
    "japanese_ceramics": ["Asian Art"],
    "chinese_landscape": ["Asian Art"],
    "korean_art": ["Asian Art"],
    "oceanic_art": ["The Michael C. Rockefeller Wing"],
    "pre-columbian_art": ["The Michael C. Rockefeller Wing"],
    "american_art": ["The American Wing"],
    "american_portrait": ["The American Wing", "Modern and Contemporary Art"],
    "impressionism": ["European Paintings"],
    "baroque_painting": ["European Paintings"],
    "dutch_masters": ["European Paintings"],
    "italian_renaissance": ["European Paintings"],
    "rococo_art": ["European Paintings", "European Sculpture and Decorative Arts"],
    "romanticism": ["European Paintings"],
    "realism_painting": ["European Paintings"],
    "portrait_painting": ["European Paintings"],
    "modern_art": ["Modern and Contemporary Art"],
    "expressionism": ["Modern and Contemporary Art"],
}


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
    log.info(f"Wrote {len(records)} records → {path}")


def build_bm25_index(objects: dict[str, dict]):
    try:
        from rank_bm25 import BM25Okapi
        obj_list = list(objects.values())
        corpus = [
            f"{o.get('title','')} {o.get('culture','')} {o.get('department','')} {o.get('medium','')} {o.get('description','')}".lower().split()
            for o in obj_list
        ]
        return BM25Okapi(corpus), obj_list
    except ImportError:
        log.warning("rank_bm25 not installed; using random negatives")
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
    if bm25 is not None:
        scores = bm25.get_scores(query_text.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        negs = [obj_list[i]["id"] for i in ranked if obj_list[i]["id"] not in exclude_ids][:k * 3]
    else:
        negs = [o["id"] for o in obj_list if o["id"] not in exclude_ids]
        rng.shuffle(negs)
    return negs[:k]


def obj_to_context_entry(obj: dict) -> dict:
    return {
        "id": obj["id"],
        "source": obj.get("source", ""),
        "title": obj.get("title", ""),
        "date": obj.get("date", ""),
        "culture": obj.get("culture", ""),
        "medium": obj.get("medium", ""),
        "description": obj.get("description", ""),
        "image_url": obj.get("image_url", ""),
        "department": obj.get("department", ""),
        "classification": obj.get("classification", ""),
        "period": obj.get("period", ""),
    }


def generate_meip_for_exhibition(
    exh: dict,
    clean_obj_ids: list[str],
    objects: dict[str, dict],
    bm25,
    obj_list: list[dict],
    n_candidates: int = 10,
    n_context: int = 4,
    max_per_exh: int = 15,
    seed: int = 99,
    existing_ids: set = None,
) -> list[dict]:
    rng = random.Random(seed)
    existing_ids = existing_ids or set()
    samples = []

    rng.shuffle(clean_obj_ids)
    for gold_idx in range(n_context, min(len(clean_obj_ids), n_context + max_per_exh)):
        sample_id = f"meip_{exh['id']}_fixed_{gold_idx}"
        if sample_id in existing_ids:
            continue

        gold_id = clean_obj_ids[gold_idx]
        context_ids = clean_obj_ids[:n_context]
        exclude_ids = set(clean_obj_ids)

        context_objs = [objects[oid] for oid in context_ids]
        query_parts = [exh.get("theme", "")]
        for o in context_objs[:2]:
            query_parts.append(f"{o.get('title','')} {o.get('culture','')}")
        query_text = " ".join(p for p in query_parts if p)

        neg_ids = get_hard_negatives(query_text, gold_id, exclude_ids, bm25, obj_list, k=n_candidates - 1, rng=rng)
        if len(neg_ids) < n_candidates - 1:
            continue

        candidate_ids = neg_ids + [gold_id]
        rng.shuffle(candidate_ids)

        samples.append({
            "id": sample_id,
            "exhibition_id": exh["id"],
            "exhibition_theme": exh.get("theme", ""),
            "context": [obj_to_context_entry(objects[oid]) for oid in context_ids],
            "candidates": [obj_to_context_entry(objects[oid]) for oid in candidate_ids],
            "gold_id": gold_id,
        })

        if len(samples) >= max_per_exh:
            break

    return samples


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    objects = load_objects(DATA / "objects_v3.jsonl")
    exhibitions_raw = load_jsonl(DATA / "exhibitions_v3.jsonl")
    meip_v3 = load_jsonl(DATA / "meip_samples_v3.jsonl")

    # 计算每个 Met 展览 purity，分类
    def get_purity_and_clean_ids(eid, exh):
        theme_key = eid.replace("met_ext_", "").replace("met_", "")
        valid_depts = THEME_DEPT_RULES.get(theme_key)
        if valid_depts is None:
            return 1.0, exh.get("object_ids", [])
        obj_ids = exh.get("object_ids", [])
        clean = [oid for oid in obj_ids if oid in objects and objects[oid].get("department", "") in valid_depts]
        purity = len(clean) / len(obj_ids) if obj_ids else 0.0
        return purity, clean

    DROP_EXH_IDS = set()
    FIX_EXH = {}  # eid -> clean_obj_ids

    for exh in exhibitions_raw:
        eid = exh["id"]
        if not eid.startswith("met"):
            continue
        purity, clean_ids = get_purity_and_clean_ids(eid, exh)
        if purity >= 0.95:
            continue  # already clean
        if len(clean_ids) >= 5:
            FIX_EXH[eid] = clean_ids
            log.info(f"FIXABLE: {eid} purity={purity:.0%} clean={len(clean_ids)}/{len(exh.get('object_ids',[]))}")
        else:
            DROP_EXH_IDS.add(eid)
            log.info(f"DROP:    {eid} purity={purity:.0%} only {len(clean_ids)} clean objects")

    log.info(f"Exhibitions to drop: {len(DROP_EXH_IDS)} → {DROP_EXH_IDS}")
    log.info(f"Exhibitions to fix:  {len(FIX_EXH)} → {set(FIX_EXH)}")

    # Build BM25
    bm25, obj_list = build_bm25_index(objects)

    # Generate replacement MEIP samples for fixable exhibitions
    exh_by_id = {e["id"]: e for e in exhibitions_raw}
    existing_ids = {s["id"] for s in meip_v3}
    new_samples: list[dict] = []

    for eid, clean_ids in FIX_EXH.items():
        exh = exh_by_id[eid]
        samps = generate_meip_for_exhibition(
            exh=exh,
            clean_obj_ids=clean_ids,
            objects=objects,
            bm25=bm25,
            obj_list=obj_list,
            max_per_exh=15,
            seed=42,
            existing_ids=existing_ids,
        )
        log.info(f"  {eid}: generated {len(samps)} new samples")
        new_samples.extend(samps)

    # Merge: discard old samples from DROP or FIX exhibitions, add new clean ones
    AFFECTED = DROP_EXH_IDS | set(FIX_EXH.keys())
    clean_old = [s for s in meip_v3 if s["exhibition_id"] not in AFFECTED]
    final = clean_old + new_samples

    log.info(f"v3 total: {len(meip_v3)}")
    log.info(f"  Removed (noisy): {len(meip_v3) - len(clean_old)}")
    log.info(f"  Added (clean):   {len(new_samples)}")
    log.info(f"Final (fixed):     {len(final)}")

    if not args.dry_run:
        out = DATA / "meip_samples_v3_fixed.jsonl"
        write_jsonl(out, final)
        print(f"\n[OK] Written: {out} ({len(final)} samples)")
    else:
        print(f"\n[dry-run] Would write {len(final)} samples to meip_samples_v3_fixed.jsonl")
        print(f"  Dropped from noisy exhibitions: {sum(1 for s in meip_v3 if s['exhibition_id'] in DROP_EXH_IDS)}")
        print(f"  Replaced from fixable exhibitions: {sum(1 for s in meip_v3 if s['exhibition_id'] in set(FIX_EXH))}")
        print(f"  New clean samples generated: {len(new_samples)}")


if __name__ == "__main__":
    main()
