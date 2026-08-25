#!/usr/bin/env python3
"""Validate released ExhibitionBench JSONL files without modifying them."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def candidate_ids(sample: dict) -> list[str]:
    raw = sample.get("candidates", sample.get("candidate_ids", []))
    return [item.get("id", "") if isinstance(item, dict) else item for item in raw]


def validate(data_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    objects = load_jsonl(data_dir / "objects.jsonl")
    exhibitions = load_jsonl(data_dir / "exhibitions.jsonl")
    meip = load_jsonl(data_dir / "meip_samples.jsonl")
    tes = load_jsonl(data_dir / "tes_samples.jsonl")
    ecd = load_jsonl(data_dir / "ecd_samples.jsonl")

    object_ids = [item.get("id") for item in objects]
    exhibition_ids = [item.get("id") for item in exhibitions]
    for name, ids in (("objects", object_ids), ("exhibitions", exhibition_ids)):
        duplicates = [key for key, count in Counter(ids).items() if key and count > 1]
        if duplicates:
            errors.append(f"{name}: {len(duplicates)} duplicate IDs")
        if any(not key for key in ids):
            errors.append(f"{name}: missing ID")

    known_objects = set(object_ids)
    for sample in meip:
        sid = sample.get("id", "<unknown>")
        ids = candidate_ids(sample)
        if len(ids) != 10 or len(set(ids)) != len(ids):
            errors.append(f"{sid}: MEIP requires 10 unique candidates")
        if sample.get("gold_id") not in ids:
            errors.append(f"{sid}: MEIP gold is absent from candidates")
        referenced = ids + [x.get("id") if isinstance(x, dict) else x for x in sample.get("context", [])]
        missing = [oid for oid in referenced if oid and oid not in known_objects]
        if missing:
            warnings.append(f"{sid}: {len(set(missing))} object IDs lack global metadata")

    for sample in tes:
        sid = sample.get("id", "<unknown>")
        ids = candidate_ids(sample)
        gold_ids = sample.get("gold_ids") or [sample.get("gold_id")]
        if len(ids) != 50 or len(set(ids)) != len(ids):
            errors.append(f"{sid}: TES requires 50 unique candidates")
        if any(gold not in ids for gold in gold_ids if gold):
            errors.append(f"{sid}: TES gold is absent from candidates")

    levels = Counter(sample.get("level") for sample in ecd)
    for sample in ecd:
        sid = sample.get("id", "<unknown>")
        positive = sample.get("positive", {}).get("items", [])
        negative = sample.get("negative", {}).get("items", [])
        if not positive or len(positive) != len(negative):
            errors.append(f"{sid}: ECD sequences must be non-empty and equally sized")
        if sample.get("label") not in (0, 1):
            errors.append(f"{sid}: ECD label must be 0 or 1")
    if set(levels) != {1, 2, 3, 4}:
        errors.append(f"ECD levels are incomplete: {dict(levels)}")

    print(f"objects={len(objects)} exhibitions={len(exhibitions)}")
    print(f"meip={len(meip)} tes={len(tes)} ecd={len(ecd)} levels={dict(levels)}")
    print(f"errors={len(errors)} warnings={len(warnings)}")
    for message in errors:
        print(f"ERROR: {message}")
    for message in warnings[:20]:
        print(f"WARNING: {message}")
    if len(warnings) > 20:
        print(f"WARNING: {len(warnings) - 20} additional warnings omitted")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    errors, _ = validate(args.data_dir)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
