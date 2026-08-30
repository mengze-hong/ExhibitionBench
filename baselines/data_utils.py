"""Schema-compatible data helpers shared by ExhibitionBench baselines."""

from __future__ import annotations

import json
from pathlib import Path


def load_objects(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        return {item["id"]: item for line in handle if (item := json.loads(line))}


def resolve_items(raw_items: list, objects: dict[str, dict]) -> list[dict]:
    """Resolve either embedded objects or object IDs without changing the data."""
    resolved = []
    for item in raw_items:
        if isinstance(item, dict):
            resolved.append(item)
        elif item in objects:
            resolved.append(objects[item])
        else:
            resolved.append({"id": item})
    return resolved


def meip_context(sample: dict, objects: dict[str, dict]) -> list[dict]:
    return resolve_items(sample.get("context", []), objects)


def meip_candidates(sample: dict, objects: dict[str, dict]) -> list[dict]:
    raw = sample.get("candidates", sample.get("candidate_ids", []))
    return resolve_items(raw, objects)


def tes_query(sample: dict) -> str:
    theme = sample.get("query_theme", sample.get("query", ""))
    description = sample.get("query_description", sample.get("description", ""))
    return f"{theme} {description}".strip()


def exhibition_to_text(exhibition: dict) -> str:
    """Represent a TES candidate only through sampled artworks (leak-free)."""
    parts = []
    for item in exhibition.get("sample_objects", []):
        parts.extend(
            [item.get("title", ""), item.get("culture", ""), item.get("date", "")]
        )
    return " ".join(str(part) for part in parts if part)
