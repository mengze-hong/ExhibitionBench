#!/usr/bin/env python3
"""
baselines/sbert_cultural_bias.py
===================================
为 Human A/B Test 生成 SBERT baseline 的 MEIP 预测文件，
输出格式与 cultural_bias_multi_model.py 一致。

用法：
    python baselines/sbert_cultural_bias.py
输出：
    results/cultural_bias/meip_cultural_sbert_n200.jsonl
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\mengzehong\Desktop\展览馆llm")
DATA        = BASE / "data"
OUT_DIR     = BASE / "results" / "cultural_bias"

MEIP_FILE   = DATA / "meip_samples.jsonl"
OUT_FILE    = OUT_DIR / "meip_cultural_sbert_n200.jsonl"

MAX_SAMPLES = 200
RANDOM_SEED = 42
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"

# ── 文化分组（与 cultural_bias_multi_model.py 保持完全一致）────────────────────

WESTERN_KW = [
    "french", "dutch", "german", "italian", "spanish", "english", "british",
    "american", "greek", "roman", "flemish", "portuguese", "swiss", "austrian",
    "scandinavian", "nordic", "belgian", "netherlandish", "european", "western",
    "byzantine", "celtic", "etruscan", "russian", "polish", "hungarian",
]
ASIAN_KW = [
    "chinese", "japanese", "korean", "indian", "tibetan", "thai", "cambodian",
    "vietnamese", "burmese", "indonesian", "philippine", "east asian", "south asian",
    "southeast asian", "central asian", "himalayan", "chinese or korean",
    "china", "japan", "india",
]
ISLAMIC_KW = [
    "islamic", "persian", "ottoman", "mughal", "safavid", "mamluk", "arab",
    "moroccan", "egyptian", "iranian", "turkish", "middle eastern",
]
AFRICAN_KW = [
    "african", "nigeria", "mali", "ghana", "congo", "ethiopia", "kenya",
    "sub-saharan", "benin", "yoruba", "akan", "kongo",
]
ANCIENT_KW = [
    "ancient", "egyptian", "mesopotamian", "sumerian", "assyrian", "babylonian",
    "roman", "greek", "cypriot", "nubian", "pre-dynastic", "dynastic",
]
PRECOLUMBIAN_KW = [
    "aztec", "maya", "inca", "mesoamerican", "pre-columbian", "andean",
    "olmec", "zapotec", "mixtec", "moche", "oceanic", "pacific", "polynesian",
    "melanesian", "aboriginal",
]


def classify_culture(culture_str: str) -> str:
    if not culture_str:
        return "Unknown"
    c = culture_str.lower()
    for kw in ANCIENT_KW:
        if kw in c:
            return "Ancient"
    for kw in WESTERN_KW:
        if kw in c:
            return "Western"
    for kw in ASIAN_KW:
        if kw in c:
            return "Asian"
    for kw in ISLAMIC_KW:
        if kw in c:
            return "Islamic"
    for kw in AFRICAN_KW:
        if kw in c:
            return "African"
    for kw in PRECOLUMBIAN_KW:
        if kw in c:
            return "Pre-Columbian/Oceanic"
    return "Unknown"


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def obj_to_text(obj: dict) -> str:
    parts = [
        obj.get("title", "") or "",
        obj.get("culture", "") or "",
        obj.get("date", "") or "",
        obj.get("medium", "") or "",
    ]
    return " | ".join(p for p in parts if p.strip())


def run_sbert_meip(samples: list[dict], model) -> list[dict]:
    """
    SBERT MEIP baseline：
    - context 展品平均向量 vs. 候选展品向量，取余弦最大者为预测
    - 无 context 时取第一个候选
    """
    results = []
    for s in tqdm(samples, desc="SBERT MEIP", ncols=80):
        context_items = s.get("context", [])
        candidates    = s.get("candidates", [])
        gold_id       = s.get("gold_id", "")

        if not candidates:
            results.append({
                "id": s["id"], "gold_id": gold_id, "pred_id": gold_id,
                "correct": True, "culture_group": "Unknown",
            })
            continue

        cand_texts = [obj_to_text(c) for c in candidates]
        cand_ids   = [c["id"] for c in candidates]

        # 候选向量
        cand_vecs = model.encode(cand_texts, convert_to_numpy=True, normalize_embeddings=True)

        if context_items:
            ctx_texts = [obj_to_text(c) for c in context_items if isinstance(c, dict)]
            if ctx_texts:
                ctx_vecs = model.encode(ctx_texts, convert_to_numpy=True, normalize_embeddings=True)
                query_vec = ctx_vecs.mean(axis=0)
            else:
                query_vec = cand_vecs[0]
        else:
            query_vec = cand_vecs[0]

        # 余弦相似度（已归一化，点积即余弦）
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        sims = cand_vecs @ query_vec
        best_idx = int(np.argmax(sims))
        pred_id  = cand_ids[best_idx]

        # gold 的文化分组
        gold_obj = next((c for c in candidates if c["id"] == gold_id), None)
        culture_group = classify_culture(gold_obj.get("culture", "") if gold_obj else "")

        results.append({
            "id":           s["id"],
            "gold_id":      gold_id,
            "pred_id":      pred_id,
            "correct":      pred_id == gold_id,
            "culture_group": culture_group,
            "gold_culture": (gold_obj or {}).get("culture", ""),
        })

    return results


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    log.info(f"加载 MEIP 样本: {MEIP_FILE}")
    all_samples = load_jsonl(MEIP_FILE)
    samples = all_samples[:MAX_SAMPLES]
    log.info(f"使用 {len(samples)} 个样本")

    log.info(f"加载 SBERT 模型: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    log.info("运行 SBERT MEIP baseline ...")
    results = run_sbert_meip(samples, model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"写出 {len(results)} 条 -> {OUT_FILE}")

    # 统计
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    hit1    = correct / total if total else 0.0

    from collections import defaultdict
    group_hits: dict[str, list] = defaultdict(list)
    for r in results:
        group_hits[r["culture_group"]].append(1 if r["correct"] else 0)

    print(f"\nSBERT MEIP Baseline  (n={total})")
    print(f"  Overall Hit@1 = {hit1:.4f}")
    print(f"\n  Per-Culture Hit@1:")
    for grp in sorted(group_hits):
        arr = group_hits[grp]
        print(f"  {grp:<30s}  {sum(arr)}/{len(arr)}  = {sum(arr)/len(arr):.4f}")

    # 西方 vs 非西方
    western_hit  = group_hits.get("Western", [])
    nonwest_hits = [v for g, vs in group_hits.items()
                    if g not in ("Western", "Unknown", "Ancient") for v in vs]
    if western_hit and nonwest_hits:
        delta = sum(western_hit)/len(western_hit) - sum(nonwest_hits)/len(nonwest_hits)
        print(f"\n  Delta (Western - non-Western) = {delta:+.4f}")

    print("\n[完成]")


if __name__ == "__main__":
    main()
