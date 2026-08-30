"""
analysis/cultural_bias.py
==========================
文化偏差分析：比较不同文化来源展品在 MEIP 和 TES 任务上的性能差异。

分析维度:
1. 展览来源（西方 vs 亚洲 vs 非洲 vs 中东 vs 其他）
2. 展品文化标签（来自 objects.jsonl 的 culture 字段）
3. 各 baseline 在不同文化分组上的 Acc@1 (MEIP) 和 NDCG@10 (TES)

使用方法:
  python analysis/cultural_bias.py --gold-meip data/meip_samples.jsonl
"""

from __future__ import annotations
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# 文化分组映射
# ─────────────────────────────────────────────────────────────────────────────

WESTERN_KEYWORDS = [
    "french", "dutch", "german", "italian", "spanish", "english", "british",
    "american", "greek", "roman", "flemish", "portuguese", "swiss", "austrian",
    "scandinavian", "nordic", "belgian", "netherlandish", "europe", "western",
    "byzantine", "celtic", "etruscan"
]
ASIAN_KEYWORDS = [
    "chinese", "japanese", "korean", "indian", "thai", "cambodian", "tibetan",
    "persian", "iranian", "mughal", "southeast asian", "central asian", "asian",
    "ottoman", "turkish", "chinese"
]
AFRICAN_KEYWORDS = [
    "african", "egyptian", "mali", "yoruba", "akan", "kongo", "benin",
    "west african", "east african", "sub-saharan"
]
MIDDLE_EAST_KEYWORDS = [
    "islamic", "arab", "mesopotamian", "babylonian", "assyrian", "sumerian",
    "middle eastern"
]
PRE_COLUMBIAN_KEYWORDS = [
    "maya", "aztec", "inca", "pre-columbian", "mesoamerica", "andean",
    "indigenous american", "native american", "oceanic"
]


def classify_culture(culture_str: str) -> str:
    """将 culture 字段分类为文化组。"""
    if not culture_str:
        return "Unknown"
    c = culture_str.lower()
    for kw in WESTERN_KEYWORDS:
        if kw in c:
            return "Western"
    for kw in ASIAN_KEYWORDS:
        if kw in c:
            return "Asian"
    for kw in AFRICAN_KEYWORDS:
        if kw in c:
            return "African"
    for kw in MIDDLE_EAST_KEYWORDS:
        if kw in c:
            return "Middle Eastern"
    for kw in PRE_COLUMBIAN_KEYWORDS:
        if kw in c:
            return "Pre-Columbian/Oceanic"
    return "Other"


def load_objects(objects_path: Path) -> dict[str, dict]:
    objs = {}
    with open(objects_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            objs[r["id"]] = r
    return objs


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


# ─────────────────────────────────────────────────────────────────────────────
# MEIP 分析
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_RESULTS = BASE / "results" / "baselines_pred"
MEIP_BASELINES = {
    "BM25":            BASELINE_RESULTS / "bm25_meip_pred.jsonl",
    "SBERT":           BASELINE_RESULTS / "sbert_meip_pred.jsonl",
    "GPT-5.2 0-shot":  BASELINE_RESULTS / "zeroshot_meip_pred.jsonl",
    "GPT-5.2 few-shot":BASELINE_RESULTS / "gpt5_fewshot_meip_pred.jsonl",
    "RAG+KG":          BASELINE_RESULTS / "rag_kg_meip_pred.jsonl",
}


def analyze_meip_by_culture(gold_samples: list[dict], objects: dict) -> None:
    """按 gold 展品的文化分组统计 MEIP Acc@1。"""
    print("\n" + "=" * 70)
    print("MEIP 文化偏差分析（Acc@1）")
    print("=" * 70)

    # 为每个样本确定文化分组（基于 gold 展品的 culture 字段）
    sample_culture = {}
    for s in gold_samples:
        gold_id = s["gold_id"]
        obj = objects.get(gold_id, {})
        culture = obj.get("culture", "")
        sample_culture[s["id"]] = classify_culture(culture)

    culture_counts = defaultdict(int)
    for c in sample_culture.values():
        culture_counts[c] += 1
    print("\n各文化分组样本数:")
    for c, n in sorted(culture_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:25s}: {n} 样本")

    # 各 baseline 的分组 Acc@1
    results = {}
    for bl_name, pred_path in MEIP_BASELINES.items():
        if not pred_path.exists():
            continue
        preds = {r["id"]: r["ranked_ids"] for r in load_jsonl(pred_path)}
        gold_map = {s["id"]: s["gold_id"] for s in gold_samples}

        group_correct = defaultdict(list)
        for sid, gold_id in gold_map.items():
            if sid not in preds or sid not in sample_culture:
                continue
            ranked = preds[sid]
            hit = 1 if ranked and ranked[0] == gold_id else 0
            group_correct[sample_culture[sid]].append(hit)

        results[bl_name] = {g: np.mean(v) for g, v in group_correct.items()}

    # 打印表格
    all_groups = sorted(culture_counts.keys(), key=lambda x: -culture_counts[x])
    header = f"{'Baseline':20s}" + "".join(f"{g:22s}" for g in all_groups)
    print("\n" + header)
    print("-" * len(header))
    for bl_name, group_scores in results.items():
        row = f"{bl_name:20s}"
        for g in all_groups:
            score = group_scores.get(g, float("nan"))
            row += f"{score:>10.4f}            " if not np.isnan(score) else f"{'N/A':>10s}            "
        print(row)

    # 偏差分析：最大分组差距
    print("\n各 Baseline 文化偏差（max - min Acc@1）:")
    for bl_name, group_scores in results.items():
        valid = [v for v in group_scores.values() if not np.isnan(v)]
        if len(valid) >= 2:
            gap = max(valid) - min(valid)
            best = max(group_scores, key=lambda g: group_scores.get(g, -1))
            worst = min(group_scores, key=lambda g: group_scores.get(g, 999))
            print(f"  {bl_name:20s}: 偏差={gap:.4f}  最佳={best}({group_scores[best]:.4f})  最差={worst}({group_scores[worst]:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# TES 分析
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="文化偏差分析")
    parser.add_argument("--gold-meip", default="data/meip_samples.jsonl")
    parser.add_argument("--objects", default="data/objects.jsonl")
    args = parser.parse_args()

    objects = load_objects(BASE / args.objects)
    log.info(f"加载展品: {len(objects)}")

    gold_samples = load_jsonl(BASE / args.gold_meip)
    log.info(f"MEIP 样本: {len(gold_samples)}")
    analyze_meip_by_culture(gold_samples, objects)


if __name__ == "__main__":
    main()
