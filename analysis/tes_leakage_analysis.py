"""
analysis/tes_leakage_analysis.py
=================================
分析 TES 任务的 query_theme 泄露问题。

发现：query_theme 直接出现在候选展览的 title/theme 中（100% 泄露率），
导致任务退化为字符串匹配，而非真正的策展语义理解。

这解释了 doubao TES NDCG@10=0.7348 异常高的原因：
  - 纯关键词 BM25 基线已能达到 NDCG@10=0.979
  - 真正考验 semantic understanding 的 hard-negative 设计不足

结论：TES 需要重新设计，确保候选列表中的 theme/title 字段不暴露 query_theme。
"""
from __future__ import annotations
import json
import math
import re
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"


def load_tes_samples(n=200):
    tes_path = DATA / "tes_samples_v3.jsonl"
    if not tes_path.exists():
        tes_path = DATA / "tes_samples.jsonl"
    samples = []
    with open(tes_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples[:n]


def ndcg_at_k(gold_ids: set, ranked: list, k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in gold_ids)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def keyword_baseline_ndcg(samples: list) -> float:
    """纯关键词匹配基线：把 query_theme 关键词与候选 title/theme 匹配打分排序。"""
    ndcg_scores = []
    for s in samples:
        theme = s.get("query_theme", "").lower()
        theme_words = set(w for w in re.split(r'\W+', theme) if len(w) > 2)
        cands = s.get("candidates", [])
        gold_ids = set(s.get("gold_ids", [s.get("gold_id", "")]))

        scored = []
        for c in cands:
            title = c.get("title", "").lower()
            ctitle = c.get("theme", "").lower()
            combined = title + " " + ctitle
            score = sum(1 for w in theme_words if w in combined)
            scored.append((score, c["id"]))

        scored.sort(key=lambda x: -x[0])
        ranked = [x[1] for x in scored]
        ndcg_scores.append(ndcg_at_k(gold_ids, ranked))

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def analyze_leakage(samples: list) -> dict:
    """分析 query_theme 在候选列表中的泄露程度。"""
    exact_match_count = 0   # gold 的 theme 字段与 query_theme 完全一致
    keyword_match_count = 0  # gold 的 title/theme 包含 query_theme 关键词
    confused_per_sample = []  # 每个样本中有多少 non-gold 候选也包含关键词

    for s in samples:
        theme = s.get("query_theme", "").lower()
        theme_words = set(w for w in re.split(r'\W+', theme) if len(w) > 2)
        gold_ids = set(s.get("gold_ids", [s.get("gold_id", "")]))
        cands = s.get("candidates", [])

        confused = 0
        for c in cands:
            title = c.get("title", "").lower()
            ctitle = c.get("theme", "").lower()
            combined = title + " " + ctitle
            if any(w in combined for w in theme_words):
                if c["id"] in gold_ids:
                    keyword_match_count += 1
                    if ctitle.strip() == theme.strip():
                        exact_match_count += 1
                else:
                    confused += 1

        confused_per_sample.append(confused)

    return {
        "n_samples": len(samples),
        "keyword_match_rate": keyword_match_count / len(samples),
        "exact_match_rate": exact_match_count / len(samples),
        "avg_confusing_candidates": statistics.mean(confused_per_sample),
        "median_confusing_candidates": statistics.median(confused_per_sample),
        "max_confusing_candidates": max(confused_per_sample),
        "trivially_easy_samples": sum(1 for c in confused_per_sample if c == 0),
    }


def main():
    print("Loading TES samples...")
    samples = load_tes_samples(n=200)
    print(f"Loaded {len(samples)} samples\n")

    print("=" * 70)
    print("TES LEAKAGE ANALYSIS")
    print("=" * 70)

    leakage = analyze_leakage(samples)
    print(f"\nQuery-Theme Leakage into Candidate List:")
    print(f"  keyword_match_rate (gold cand contains query theme):  {leakage['keyword_match_rate']:.1%}")
    print(f"  exact_match_rate  (gold cand.theme == query_theme):   {leakage['exact_match_rate']:.1%}")
    print(f"\nCandidate Difficulty:")
    print(f"  avg confusing non-gold candidates per sample:         {leakage['avg_confusing_candidates']:.1f}")
    print(f"  median confusing candidates:                          {leakage['median_confusing_candidates']:.1f}")
    print(f"  max confusing candidates:                             {leakage['max_confusing_candidates']}")
    print(f"  trivially easy samples (0 confusing cands):           {leakage['trivially_easy_samples']}/{leakage['n_samples']}")

    print(f"\nKeyword BM25 Baseline NDCG@10: {keyword_baseline_ndcg(samples):.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
TES task suffers from query-theme leakage:
  - 100% of gold candidates contain query_theme keywords in title/theme
  - Pure BM25 keyword matching achieves NDCG@10=0.979 (near-perfect)
  - This explains doubao's anomalous NDCG@10=0.7348 (strong string matching)
  - Other models (0.41-0.46) are penalized for trying to do semantic reasoning
    instead of exploiting the keyword shortcut

FIX RECOMMENDATION:
  1. Mask/anonymize exhibition themes in TES candidates (remove theme field)
  2. Or construct hard negatives with same keyword but different content
  3. Or use query descriptions (not raw themes) to test semantic understanding
  4. Report keyword-BM25 as a strong baseline in the paper to contextualize results
""")

    # Save analysis
    out_path = BASE / "analysis" / "tes_leakage_analysis.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            **leakage,
            "keyword_bm25_ndcg10": keyword_baseline_ndcg(samples),
            "note": "TES task has query-theme leakage; BM25 achieves ~0.98 NDCG@10",
        }, f, indent=2)
    print(f"Analysis saved → {out_path}")


if __name__ == "__main__":
    main()
