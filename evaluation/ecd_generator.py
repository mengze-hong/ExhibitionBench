"""
benchmark/ecd_generator.py
===========================
ECD (Exhibition Coherence Discrimination) 任务构造器

任务定义:
  给定两个展览序列（e+真实 vs e-被扰动），判断哪个更连贯。
  f(e+, e-) ∈ {0, 1}，其中 e- 为 4 种扰动之一。

4 级扰动设计（难度递增）:
  L1 - Temporal Anachronism : 插入时代错配展品（时间差 > 500 年）
  L2 - Cultural Drift       : 插入文化圈不符展品
  L3 - Thematic Deviation   : 插入 BM25 相关但主题不符展品（hard negative）
  L4 - Subtle Incoherence   : 插入 SBERT 语义相似但风格细节矛盾的展品

评估指标:
  PairAcc_l = mean( 1[y_hat = y] )  for l in {L1,L2,L3,L4}
  最终报告 macro-average + per-level breakdown

输出:
  data/ecd_samples.jsonl
  每条记录格式:
    {
      "id": "ecd_<uuid>",
      "level": 1,                     # 1-4
      "perturbation_type": "temporal_anachronism",
      "positive": {                   # 真实展览
          "exhibition_id": "...",
          "theme": "...",
          "items": [{"id":..., "title":..., ...}, ...]
      },
      "negative": {                   # 扰动展览
          "exhibition_id": "...",
          "theme": "...",
          "items": [...],
          "perturbed_index": 2,       # 被替换的位置
          "intruder_id": "..."        # 入侵展品 id
      },
      "label": 0                      # 0=positive is correct answer
    }

用法:
  python benchmark/ecd_generator.py --objects data/objects_v2.jsonl
      --exhibitions data/exhibitions_v2.jsonl --out data/ecd_samples.jsonl
  python benchmark/ecd_generator.py  # 使用默认路径
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

# ─────────────────────────────────────────────────────────────────────────────
# 年份解析 + 文化分类（与 error_analysis.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────

ERA_PATTERNS = [
    (r"\b(\d{1,4})\s*BCE?\b",     lambda m: -int(m.group(1))),
    (r"\b(\d{4})\s*CE?\b",        lambda m: int(m.group(1))),
    (r"\b(\d{4})[-–](\d{4})\b",   lambda m: (int(m.group(1)) + int(m.group(2))) // 2),
    (r"\b(\d{4})\b",              lambda m: int(m.group(1))),
    (r"(\d+)th\s+century",        lambda m: int(m.group(1)) * 100 - 50),
    (r"(\d+)st\s+century",        lambda m: int(m.group(1)) * 100 - 50),
    (r"(\d+)nd\s+century",        lambda m: int(m.group(1)) * 100 - 50),
    (r"(\d+)rd\s+century",        lambda m: int(m.group(1)) * 100 - 50),
]


def parse_year(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    for pattern, extractor in ERA_PATTERNS:
        m = re.search(pattern, str(date_str), re.IGNORECASE)
        if m:
            try:
                return extractor(m)
            except Exception:
                continue
    return None


WESTERN_KW = ["french", "dutch", "german", "italian", "spanish", "english", "british",
              "american", "greek", "roman", "flemish", "portuguese", "swiss", "austrian",
              "scandinavian", "nordic", "belgian", "netherlandish", "europe", "western",
              "byzantine", "celtic", "etruscan"]
ASIAN_KW   = ["chinese", "japanese", "korean", "indian", "thai", "cambodian", "tibetan",
              "persian", "iranian", "mughal", "southeast asian", "central asian", "asian",
              "ottoman", "turkish", "east asian", "south asian", "japan", "china", "korea"]
AFRICAN_KW = ["african", "mali", "yoruba", "akan", "kongo", "benin",
              "west african", "east african", "sub-saharan"]
MIDEAST_KW = ["islamic", "arab", "mesopotamian", "babylonian", "assyrian", "sumerian",
              "middle eastern", "egypt", "egyptian"]


def classify_culture(culture_str: str) -> str:
    if not culture_str:
        return "Unknown"
    c = culture_str.lower()
    for kw in WESTERN_KW:
        if kw in c:
            return "Western"
    for kw in ASIAN_KW:
        if kw in c:
            return "Asian"
    for kw in AFRICAN_KW:
        if kw in c:
            return "African"
    for kw in MIDEAST_KW:
        if kw in c:
            return "Middle Eastern"
    return "Other"


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
    objs = {}
    for r in load_jsonl(path):
        objs[r["id"]] = r
    return objs


# ─────────────────────────────────────────────────────────────────────────────
# BM25 hard negative（用于 L3 扰动）
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25_index(objects: dict[str, dict]):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.warning("rank_bm25 not installed; L3 will use random negatives")
        return None, []

    obj_list = list(objects.values())
    corpus = []
    for o in obj_list:
        text = f"{o.get('title','')} {o.get('culture','')} {o.get('department','')} {o.get('description','')}"
        corpus.append(text.lower().split())

    bm25 = BM25Okapi(corpus)
    return bm25, obj_list


def bm25_topk(bm25, obj_list: list[dict], query: str, k: int = 50,
              exclude_ids: set | None = None) -> list[dict]:
    if bm25 is None:
        return []
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    results = []
    for idx in ranked:
        if len(results) >= k:
            break
        obj = obj_list[idx]
        if exclude_ids and obj["id"] in exclude_ids:
            continue
        results.append(obj)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SBERT cosine（用于 L4 扰动）
# ─────────────────────────────────────────────────────────────────────────────

def build_sbert_index(objects: dict[str, dict], model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer(model_name)
        obj_list = list(objects.values())
        texts = [
            f"{o.get('title','')} {o.get('medium','')} {o.get('department','')} {o.get('description','')[:200]}"
            for o in obj_list
        ]
        log.info(f"Encoding {len(texts)} objects with SBERT...")
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                                  convert_to_numpy=True, normalize_embeddings=True)
        return obj_list, embeddings
    except ImportError:
        log.warning("sentence-transformers not installed; L4 will use BM25 fallback")
        return [], None


def sbert_topk(obj_list: list[dict], embeddings, query_emb, k: int = 20,
               exclude_ids: set | None = None) -> list[dict]:
    import numpy as np
    if embeddings is None or len(obj_list) == 0:
        return []
    scores = embeddings @ query_emb
    ranked = scores.argsort()[::-1]
    results = []
    for idx in ranked:
        if len(results) >= k:
            break
        obj = obj_list[idx]
        if exclude_ids and obj["id"] in exclude_ids:
            continue
        results.append(obj)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 展览上下文摘要（用于 BM25/SBERT 查询）
# ─────────────────────────────────────────────────────────────────────────────

def exhibition_query_text(exh: dict, item_objects: list[dict]) -> str:
    parts = [exh.get("theme", ""), exh.get("description", "")]
    for o in item_objects[:3]:
        parts.append(f"{o.get('title','')} {o.get('culture','')} {o.get('department','')}")
    return " ".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────────────────
# 扰动生成函数
# ─────────────────────────────────────────────────────────────────────────────

def make_intruder_l1_temporal(
    context_objects: list[dict],
    all_objects: dict[str, dict],
    exclude_ids: set,
    threshold_years: int = 500,
    rng: random.Random = random,
) -> Optional[dict]:
    """L1: 时代错配 — 找一个年代偏差 > threshold 的展品。"""
    ctx_years = [parse_year(o.get("date", "")) for o in context_objects]
    ctx_years = [y for y in ctx_years if y is not None]
    if not ctx_years:
        return None
    ctx_mean = sum(ctx_years) / len(ctx_years)

    candidates = [
        o for oid, o in all_objects.items()
        if oid not in exclude_ids
        and parse_year(o.get("date", "")) is not None
        and abs(parse_year(o["date"]) - ctx_mean) > threshold_years
    ]
    return rng.choice(candidates) if candidates else None


def make_intruder_l2_culture(
    context_objects: list[dict],
    all_objects: dict[str, dict],
    exclude_ids: set,
    rng: random.Random = random,
) -> Optional[dict]:
    """L2: 文化漂移 — 找一个文化圈与上下文主流文化不同的展品。"""
    from collections import Counter
    ctx_cultures = [classify_culture(o.get("culture", "")) for o in context_objects]
    dominant = Counter(c for c in ctx_cultures if c != "Unknown")
    if not dominant:
        return None
    dom_culture = dominant.most_common(1)[0][0]

    candidates = [
        o for oid, o in all_objects.items()
        if oid not in exclude_ids
        and classify_culture(o.get("culture", "")) not in (dom_culture, "Unknown", "Other")
    ]
    return rng.choice(candidates) if candidates else None


def make_intruder_l3_thematic(
    exh: dict,
    context_objects: list[dict],
    all_objects: dict[str, dict],
    exclude_ids: set,
    bm25=None,
    obj_list: list[dict] = (),
    rng: random.Random = random,
) -> Optional[dict]:
    """L3: BM25 相关但主题不符。"""
    query = exhibition_query_text(exh, context_objects)
    candidates = bm25_topk(bm25, list(obj_list), query, k=50, exclude_ids=exclude_ids)
    if not candidates:
        # 降级：随机选
        pool = [o for oid, o in all_objects.items() if oid not in exclude_ids]
        return rng.choice(pool) if pool else None

    # 从 top-50 BM25 中挑主题偏离的（部门/分类与展览主题关键词不重叠）
    theme_words = set(exh.get("theme", "").lower().split())
    filtered = []
    for cand in candidates:
        dept_words = set(((cand.get("department") or "") + " " + (cand.get("classification") or "")).lower().split())
        if not (theme_words & dept_words):  # 无主题词重叠
            filtered.append(cand)

    if filtered:
        return rng.choice(filtered[:20])
    return rng.choice(candidates[:10]) if candidates else None


def make_intruder_l4_subtle(
    context_objects: list[dict],
    all_objects: dict[str, dict],
    exclude_ids: set,
    sbert_obj_list: list[dict],
    sbert_embeddings,
    sbert_model=None,
    rng: random.Random = random,
) -> Optional[dict]:
    """L4: SBERT 语义相似但风格细节矛盾。"""
    if sbert_embeddings is None or not sbert_obj_list:
        pool = [o for oid, o in all_objects.items() if oid not in exclude_ids]
        return rng.choice(pool) if pool else None

    import numpy as np
    # 用上下文均值作为查询
    ctx_texts = [
        f"{o.get('title','')} {o.get('medium','')} {o.get('description','')[:100]}"
        for o in context_objects
    ]
    # 找对应的 embedding
    sbert_id_to_idx = {o["id"]: i for i, o in enumerate(sbert_obj_list)}
    ctx_embs = []
    for o in context_objects:
        if o["id"] in sbert_id_to_idx:
            ctx_embs.append(sbert_embeddings[sbert_id_to_idx[o["id"]]])
    if not ctx_embs:
        pool = [o for oid, o in all_objects.items() if oid not in exclude_ids]
        return rng.choice(pool) if pool else None

    query_emb = np.mean(ctx_embs, axis=0)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

    candidates = sbert_topk(sbert_obj_list, sbert_embeddings, query_emb,
                            k=30, exclude_ids=exclude_ids)
    if not candidates:
        return None

    # 在 top-30 中找与上下文文化/时代矛盾的
    from collections import Counter
    ctx_cultures = [classify_culture(o.get("culture", "")) for o in context_objects]
    dom_culture = Counter(c for c in ctx_cultures if c not in ("Unknown", "Other")).most_common(1)
    dom_culture = dom_culture[0][0] if dom_culture else None

    ctx_years = [parse_year(o.get("date", "")) for o in context_objects]
    ctx_years = [y for y in ctx_years if y is not None]
    ctx_mean_yr = sum(ctx_years) / len(ctx_years) if ctx_years else None

    subtle_candidates = []
    for cand in candidates:
        cand_culture = classify_culture(cand.get("culture", ""))
        cand_year = parse_year(cand.get("date", ""))
        # 语义相似（已是 top-30）但细节矛盾：文化圈不同 或 年代差距 100-500 年（不太离谱，L4 难检测）
        culture_mismatch = dom_culture and cand_culture not in (dom_culture, "Unknown", "Other")
        temporal_subtle = (ctx_mean_yr and cand_year and
                           100 < abs(cand_year - ctx_mean_yr) < 500)
        if culture_mismatch or temporal_subtle:
            subtle_candidates.append(cand)

    if subtle_candidates:
        return rng.choice(subtle_candidates)
    return rng.choice(candidates[:5])


# ─────────────────────────────────────────────────────────────────────────────
# 主生成函数
# ─────────────────────────────────────────────────────────────────────────────

def generate_ecd_samples(
    exhibitions: list[dict],
    objects: dict[str, dict],
    samples_per_level: int = 200,
    seed: int = 42,
    min_items: int = 5,
    use_sbert: bool = True,
) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    counts = {1: 0, 2: 0, 3: 0, 4: 0}

    # 过滤：至少有 min_items 件展品且展品在 objects 中
    valid_exhs = []
    for exh in exhibitions:
        obj_ids = [oid for oid in exh.get("object_ids", []) if oid in objects]
        if len(obj_ids) >= min_items:
            valid_exhs.append({**exh, "object_ids": obj_ids})

    log.info(f"Valid exhibitions for ECD: {len(valid_exhs)}")

    # 构建索引
    log.info("Building BM25 index...")
    bm25, bm25_obj_list = build_bm25_index(objects)

    sbert_obj_list, sbert_embeddings = [], None
    if use_sbert:
        log.info("Building SBERT index...")
        sbert_obj_list, sbert_embeddings = build_sbert_index(objects)

    all_object_ids = set(objects.keys())
    rng.shuffle(valid_exhs)

    # 每个展览每级别尝试生成 1 个样本
    max_iterations = len(valid_exhs) * 10  # 防止死循环
    iteration = 0

    while min(counts.values()) < samples_per_level and iteration < max_iterations:
        iteration += 1
        exh = rng.choice(valid_exhs)
        obj_ids = exh["object_ids"]
        rng.shuffle(obj_ids)

        # 随机取 5-8 件作为正例展览
        n_items = rng.randint(min_items, min(8, len(obj_ids)))
        positive_ids = obj_ids[:n_items]
        positive_objects = [objects[oid] for oid in positive_ids]
        exclude_ids = set(positive_ids) | all_object_ids - set(objects.keys())

        # 确定这次生成哪个 level（优先不够的 level）
        level = min(counts, key=lambda l: counts[l])
        if counts[level] >= samples_per_level:
            continue

        # 生成 intruder
        intruder: Optional[dict] = None
        perturbation_type = ""

        if level == 1:
            intruder = make_intruder_l1_temporal(positive_objects, objects, set(positive_ids), rng=rng)
            perturbation_type = "temporal_anachronism"
        elif level == 2:
            intruder = make_intruder_l2_culture(positive_objects, objects, set(positive_ids), rng=rng)
            perturbation_type = "cultural_drift"
        elif level == 3:
            intruder = make_intruder_l3_thematic(exh, positive_objects, objects, set(positive_ids),
                                                  bm25=bm25, obj_list=bm25_obj_list, rng=rng)
            perturbation_type = "thematic_deviation"
        elif level == 4:
            intruder = make_intruder_l4_subtle(positive_objects, objects, set(positive_ids),
                                               sbert_obj_list, sbert_embeddings, rng=rng)
            perturbation_type = "subtle_incoherence"

        if intruder is None:
            continue

        # 构造负例（替换随机位置）
        perturb_idx = rng.randint(0, n_items - 1)
        negative_ids = positive_ids[:]
        negative_ids[perturb_idx] = intruder["id"]
        negative_objects = [
            intruder if i == perturb_idx else objects[oid]
            for i, oid in enumerate(negative_ids)
        ]

        def obj_summary(o: dict) -> dict:
            return {
                "id": o["id"],
                "title": o.get("title", ""),
                "date": o.get("date", ""),
                "culture": o.get("culture", ""),
                "medium": o.get("medium", ""),
                "department": o.get("department", ""),
            }

        sample = {
            "id": f"ecd_{uuid.uuid4().hex[:12]}",
            "level": level,
            "perturbation_type": perturbation_type,
            "positive": {
                "exhibition_id": exh["id"],
                "theme": exh.get("theme", ""),
                "items": [obj_summary(o) for o in positive_objects],
            },
            "negative": {
                "exhibition_id": exh["id"],
                "theme": exh.get("theme", ""),
                "items": [obj_summary(o) for o in negative_objects],
                "perturbed_index": perturb_idx,
                "intruder_id": intruder["id"],
            },
            "label": 0,  # 0 = positive is always the correct sequence
        }
        samples.append(sample)
        counts[level] += 1

        if sum(counts.values()) % 50 == 0:
            log.info(f"ECD progress: {counts}")

    log.info(f"Generated ECD samples: {counts} = {sum(counts.values())} total")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 统计输出
# ─────────────────────────────────────────────────────────────────────────────

def print_ecd_statistics(samples: list[dict]) -> None:
    from collections import Counter
    level_counts = Counter(s["level"] for s in samples)
    ptype_counts = Counter(s["perturbation_type"] for s in samples)

    print("\n" + "=" * 60)
    print(f"ECD Dataset Statistics")
    print("=" * 60)
    print(f"Total samples    : {len(samples)}")
    print("\nPer level:")
    for lvl in [1, 2, 3, 4]:
        ptype = {1: "Temporal Anachronism", 2: "Cultural Drift",
                 3: "Thematic Deviation", 4: "Subtle Incoherence"}[lvl]
        print(f"  L{lvl} {ptype:25s}: {level_counts.get(lvl, 0)}")
    print(f"\nMacro-average target: {len(samples) // 4} per level")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="ECD task generator for ExhibitionBench")
    parser.add_argument("--exhibitions", default="data/exhibitions_v2.jsonl",
                        help="Exhibition JSONL file")
    parser.add_argument("--objects", default="data/objects_v2.jsonl",
                        help="Objects JSONL file")
    parser.add_argument("--out", default="data/ecd_samples.jsonl",
                        help="Output ECD samples file")
    parser.add_argument("--samples-per-level", type=int, default=200,
                        help="Target samples per perturbation level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sbert", action="store_true",
                        help="Skip SBERT (faster, worse L4 quality)")
    args = parser.parse_args()

    exh_path = BASE / args.exhibitions
    obj_path = BASE / args.objects
    out_path = BASE / args.out

    # Fallback to v1 if v2 not ready
    if not exh_path.exists():
        exh_path = DATA / "exhibitions.jsonl"
        log.warning(f"exhibitions_v2.jsonl not found, using {exh_path}")
    if not obj_path.exists():
        obj_path = DATA / "objects.jsonl"
        log.warning(f"objects_v2.jsonl not found, using {obj_path}")

    exhibitions = load_jsonl(exh_path)
    objects = load_objects(obj_path)
    log.info(f"Loaded {len(exhibitions)} exhibitions, {len(objects)} objects")

    samples = generate_ecd_samples(
        exhibitions, objects,
        samples_per_level=args.samples_per_level,
        seed=args.seed,
        use_sbert=not args.no_sbert,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(samples)} ECD samples -> {out_path}")

    print_ecd_statistics(samples)


if __name__ == "__main__":
    main()
