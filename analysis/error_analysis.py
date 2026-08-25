"""
analysis/error_analysis.py
===========================
错误分析：识别 LLM baseline 在 MEIP 任务上的典型失败模式。

分析的失败模式:
1. 时代混搭 (Temporal Mismatch): 推荐的展品与展览时代严重不符
2. 风格矛盾 (Style Contradiction): 推荐的展品风格与展览主题矛盾（如将现代艺术放入文艺复兴展）
3. 文化漂移 (Cultural Drift): 推荐的展品来自完全不同的文化圈

输出:
  - 每种失败模式的典型案例（case study）
  - 各 baseline 的失败模式分布统计

使用方法:
  python analysis/error_analysis.py --baseline gpt5_fewshot
  python analysis/error_analysis.py --all
"""

from __future__ import annotations
import json
import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict, Counter
import re

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
BASELINE_RESULTS = BASE / "results" / "baselines_pred"

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

BASELINES_MEIP = {
    "BM25":             BASELINE_RESULTS / "bm25_meip_pred.jsonl",
    "SBERT":            BASELINE_RESULTS / "sbert_meip_pred.jsonl",
    "GPT-5.2 0-shot":   BASELINE_RESULTS / "zeroshot_meip_pred.jsonl",
    "GPT-5.2 few-shot": BASELINE_RESULTS / "gpt5_fewshot_meip_pred.jsonl",
    "RAG+KG":           BASELINE_RESULTS / "rag_kg_meip_pred.jsonl",
}

# 粗略的时代映射（世纪）
ERA_PATTERNS = [
    (r"\b(\d{4})\s*BCE?\b",     lambda m: -int(m.group(1))),   # "210 BCE" → -210
    (r"\b(\d{4})\s*CE?\b",      lambda m: int(m.group(1))),    # "1890 CE" → 1890
    (r"\b(\d{4})[-–](\d{4})\b", lambda m: (int(m.group(1)) + int(m.group(2))) // 2),
    (r"\b(\d{4})\b",            lambda m: int(m.group(1))),
    (r"(\d+)th century",        lambda m: int(m.group(1)) * 100 - 50),
    (r"(\d+)st century",        lambda m: int(m.group(1)) * 100 - 50),
    (r"(\d+)nd century",        lambda m: int(m.group(1)) * 100 - 50),
    (r"(\d+)rd century",        lambda m: int(m.group(1)) * 100 - 50),
]


def parse_year(date_str: str) -> int | None:
    """从日期字符串中解析年份（正数=CE，负数=BCE）。"""
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


def temporal_gap(year1: int | None, year2: int | None) -> int | None:
    """计算两个年份的差距（绝对值）。"""
    if year1 is None or year2 is None:
        return None
    return abs(year1 - year2)


# ─────────────────────────────────────────────────────────────────────────────
# 文化分组（同 cultural_bias.py）
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
    "ottoman", "turkish"
]
AFRICAN_KEYWORDS = [
    "african", "egyptian", "mali", "yoruba", "akan", "kongo", "benin",
    "west african", "east african", "sub-saharan"
]
MIDDLE_EAST_KEYWORDS = [
    "islamic", "arab", "mesopotamian", "babylonian", "assyrian", "sumerian",
    "middle eastern"
]


def classify_culture(culture_str: str) -> str:
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
    return "Other"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def load_objects(objects_path: Path) -> dict[str, dict]:
    objs = {}
    with open(objects_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            objs[r["id"]] = r
    return objs


# ─────────────────────────────────────────────────────────────────────────────
# 错误模式分类
# ─────────────────────────────────────────────────────────────────────────────

def analyze_errors(
    gold_samples: list[dict],
    pred_path: Path,
    objects: dict,
    baseline_name: str,
    n_cases: int = 3,
    temporal_threshold: int = 200,
) -> dict:
    """
    分析 MEIP 错误样本，分类三种典型失败模式。
    返回统计信息和 case study。
    """
    if not pred_path.exists():
        return {}

    preds = {r["id"]: r.get("ranked_ids") or r.get("pred_ids") or [] for r in load_jsonl(pred_path)}
    gold_map = {s["id"]: s for s in gold_samples}

    # 各类失败计数
    error_types = Counter()
    case_studies = defaultdict(list)

    for sid, sample in gold_map.items():
        if sid not in preds:
            continue
        ranked = preds[sid]
        if not ranked:
            continue
        top1 = ranked[0]
        gold_id = sample["gold_id"]
        if top1 == gold_id:
            continue  # 正确预测，跳过

        # ---- 错误样本分析 ----
        gold_obj = objects.get(gold_id, {})
        pred_obj = objects.get(top1, {})
        context_objs = [objects.get(oid, {}) for oid in (c["id"] for c in sample.get("context", []))]

        # 1. 时代混搭分析
        gold_year = parse_year(gold_obj.get("date", ""))
        pred_year = parse_year(pred_obj.get("date", ""))
        ctx_years = [parse_year(o.get("date", "")) for o in context_objs]
        ctx_years = [y for y in ctx_years if y is not None]
        if ctx_years:
            ctx_mean = sum(ctx_years) / len(ctx_years)
            gold_gap = temporal_gap(gold_year, int(ctx_mean))
            pred_gap = temporal_gap(pred_year, int(ctx_mean))
        else:
            gold_gap = pred_gap = None

        temporal_error = (
            pred_gap is not None and gold_gap is not None
            and pred_gap > temporal_threshold
            and pred_gap > gold_gap + 100  # 预测展品比 gold 偏离时代更多
        )

        # 2. 文化漂移分析
        gold_culture = classify_culture(gold_obj.get("culture", ""))
        pred_culture = classify_culture(pred_obj.get("culture", ""))
        ctx_cultures = [classify_culture(o.get("culture", "")) for o in context_objs]
        ctx_most_common = Counter(ctx_cultures).most_common(1)[0][0] if ctx_cultures else "Unknown"

        culture_drift = (
            pred_culture != "Unknown"
            and ctx_most_common != "Unknown"
            and pred_culture != ctx_most_common
            and gold_culture == ctx_most_common  # gold 文化与展览一致，但预测漂移
        )

        # 3. 风格矛盾分析（基于主题关键词 vs 展品分类）
        theme = sample.get("exhibition_theme", "").lower()
        pred_medium = pred_obj.get("medium", "").lower()
        pred_title = pred_obj.get("title", "").lower()
        pred_dept = pred_obj.get("department", "").lower()

        STYLE_CONFLICTS = [
            ({"modern", "contemporary", "abstract", "expressionism"}, {"ancient", "classical", "roman", "greek", "egypt", "medieval", "renaissance"}),
            ({"ancient", "egypt", "roman", "greek", "classical"}, {"modern", "contemporary", "abstract", "photograph", "20th century"}),
            ({"medieval", "gothic", "tapestry"}, {"modern", "impressionism", "photograph", "digital"}),
        ]
        style_conflict = False
        for modern_kw, ancient_kw in STYLE_CONFLICTS:
            if any(kw in theme for kw in modern_kw) and any(kw in pred_dept or kw in pred_title or kw in pred_medium for kw in ancient_kw):
                style_conflict = True
                break
            if any(kw in theme for kw in ancient_kw) and any(kw in pred_dept or kw in pred_title or kw in pred_medium for kw in modern_kw):
                style_conflict = True
                break

        # 记录错误类型
        if temporal_error:
            error_types["temporal_mismatch"] += 1
            if len(case_studies["temporal_mismatch"]) < n_cases:
                case_studies["temporal_mismatch"].append({
                    "sample_id": sid,
                    "theme": sample.get("exhibition_theme", ""),
                    "context_summary": [f'{o.get("title","")} ({o.get("date","")}, {o.get("culture","")})' for o in context_objs[:3]],
                    "gold": f'{gold_obj.get("title","")} (year≈{gold_year}, gap≈{gold_gap}yr)',
                    "predicted": f'{pred_obj.get("title","")} (year≈{pred_year}, gap≈{pred_gap}yr)',
                    "error_type": "Temporal Mismatch",
                    "explanation": f"展览上下文年代约{int(ctx_mean) if ctx_years else '?'}，预测展品偏离{pred_gap}年，gold偏离{gold_gap}年",
                })
        elif culture_drift:
            error_types["cultural_drift"] += 1
            if len(case_studies["cultural_drift"]) < n_cases:
                case_studies["cultural_drift"].append({
                    "sample_id": sid,
                    "theme": sample.get("exhibition_theme", ""),
                    "context_culture": ctx_most_common,
                    "gold": f'{gold_obj.get("title","")} (culture={gold_obj.get("culture","")})',
                    "predicted": f'{pred_obj.get("title","")} (culture={pred_obj.get("culture","")})',
                    "error_type": "Cultural Drift",
                    "explanation": f"展览文化圈={ctx_most_common}，预测展品文化={pred_culture}（应为{gold_culture}）",
                })
        elif style_conflict:
            error_types["style_conflict"] += 1
            if len(case_studies["style_conflict"]) < n_cases:
                case_studies["style_conflict"].append({
                    "sample_id": sid,
                    "theme": sample.get("exhibition_theme", ""),
                    "gold": f'{gold_obj.get("title","")} (dept={gold_obj.get("department","")})',
                    "predicted": f'{pred_obj.get("title","")} (dept={pred_obj.get("department","")}, medium={pred_obj.get("medium","")})',
                    "error_type": "Style Conflict",
                    "explanation": f"主题 '{sample.get('exhibition_theme','')}' 与预测展品风格矛盾",
                })
        else:
            error_types["other"] += 1

    total_errors = sum(error_types.values())
    return {
        "baseline": baseline_name,
        "total_errors": total_errors,
        "error_distribution": dict(error_types),
        "case_studies": dict(case_studies),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def print_analysis(result: dict) -> None:
    """格式化打印错误分析结果。"""
    if not result:
        return

    bl = result["baseline"]
    total = result["total_errors"]
    dist = result["error_distribution"]

    print(f"\n{'='*70}")
    print(f"错误分析: {bl}  (错误样本总数: {total})")
    print(f"{'='*70}")
    for etype, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        label_map = {
            "temporal_mismatch": "时代混搭 (Temporal Mismatch)",
            "cultural_drift": "文化漂移 (Cultural Drift)",
            "style_conflict": "风格矛盾 (Style Conflict)",
            "other": "其他错误 (Other)",
        }
        print(f"  {label_map.get(etype, etype):40s}: {count:4d} ({pct:.1f}%)")

    cases = result["case_studies"]
    for etype, case_list in cases.items():
        label_map = {
            "temporal_mismatch": "时代混搭",
            "cultural_drift": "文化漂移",
            "style_conflict": "风格矛盾",
        }
        if not case_list:
            continue
        print(f"\n--- {label_map.get(etype, etype)} Case Studies ---")
        for i, case in enumerate(case_list, 1):
            def s(x): return str(x).replace('\xa0', ' ').encode('ascii', 'replace').decode('ascii') if x else ''
            print(f"\n  Case {i} [{case['sample_id']}] 主题: {case['theme']}")
            if "context_summary" in case:
                print(f"    展览上下文 (前3): {'; '.join(str(c).replace(chr(160),' ') for c in case['context_summary'])}")
            print(f"    Gold展品:   {str(case['gold']).replace(chr(160),' ')}")
            print(f"    模型预测:   {str(case['predicted']).replace(chr(160),' ')}")
            print(f"    分析:       {str(case['explanation']).replace(chr(160),' ')}")


def main():
    import io
    # Fix Windows GBK encoding issue
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="MEIP 错误分析")
    parser.add_argument("--baseline", default=None, help="指定 baseline 名称（默认所有）")
    parser.add_argument("--all", action="store_true", help="分析所有 baseline")
    parser.add_argument("--gold-meip", default="data/meip_samples.jsonl")
    parser.add_argument("--objects", default="data/objects.jsonl")
    parser.add_argument("--n-cases", type=int, default=3, help="每种错误模式输出的 case 数量")
    args = parser.parse_args()

    gold_samples = load_jsonl(BASE / args.gold_meip)
    objects = load_objects(BASE / args.objects)
    log.info(f"加载 {len(gold_samples)} 个 MEIP 样本, {len(objects)} 件展品")

    to_analyze = {}
    if args.all or args.baseline is None:
        to_analyze = BASELINES_MEIP
    else:
        if args.baseline in BASELINES_MEIP:
            to_analyze = {args.baseline: BASELINES_MEIP[args.baseline]}
        else:
            print(f"未知 baseline: {args.baseline}. 可用: {list(BASELINES_MEIP.keys())}")
            return

    print("\n" + "=" * 70)
    print("MEIP 错误模式分析")
    print("=" * 70)
    print("\n三种主要失败模式:")
    print("  1. 时代混搭 (Temporal Mismatch): 预测展品与展览时代严重偏离")
    print("  2. 文化漂移 (Cultural Drift):    预测展品来自错误的文化圈")
    print("  3. 风格矛盾 (Style Conflict):    预测展品风格与展览主题矛盾")
    print("  4. 其他错误 (Other):             语义相似但展览逻辑不连贯等")

    all_results = []
    for bl_name, pred_path in to_analyze.items():
        result = analyze_errors(
            gold_samples, pred_path, objects, bl_name, n_cases=args.n_cases
        )
        if result:
            all_results.append(result)
            print_analysis(result)

    # 汇总对比表
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("各 Baseline 错误模式分布对比")
        print(f"{'='*70}")
        labels = ["temporal_mismatch", "cultural_drift", "style_conflict", "other"]
        header = f"{'Baseline':20s}" + "".join(f"{'时代混搭':>12s}{'文化漂移':>12s}{'风格矛盾':>12s}{'其他':>10s}")
        print(header)
        print("-" * len(header))
        for res in all_results:
            dist = res["error_distribution"]
            total = res["total_errors"] or 1
            row = f"{res['baseline']:20s}"
            for lbl in labels:
                count = dist.get(lbl, 0)
                pct = count / total * 100
                row += f"{pct:>10.1f}%  " if lbl != "other" else f"{pct:>8.1f}%"
            print(row)


if __name__ == "__main__":
    main()
