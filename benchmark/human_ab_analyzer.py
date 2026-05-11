#!/usr/bin/env python3
"""
ExhibitionBench Human A/B Preference Analyzer
读取填好的标注文件，计算 Inter-annotator Agreement / Win Rate / Elo / 相关性。

用法（单标注员）：
    python benchmark/human_ab_analyzer.py

用法（多标注员）：
    python benchmark/human_ab_analyzer.py \\
        benchmark/annotation_tasks_ann1.jsonl \\
        benchmark/annotation_tasks_ann2.jsonl

如不传参数，默认读取 benchmark/annotation_tasks.jsonl。
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── 路径常量 ────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\mengzehong\Desktop\展览馆llm")
BENCHMARK = BASE / "benchmark"
RESULTS   = BASE / "results" / "cultural_bias"

DEFAULT_ANN_FILE = BENCHMARK / "annotation_tasks.jsonl"

# Elo 超参
ELO_INIT = 1200
ELO_K    = 32

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_annotator_file(path: Path) -> dict:
    """返回 {task_id: annotator_choice}，忽略 choice=null 的条目。"""
    items = load_jsonl(path)
    mapping = {}
    for it in items:
        tid    = it.get("task_id")
        choice = it.get("annotator_choice")
        if tid and choice in ("A", "B", "Tie"):
            mapping[tid] = choice
    return mapping


# ── Inter-annotator Agreement (Cohen's κ) ────────────────────────────────────

LABELS = ("A", "Tie", "B")


def cohens_kappa(ann1: dict, ann2: dict) -> Optional[float]:
    """
    对共同标注的条目计算 Cohen's κ。
    若共同条目不足 2 条，返回 None。
    """
    common = sorted(set(ann1) & set(ann2))
    n = len(common)
    if n < 2:
        return None

    # 混淆矩阵
    mat = {(a, b): 0 for a in LABELS for b in LABELS}
    for tid in common:
        mat[(ann1[tid], ann2[tid])] += 1

    # Observed agreement
    p_o = sum(mat[(l, l)] for l in LABELS) / n

    # Expected agreement
    p_e = 0.0
    for l in LABELS:
        freq_1 = sum(mat[(l, b)] for b in LABELS) / n
        freq_2 = sum(mat[(a, l)] for a in LABELS) / n
        p_e   += freq_1 * freq_2

    if abs(1 - p_e) < 1e-9:
        return 1.0 if p_o >= 1.0 - 1e-9 else None

    return (p_o - p_e) / (1 - p_e)


def agreement_summary(annotators: dict[str, dict]) -> None:
    """打印所有标注员两两之间的 Cohen's κ。"""
    names = sorted(annotators)
    if len(names) < 2:
        print("[IAA] 标注员不足 2 名，跳过 Inter-annotator Agreement 计算。\n")
        return

    print("=" * 55)
    print("Inter-annotator Agreement (Cohen's κ)")
    print("=" * 55)
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            kappa = cohens_kappa(annotators[n1], annotators[n2])
            if kappa is None:
                print(f"  {n1} ↔ {n2}: 共同条目不足，无法计算")
            else:
                level = "优秀(>0.8)" if kappa > 0.8 else (
                        "良好(0.6-0.8)" if kappa > 0.6 else (
                        "中等(0.4-0.6)" if kappa > 0.4 else "较低(<0.4)"))
                print(f"  {n1} ↔ {n2}: κ = {kappa:.4f}  [{level}]")
    print()


# ── Win Rate 矩阵 ─────────────────────────────────────────────────────────────

def compute_win_rates(tasks: list, choices: dict) -> dict:
    """
    对每个 comparison 分组，统计 (system_a_model vs system_b_model) 的
    Win / Tie / Loss 次数和胜率。

    返回:
        { group: { "a_model": str, "b_model": str,
                   "n": int, "a_win": int, "tie": int, "b_win": int } }
    """
    stats: dict = {}
    for task in tasks:
        tid   = task["task_id"]
        group = task["comparison"]
        ch    = choices.get(tid)
        if ch is None:
            continue
        if group not in stats:
            stats[group] = {
                "a_model": task["system_a"]["model"],
                "b_model": task["system_b"]["model"],
                "n": 0, "a_win": 0, "tie": 0, "b_win": 0,
            }
        s = stats[group]
        s["n"] += 1
        if ch == "A":
            s["a_win"] += 1
        elif ch == "B":
            s["b_win"] += 1
        else:
            s["tie"] += 1
    return stats


def print_win_rate_table(stats: dict) -> None:
    print("=" * 60)
    print("Win Rate 矩阵")
    print("=" * 60)
    header = f"{'对比组':<32} {'N':>4}  {'A Win%':>7}  {'Tie%':>6}  {'B Win%':>7}"
    print(header)
    print("-" * 60)
    for group, s in stats.items():
        n    = s["n"]
        if n == 0:
            continue
        a_pct  = s["a_win"] / n * 100
        tie_pct = s["tie"]  / n * 100
        b_pct  = s["b_win"] / n * 100
        label  = f"{s['a_model']} vs {s['b_model']}"[:31]
        print(f"{label:<32} {n:>4}  {a_pct:>6.1f}%  {tie_pct:>5.1f}%  {b_pct:>6.1f}%")
    print()


def latex_win_rate(stats: dict) -> str:
    """生成可直接插入论文的 LaTeX booktabs 表格。"""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Human A/B Preference: Win Rate Matrix}")
    lines.append(r"  \label{tab:human_ab_winrate}")
    lines.append(r"  \begin{tabular}{lrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Comparison} & \textbf{N} & \textbf{A Win\%} & \textbf{Tie\%} & \textbf{B Win\%} \\")
    lines.append(r"    \midrule")
    for group, s in stats.items():
        n = s["n"]
        if n == 0:
            continue
        a_pct  = s["a_win"] / n * 100
        tie_pct = s["tie"]  / n * 100
        b_pct  = s["b_win"] / n * 100
        # 转义特殊字符
        label = group.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
        lines.append(
            f"    {label} & {n} & {a_pct:.1f} & {tie_pct:.1f} & {b_pct:.1f} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Elo 排名 ──────────────────────────────────────────────────────────────────

def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update_elo(ra: float, rb: float, score_a: float) -> tuple[float, float]:
    """score_a: 1=A赢, 0=A输, 0.5=平局"""
    ea    = expected_score(ra, rb)
    new_a = ra + ELO_K * (score_a - ea)
    new_b = rb + ELO_K * ((1 - score_a) - (1 - ea))
    return new_a, new_b


def compute_elo(tasks: list, choices: dict) -> dict[str, float]:
    """计算所有模型（包括 human-curator / gold）的 Elo 分。"""
    elo: dict[str, float] = defaultdict(lambda: float(ELO_INIT))

    for task in tasks:
        tid = task["task_id"]
        ch  = choices.get(tid)
        if ch is None:
            continue
        model_a = task["system_a"]["model"]
        model_b = task["system_b"]["model"]

        score_a = 1.0 if ch == "A" else (0.5 if ch == "Tie" else 0.0)

        new_a, new_b = update_elo(elo[model_a], elo[model_b], score_a)
        elo[model_a] = new_a
        elo[model_b] = new_b

    return dict(elo)


def print_elo_ranking(elo: dict) -> None:
    print("=" * 40)
    print(f"Elo 排名  (初始={ELO_INIT}, K={ELO_K})")
    print("=" * 40)
    ranked = sorted(elo.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (model, score) in enumerate(ranked, 1):
        print(f"  #{rank:2d}  {model:<30s} {score:.1f}")
    print()


def latex_elo(elo: dict) -> str:
    ranked = sorted(elo.items(), key=lambda kv: kv[1], reverse=True)
    lines  = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Elo Ratings from Human A/B Preference}")
    lines.append(r"  \label{tab:elo}")
    lines.append(r"  \begin{tabular}{clr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Rank} & \textbf{Model} & \textbf{Elo} \\")
    lines.append(r"    \midrule")
    for rank, (model, score) in enumerate(ranked, 1):
        m_tex = model.replace("_", r"\_").replace("&", r"\&")
        lines.append(f"    {rank} & {m_tex} & {score:.1f} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── 与自动指标的 Spearman ρ ───────────────────────────────────────────────────

def load_hit_at_1(model_slug: str, result_file: str) -> Optional[float]:
    """从 results/cultural_bias/ 读取某模型的 Hit@1。"""
    path = RESULTS / result_file
    if not path.exists():
        return None
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        return None
    return sum(it.get("hit", 0) for it in items) / len(items)


# 已知模型结果文件映射
KNOWN_RESULT_FILES = {
    "gpt-5.2":        "meip_cultural_gpt-5.2_n200.jsonl",
    "gemini-2.5-pro": "meip_cultural_gemini-2.5-pro_n200.jsonl",
    "sbert":          "meip_cultural_sbert_n200.jsonl",
}


def spearman_rank_corr(xs: list, ys: list) -> Optional[float]:
    """Spearman ρ，不依赖 scipy。"""
    n = len(xs)
    if n < 3:
        return None

    def rank_list(lst):
        sorted_items = sorted(enumerate(lst), key=lambda iv: iv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_items[j+1][1] == sorted_items[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j+1):
                ranks[sorted_items[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = rank_list(xs)
    ry = rank_list(ys)
    n_f = float(n)
    mean_rx = sum(rx) / n_f
    mean_ry = sum(ry) / n_f
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mean_rx) ** 2 for i in range(n)) *
        sum((ry[i] - mean_ry) ** 2 for i in range(n))
    )
    if abs(den) < 1e-12:
        return None
    return num / den


def compute_auto_correlation(elo: dict) -> None:
    """
    计算人类偏好（Elo）与各模型自动 Hit@1 的 Spearman ρ。
    只对两者都有数据的模型计算。
    """
    hit_scores: dict[str, float] = {}
    for model, fname in KNOWN_RESULT_FILES.items():
        h = load_hit_at_1(model, fname)
        if h is not None:
            hit_scores[model] = h

    # 取两者共有的模型
    common_models = sorted(set(elo.keys()) & set(hit_scores.keys()))
    if len(common_models) < 3:
        print("[Correlation] 共有数据点不足 3 个，无法计算有效相关性。")
        print(f"  Elo 模型: {sorted(elo.keys())}")
        print(f"  Hit@1 模型: {sorted(hit_scores.keys())}\n")
        return

    elo_vals  = [elo[m]        for m in common_models]
    hit_vals  = [hit_scores[m] for m in common_models]

    rho = spearman_rank_corr(elo_vals, hit_vals)

    print("=" * 55)
    print("自动指标（Hit@1）与人类偏好（Elo）的 Spearman ρ")
    print("=" * 55)
    print(f"  参与模型: {', '.join(common_models)}")
    print(f"  Elo  : {[f'{v:.1f}' for v in elo_vals]}")
    print(f"  Hit@1: {[f'{v:.3f}' for v in hit_vals]}")
    if rho is not None:
        print(f"  Spearman ρ = {rho:.4f}")
    else:
        print("  Spearman ρ = 无法计算（方差为零）")
    print()


# ── 多标注员合并：多数投票 ────────────────────────────────────────────────────

def merge_majority(annotators: dict[str, dict]) -> dict:
    """
    合并多名标注员结果（多数投票），若平局则按 A > Tie > B 优先。
    """
    if not annotators:
        return {}
    all_ids = set()
    for ann in annotators.values():
        all_ids |= set(ann.keys())

    merged = {}
    for tid in all_ids:
        votes = [ann[tid] for ann in annotators.values() if tid in ann]
        if not votes:
            continue
        cnt = {v: votes.count(v) for v in LABELS}
        # 多数
        best = max(LABELS, key=lambda l: (cnt[l], -LABELS.index(l)))
        merged[tid] = best
    return merged


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main(ann_files: list[Path]) -> None:
    if not ann_files:
        ann_files = [DEFAULT_ANN_FILE]

    # 1. 加载标注结果
    annotators: dict[str, dict] = {}
    base_tasks: list = []

    for fp in ann_files:
        if not fp.exists():
            print(f"[WARN] 标注文件不存在，跳过: {fp}")
            continue
        items = load_jsonl(fp)
        if not base_tasks:
            base_tasks = items
        name = fp.stem   # e.g. annotation_tasks_ann1
        ann  = {}
        for it in items:
            tid    = it.get("task_id")
            choice = it.get("annotator_choice")
            if tid and choice in ("A", "B", "Tie"):
                ann[tid] = choice
        annotators[name] = ann
        print(f"[OK]  {name}: {len(ann)} 条有效标注  ({fp})")

    if not annotators:
        print("[ERROR] 无可用标注文件，退出。")
        return

    if not base_tasks:
        print("[ERROR] 无法加载任务列表，退出。")
        return

    print(f"\n[INFO] 共 {len(base_tasks)} 个标注任务，{len(annotators)} 名标注员\n")

    # 2. Inter-annotator Agreement
    agreement_summary(annotators)

    # 3. 合并多人标注（多数投票）
    if len(annotators) == 1:
        merged_choices = next(iter(annotators.values()))
    else:
        merged_choices = merge_majority(annotators)
        n_done = sum(1 for v in merged_choices.values() if v is not None)
        print(f"[INFO] 多数投票合并后: {n_done} 条有效标注\n")

    # 4. Win Rate 矩阵
    win_stats = compute_win_rates(base_tasks, merged_choices)
    print_win_rate_table(win_stats)

    # 5. Elo 排名
    elo = compute_elo(base_tasks, merged_choices)
    print_elo_ranking(elo)

    # 6. 与自动指标相关性
    compute_auto_correlation(elo)

    # 7. 生成 LaTeX 表格并写文件
    latex_wr  = latex_win_rate(win_stats)
    latex_elo_table = latex_elo(elo)

    latex_out = BENCHMARK / "human_ab_latex.tex"
    BENCHMARK.mkdir(parents=True, exist_ok=True)
    with open(latex_out, "w", encoding="utf-8") as f:
        f.write("% ===== Win Rate 矩阵 =====\n")
        f.write(latex_wr)
        f.write("\n\n% ===== Elo 排名 =====\n")
        f.write(latex_elo_table)
        f.write("\n")
    print(f"[OK]  LaTeX 表格 → {latex_out}")

    # 8. 汇总统计写 JSON
    n_done     = len(merged_choices)
    n_total    = len(base_tasks)
    coverage   = n_done / n_total * 100 if n_total else 0.0
    summary = {
        "total_tasks":    n_total,
        "annotated":      n_done,
        "coverage_pct":   round(coverage, 2),
        "n_annotators":   len(annotators),
        "elo":            {m: round(v, 2) for m, v in elo.items()},
        "win_rates":      {
            grp: {
                "n":      s["n"],
                "a_model": s["a_model"],
                "b_model": s["b_model"],
                "a_win_pct":  round(s["a_win"] / s["n"] * 100, 2) if s["n"] else 0,
                "tie_pct":    round(s["tie"]   / s["n"] * 100, 2) if s["n"] else 0,
                "b_win_pct":  round(s["b_win"] / s["n"] * 100, 2) if s["n"] else 0,
            }
            for grp, s in win_stats.items() if s["n"] > 0
        },
    }
    stats_out = BENCHMARK / "human_ab_stats.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK]  汇总统计 → {stats_out}")
    print(f"\n[INFO] 标注覆盖率: {n_done}/{n_total} ({coverage:.1f}%)")


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else []
    main(paths)
