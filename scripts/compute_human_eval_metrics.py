"""
scripts/compute_human_eval_metrics.py
======================================
读取 human_eval/human_eval_annotations.jsonl，
计算并输出 Human Eval 指标：

  1. 每个模型的 Human Preference Win Rate（在 A/B 对中被评注者选中的比例）
  2. Gold Accuracy（模型答对 gold 的比例，来自 gold_correct_a/b 字段）
  3. Human-Gold Agreement Rate（人类偏好与 gold 准确率的一致性）
  4. 按任务（MEIP / ECD / TES）分组汇总

用法：
    python scripts/compute_human_eval_metrics.py
    python scripts/compute_human_eval_metrics.py --latex
    python scripts/compute_human_eval_metrics.py --input human_eval/my_annotations.jsonl
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
HE_DIR = BASE / "human_eval"
DEFAULT_ANN = HE_DIR / "human_eval_annotations.jsonl"


def load_annotations(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        print(f"[WARN] 文件不存在: {path}")
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    print(f"加载 {len(rows)} 条评注 from {path.name}")
    return rows


def dedup_annotations(rows: list[dict]) -> list[dict]:
    """同一 (task, sample_id) 多次评注时取最新一条。"""
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("task", ""), r.get("sample_id", ""), r.get("idx", 0))
        seen[key] = r
    return list(seen.values())


def compute_metrics(rows: list[dict]) -> dict:
    """
    返回 {
      "total": n,
      "by_task": {
        task: {
          "n": int,
          "models": {model: {"wins": int, "losses": int, "ties": int, "appearances": int,
                             "gold_correct": int, "gold_total": int}},
          "human_gold_agree": int,   # preference 对的 model 也 gold_correct
          "human_gold_total": int,   # 有 gold_correct 信息的评注数
        }
      }
    }
    """
    rows = dedup_annotations(rows)
    total = len(rows)

    by_task: dict[str, dict] = {}

    for r in rows:
        task = r.get("task", "unknown")
        if task not in by_task:
            by_task[task] = {
                "n": 0,
                "models": defaultdict(lambda: {
                    "wins": 0, "losses": 0, "ties": 0, "appearances": 0,
                    "gold_correct": 0, "gold_total": 0
                }),
                "human_gold_agree": 0,
                "human_gold_total": 0,
            }
        td = by_task[task]
        td["n"] += 1

        ma = r.get("model_a", "?")
        mb = r.get("model_b", "?")
        pref = r.get("preference", "")  # "A", "B", "tie"

        # --- 出场次数 ---
        td["models"][ma]["appearances"] += 1
        td["models"][mb]["appearances"] += 1

        # --- win/tie/loss ---
        if pref == "A":
            td["models"][ma]["wins"] += 1
            td["models"][mb]["losses"] += 1
        elif pref == "B":
            td["models"][mb]["wins"] += 1
            td["models"][ma]["losses"] += 1
        elif "tie" in pref.lower() or "平" in pref:
            td["models"][ma]["ties"] += 1
            td["models"][mb]["ties"] += 1

        # --- gold correct ---
        gca = r.get("gold_correct_a")
        gcb = r.get("gold_correct_b")
        if gca is not None:
            td["models"][ma]["gold_total"] += 1
            if gca:
                td["models"][ma]["gold_correct"] += 1
        if gcb is not None:
            td["models"][mb]["gold_total"] += 1
            if gcb:
                td["models"][mb]["gold_correct"] += 1

        # --- human-gold agreement ---
        # agreement = 人类选了 gold_correct=True 的那个模型
        if pref in ("A", "B") and gca is not None and gcb is not None:
            td["human_gold_total"] += 1
            preferred_correct = gca if pref == "A" else gcb
            if preferred_correct:
                td["human_gold_agree"] += 1

    return {"total": total, "by_task": by_task}


def print_report(metrics: dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"Human Eval Metrics  (总评注 {metrics['total']} 条)")
    print(sep)

    for task, td in metrics["by_task"].items():
        n = td["n"]
        print(f"\n[{task.upper()}]  n={n} 条评注")
        print("-" * 60)

        # header
        print(f"  {'Model':<30} {'出场':>5} {'Win%':>7} {'Gold%':>7} {'H-G一致%':>9}")
        print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*7} {'-'*9}")

        # sort by win rate descending
        def sort_key(item):
            m, md = item
            app = md["appearances"]
            return md["wins"] / app if app else 0.0

        for m, md in sorted(td["models"].items(), key=sort_key, reverse=True):
            app = md["appearances"]
            win_rate  = md["wins"]  / app        if app else 0.0
            gold_acc  = md["gold_correct"] / md["gold_total"] if md["gold_total"] else None
            gold_str  = f"{gold_acc:.1%}" if gold_acc is not None else "  N/A"

            # Human-Gold agreement (per-model: 被选中时 gold 是否 correct)
            # This is a simplified per-model view
            # Full aggregate is below
            print(
                f"  {m:<30} {app:>5} {win_rate:>7.1%} {gold_str:>7}     —"
            )

        # Overall human-gold agreement
        hgt = td["human_gold_total"]
        if hgt > 0:
            agree_rate = td["human_gold_agree"] / hgt
            print(f"\n  Human-Gold Agreement (人类选择 = gold 正确模型): "
                  f"{td['human_gold_agree']}/{hgt} = {agree_rate:.1%}")
        else:
            print(f"\n  Human-Gold Agreement: N/A (gold_correct 字段缺失)")

    print(f"\n{sep}\n")


def generate_latex(metrics: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Human evaluation results. Win Rate = fraction of A/B comparisons "
        r"where annotators preferred this model. Gold Acc.\ = fraction of samples "
        r"where the model's answer matched the gold label. H-G Agree.\ = fraction of "
        r"comparisons where the human-preferred model also had the correct gold answer.}",
        r"\label{tab:human-eval}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Model} & \textbf{Win Rate} & \textbf{Gold Acc.} & \textbf{H-G Agree.} \\",
        r"\midrule",
    ]

    for task, td in metrics["by_task"].items():
        n = td["n"]
        hgt = td["human_gold_total"]
        agree_str = (
            f"{td['human_gold_agree'] / hgt:.1%}"
            if hgt > 0 else "N/A"
        )

        def sort_key(item):
            m, md = item
            app = md["appearances"]
            return md["wins"] / app if app else 0.0

        sorted_models = sorted(td["models"].items(), key=sort_key, reverse=True)
        first = True
        for m, md in sorted_models:
            app = md["appearances"]
            win_rate = md["wins"] / app if app else 0.0
            gold_acc = (
                f"{md['gold_correct'] / md['gold_total']:.1%}"
                if md["gold_total"] > 0 else "N/A"
            )
            task_label = f"\\textsc{{{task.upper()}}} ($n$={n})" if first else ""
            first = False
            agree_col = agree_str if first is False and m == sorted_models[0][0] else ""
            # Use agree_str only once per task block in first row
            lines.append(
                f"{task_label} & {m} & {win_rate:.1%} & {gold_acc} & {agree_col} \\\\"
            )
        lines.append(r"\midrule")

    # Remove last \midrule, replace with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="计算 Human Eval 指标")
    parser.add_argument("--input", default=str(DEFAULT_ANN),
                        help="评注文件路径（默认 human_eval/human_eval_annotations.jsonl）")
    parser.add_argument("--latex", action="store_true",
                        help="同时输出 LaTeX 表格到 human_eval/latex_human_eval.tex")
    args = parser.parse_args()

    rows = load_annotations(Path(args.input))
    if not rows:
        print("无评注数据，退出。")
        return

    metrics = compute_metrics(rows)
    print_report(metrics)

    if args.latex:
        latex = generate_latex(metrics)
        out = HE_DIR / "latex_human_eval.tex"
        out.write_text(latex, encoding="utf-8")
        print(f"[OK] LaTeX 已保存 -> {out}")
        print("\n--- LaTeX 预览 ---")
        print(latex)


if __name__ == "__main__":
    main()
