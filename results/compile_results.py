"""
results/compile_results.py
===========================
汇总所有 baseline 的评测结果，生成 main_table.csv。

使用方法：
  python results/compile_results.py
"""
from __future__ import annotations
import json
import csv
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# 配置：各 baseline 的预测文件
# ─────────────────────────────────────────────────────────────────────────────
MEIP_CONFIGS = [
    ("Random",           None),  # 特殊处理，直接给随机基线值
    ("BM25",             "results/bm25_meip_pred.jsonl"),
    ("SBERT",            "results/sbert_meip_pred.jsonl"),
    ("GPT-5.2 0-shot",   "results/zeroshot_meip_pred.jsonl"),
    ("GPT-5.2 few-shot", "results/gpt5_fewshot_meip_pred.jsonl"),
    ("RAG+KG",           "results/rag_kg_meip_pred.jsonl"),
]

TES_CONFIGS = [
    ("BM25",             "results/bm25_tes_pred.jsonl"),
    ("SBERT",            "results/sbert_tes_pred.jsonl"),
    ("GPT-5.2 0-shot",   "results/zeroshot_tes_pred.jsonl"),
    ("GPT-5.2 few-shot", "results/gpt5_fewshot_tes_pred.jsonl"),
]


def run_meip_eval(pred_path: Path, gold_path: Path) -> dict:
    """运行 meip_eval.py 返回指标 dict。"""
    result = subprocess.run(
        [sys.executable, str(BASE / "benchmark/meip_eval.py"), "eval",
         "--gold", str(gold_path), "--pred", str(pred_path)],
        capture_output=True, text=True
    )
    metrics = {}
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        for key in ("Acc@1", "Hits@3", "Hits@5", "MRR"):
            if f"{key}" in line and ":" in line:
                try:
                    val = float(line.split(":")[-1].strip())
                    metrics[key] = val
                except ValueError:
                    pass
    return metrics


def run_tes_eval(pred_path: Path, gold_path: Path) -> dict:
    """运行 tes_eval.py 返回指标 dict。"""
    result = subprocess.run(
        [sys.executable, str(BASE / "benchmark/tes_eval.py"), "eval",
         "--gold", str(gold_path), "--pred", str(pred_path), "--k", "5", "10"],
        capture_output=True, text=True
    )
    metrics = {}
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        for key in ("F1@10", "F1@5", "NDCG@10", "NDCG@5", "P@10", "P@5", "R@10", "R@5"):
            if key in line and ":" in line:
                try:
                    val = float(line.split(":")[-1].strip())
                    metrics[key] = val
                except ValueError:
                    pass
    return metrics


def main():
    gold_meip = BASE / "data/meip_samples.jsonl"
    gold_tes  = BASE / "data/tes_samples.jsonl"

    print("\n" + "=" * 70)
    print("MEIP 结果")
    print("=" * 70)
    meip_rows = []
    for name, pred_rel in MEIP_CONFIGS:
        if pred_rel is None:
            # Random baseline
            metrics = {"Acc@1": 0.100, "Hits@3": 0.300, "Hits@5": 0.500, "MRR": 0.293}
        else:
            pred_path = BASE / pred_rel
            if not pred_path.exists():
                print(f"  跳过 {name}（文件不存在: {pred_rel}）")
                continue
            metrics = run_meip_eval(pred_path, gold_meip)
        row = {"Baseline": name, **metrics}
        meip_rows.append(row)
        print(f"  {name:20s} Acc@1={metrics.get('Acc@1','N/A'):.4f}  MRR={metrics.get('MRR','N/A'):.4f}  "
              f"Hits@3={metrics.get('Hits@3','N/A'):.4f}  Hits@5={metrics.get('Hits@5','N/A'):.4f}")

    print("\n" + "=" * 70)
    print("TES 结果")
    print("=" * 70)
    tes_rows = []
    for name, pred_rel in TES_CONFIGS:
        pred_path = BASE / pred_rel
        if not pred_path.exists():
            print(f"  跳过 {name}（文件不存在: {pred_rel}）")
            continue
        metrics = run_tes_eval(pred_path, gold_tes)
        row = {"Baseline": name, **metrics}
        tes_rows.append(row)
        print(f"  {name:20s} NDCG@10={metrics.get('NDCG@10','N/A'):.4f}  "
              f"F1@10={metrics.get('F1@10','N/A'):.4f}  P@10={metrics.get('P@10','N/A'):.4f}  "
              f"R@10={metrics.get('R@10','N/A'):.4f}")

    # 写 CSV
    results_dir = BASE / "results"
    results_dir.mkdir(exist_ok=True)

    # MEIP CSV
    if meip_rows:
        meip_csv = results_dir / "meip_main_table.csv"
        meip_keys = ["Baseline", "Acc@1", "Hits@3", "Hits@5", "MRR"]
        with open(meip_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=meip_keys, extrasaction="ignore")
            writer.writeheader()
            for row in meip_rows:
                writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                                  for k, v in row.items()})
        print(f"\nMEIP 表格 → {meip_csv}")

    # TES CSV
    if tes_rows:
        tes_csv = results_dir / "tes_main_table.csv"
        tes_keys = ["Baseline", "P@5", "R@5", "F1@5", "NDCG@5", "P@10", "R@10", "F1@10", "NDCG@10"]
        with open(tes_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=tes_keys, extrasaction="ignore")
            writer.writeheader()
            for row in tes_rows:
                writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                                  for k, v in row.items()})
        print(f"TES 表格 → {tes_csv}")


if __name__ == "__main__":
    main()
