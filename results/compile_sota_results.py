"""
results/compile_sota_results.py
================================
汇总所有 sota_eval.py 输出的 JSON 结果，生成完整的 main_table（含 latency + token cost）。

使用方法：
  python results/compile_sota_results.py
  python results/compile_sota_results.py --shot 0 1 3 5   # 比较不同 shot
  python results/compile_sota_results.py --latex           # 输出 LaTeX 表格
"""
from __future__ import annotations
import json
import csv
import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent

MODEL_DISPLAY = {
    "gpt-5.2":              "GPT-5.2",
    "gpt-5":                "GPT-5",
    "gpt-5.1":              "GPT-5.1",
    "claude-opus-4.6":      "Claude Opus 4.6",
    "claude-opus-4.5":      "Claude Opus 4.5",
    "claude-sonnet-4.5":    "Claude Sonnet 4.5",
    "gemini-2.5-pro":       "Gemini 2.5 Pro",
    "gemini-2.5-flash":     "Gemini 2.5 Flash",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "deepseek-r1":          "DeepSeek-R1",
    "deepseek-v3.2":        "DeepSeek-V3.2",
    "deepseek-v3.1":        "DeepSeek-V3.1",
    "deepseek-v3":          "DeepSeek-V3",
    "kimi-k2.5":            "Kimi K2.5",
    "doubao-seed-2.0-pro":  "Doubao Seed 2.0 Pro",
    "doubao-seed-2.0-lite": "Doubao Seed 2.0 Lite",
    "doubao-seed-1.6":      "Doubao Seed 1.6",
    "doubao-seed-1.6-251015": "Doubao Seed 1.6 (251015)",
    # "doubao-seed-1.6-thinking": endpoint never ran, removed 2026-05-13
    # "doubao-seed-1.6-lite": "Doubao Seed 1.6 Lite",        # endpoint closed
    # "doubao-seed-1.6-flash-250715": "Doubao Seed 1.6 Flash",  # endpoint closed
    "glm-5":                "GLM-5",
    "minimax-m2.5":         "MiniMax M2.5",
    # Open-weight models via xty.app
    "llama-3.3-70b":          "Llama-3.3-70B",
    "llama-3.1-70b-instruct": "Llama-3.1-70B",
    "llama-3.1-8b-instruct":  "Llama-3.1-8B",
    "qwen2.5-72b-instruct":   "Qwen2.5-72B",
    "qwen2.5-7b-instruct":    "Qwen2.5-7B",
    "qwen3-8b":               "Qwen3-8B",
    "qwen3-14b":              "Qwen3-14B",
}

# Preferred display order
MODEL_ORDER = [
    "gpt-5.2", "gpt-5", "gpt-5.1",
    "claude-opus-4.6", "claude-opus-4.5", "claude-sonnet-4.5",
    "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-3.1-pro-preview",
    "deepseek-r1", "deepseek-v3.2", "deepseek-v3.1", "deepseek-v3",
    "doubao-seed-2.0-pro", "doubao-seed-2.0-lite",
    "doubao-seed-1.6", "doubao-seed-1.6-251015",
    # "doubao-seed-1.6-thinking",  # no results, removed 2026-05-13
    # "doubao-seed-1.6-lite", "doubao-seed-1.6-flash-250715",  # endpoints closed
    "kimi-k2.5", "glm-5", "minimax-m2.5",
    # Open-weight models via xty.app
    "llama-3.3-70b", "llama-3.1-70b-instruct", "llama-3.1-8b-instruct",
    "qwen2.5-72b-instruct", "qwen2.5-7b-instruct",
    "qwen3-8b", "qwen3-14b",
]


def load_results(shot: int) -> dict[str, dict[str, dict]]:
    """
    Returns: {model: {task: result_dict}}
    Scans all files matching {task}_{model}_shot{shot}.json
    For TES, prefers the _noleak variant when available (eliminates candidate-label leakage).
    """
    data: dict[str, dict[str, dict]] = {}
    for path in RESULTS_DIR.glob(f"*_shot{shot}.json"):
        fname = path.stem  # e.g. "meip_gpt-5.2_shot0"
        # Skip _noleak files — they are loaded via preference logic below
        if "_noleak" in fname:
            continue
        # Remove _shot{n} suffix
        stem = fname.rsplit("_shot", 1)[0]  # "meip_gpt-5.2"
        # Split task prefix
        for task in ("meip", "tes", "ecd"):
            if stem.startswith(task + "_"):
                model = stem[len(task)+1:]
                # For TES: prefer _noleak variant if it exists
                if task == "tes":
                    noleak_path = path.with_stem(path.stem + "_noleak")
                    if noleak_path.exists():
                        path = noleak_path
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                except Exception as e:
                    import warnings
                    warnings.warn(f"[WARN] Failed to load {path}: {e}")
                    continue
                if model not in data:
                    data[model] = {}
                data[model][task] = result
                break
    return data


# Models with pending few-shot results (shown as "TBC" in shot>0 tables)
# Format: {model_id: {task, ...}}
TBC_FEW_SHOT: dict[str, set] = {
    "kimi-k2.5": {"tes"},
}


def fmt(val, decimals=4):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def fmt_tbc(val, model: str, task: str, shot: int, decimals=4):
    """Like fmt() but shows 'TBC' for pending few-shot cells instead of '—'."""
    if val is None and shot > 0:
        if model in TBC_FEW_SHOT and task in TBC_FEW_SHOT[model]:
            return "TBC"
    return fmt(val, decimals)


def fmt_int(val):
    if val is None or val == 0:
        return "—"
    if val >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val/1_000:.0f}K"
    return str(val)


def print_main_table(data: dict, shot: int) -> None:
    print(f"\n{'='*110}")
    print(f"ExhibitionBench Results (shot={shot})")
    print(f"{'='*110}")

    header = (
        f"{'Model':<22} | "
        f"{'MEIP':^17} | "
        f"{'TES':^17} | "
        f"{'ECD':^39} | "
        f"{'Latency(s)':^12} | "
        f"{'Tokens (K)':^14}"
    )
    sub = (
        f"{'':22} | "
        f"{'MRR':>8} {'Hit@1':>8} | "
        f"{'NDCG@10':>8} {'MRR':>8} | "
        f"{'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8} {'Macro':>8} | "
        f"{'avg/call':>12} | "
        f"{'total':>8} {'avg':>5}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sub)
    print(sep)

    # Collect all models seen — skip vision variants, failed models (n=0 across all tasks), open-weight aliases
    SKIP_SUFFIXES = ("_vision", "_v3fixed", "_v4clean", "_noleak", "_cot")
    all_models = [m for m in data.keys()
                  if not any(m.endswith(s) for s in SKIP_SUFFIXES)]
    # Also skip models where all tasks have n_samples == 0
    all_models = [m for m in all_models
                  if any(t.get("n_samples", 0) > 0 for t in data[m].values())]
    # Use Qwen2.5-7B canonical name if present
    if "Qwen_Qwen2_5-7B-Instruct" in all_models:
        all_models.remove("Qwen_Qwen2_5-7B-Instruct")
    ordered = [m for m in MODEL_ORDER if m in all_models]
    ordered += [m for m in all_models if m not in ordered]

    for model in ordered:
        tasks = data[model]
        mr = tasks.get("meip", {})
        tr = tasks.get("tes", {})
        er = tasks.get("ecd", {})

        # Compute combined avg latency and tokens
        total_lat = sum(
            t.get("total_latency_sec", 0) for t in tasks.values()
        )
        total_n = sum(
            t.get("n_samples", 0) for t in tasks.values()
        )
        avg_lat = total_lat / total_n if total_n > 0 else 0

        total_tok = sum(
            t.get("total_tokens", 0) for t in tasks.values()
        )
        avg_tok_per_call = total_tok / total_n if total_n > 0 else 0

        display = MODEL_DISPLAY.get(model, model)
        line = (
            f"{display:<22} | "
            f"{fmt(mr.get('mrr')):>8} {fmt(mr.get('hit@1')):>8} | "
            f"{fmt_tbc(tr.get('ndcg@10'), model, 'tes', shot):>8} "
            f"{fmt_tbc(tr.get('mrr'), model, 'tes', shot):>8} | "
            f"{fmt(er.get('pairaccc_L1')):>8} "
            f"{fmt(er.get('pairaccc_L2')):>8} "
            f"{fmt(er.get('pairaccc_L3')):>8} "
            f"{fmt(er.get('pairaccc_L4')):>8} "
            f"{fmt(er.get('macro_pairaccc')):>8} | "
            f"{avg_lat:>12.2f} | "
            f"{fmt_int(total_tok):>8} {fmt_int(int(avg_tok_per_call)):>5}"
        )
        print(line)
    print(sep)


def print_latency_cost_table(data: dict, shot: int) -> None:
    """Detailed latency and token cost breakdown per model per task."""
    print(f"\n{'='*100}")
    print(f"Inference Latency & Token Cost Breakdown (shot={shot})")
    print(f"{'='*100}")
    header = (
        f"{'Model':<22} {'Task':<6} | "
        f"{'n':>5} {'total_lat(s)':>12} {'avg_lat(s)':>10} | "
        f"{'prompt_tok':>10} {'compl_tok':>10} {'total_tok':>10}"
    )
    print(header)
    print("-" * len(header))

    SKIP_SUFFIXES = ("_vision", "_v3fixed", "_v4clean", "_noleak", "_cot")
    all_models = [m for m in data.keys()
                  if not any(m.endswith(s) for s in SKIP_SUFFIXES)
                  and any(t.get("n_samples", 0) > 0 for t in data[m].values())
                  and m != "Qwen_Qwen2_5-7B-Instruct"]
    ordered = [m for m in MODEL_ORDER if m in all_models]
    ordered += [m for m in all_models if m not in ordered]

    for model in ordered:
        tasks = data[model]
        display = MODEL_DISPLAY.get(model, model)
        for task in ["meip", "tes", "ecd"]:
            r = tasks.get(task, {})
            if not r:
                continue
            n = r.get("n_samples", 0)
            tl = r.get("total_latency_sec", 0)
            al = r.get("avg_latency_sec", 0)
            pt = r.get("total_prompt_tokens", 0)
            ct = r.get("total_completion_tokens", 0)
            tt = r.get("total_tokens", 0)
            print(
                f"{display:<22} {task.upper():<6} | "
                f"{n:>5} {tl:>12.1f} {al:>10.3f} | "
                f"{fmt_int(pt):>10} {fmt_int(ct):>10} {fmt_int(tt):>10}"
            )
        print()


def export_csv(data: dict, shot: int) -> None:
    out = RESULTS_DIR / f"sota_main_table_shot{shot}.csv"
    rows = []
    SKIP_SUFFIXES = ("_vision", "_v3fixed", "_v4clean", "_noleak", "_cot")
    all_models = [m for m in data.keys()
                  if not any(m.endswith(s) for s in SKIP_SUFFIXES)
                  and any(t.get("n_samples", 0) > 0 for t in data[m].values())
                  and m != "Qwen_Qwen2_5-7B-Instruct"]
    ordered = [m for m in MODEL_ORDER if m in all_models]
    ordered += [m for m in all_models if m not in ordered]

    for model in ordered:
        tasks = data[model]
        mr = tasks.get("meip", {})
        tr = tasks.get("tes", {})
        er = tasks.get("ecd", {})
        total_tok = sum(t.get("total_tokens", 0) for t in tasks.values())
        total_lat = sum(t.get("total_latency_sec", 0) for t in tasks.values())
        total_n   = sum(t.get("n_samples", 0) for t in tasks.values())
        rows.append({
            "model":           MODEL_DISPLAY.get(model, model),
            "shot":            shot,
            "meip_mrr":        mr.get("mrr", ""),
            "meip_hit1":       mr.get("hit@1", ""),
            "meip_n":          mr.get("n_samples", ""),
            "tes_ndcg10":      tr.get("ndcg@10", ""),
            "tes_mrr":         tr.get("mrr", ""),
            "tes_n":           tr.get("n_samples", ""),
            "ecd_L1":          er.get("pairaccc_L1", ""),
            "ecd_L2":          er.get("pairaccc_L2", ""),
            "ecd_L3":          er.get("pairaccc_L3", ""),
            "ecd_L4":          er.get("pairaccc_L4", ""),
            "ecd_macro":       er.get("macro_pairaccc", ""),
            "ecd_n":           er.get("n_samples", ""),
            "total_latency_s": round(total_lat, 1) if total_lat else "",
            "avg_lat_per_call": round(total_lat / total_n, 3) if total_n else "",
            "total_tokens":    total_tok if total_tok else "",
        })

    with open(out, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nCSV saved → {out}")


def export_latex(data: dict, shot: int) -> None:
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{ExhibitionBench Main Results (Zero-shot)}")
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\begin{tabular}{l|cc|cc|ccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & "
        r"\multicolumn{2}{c|}{\textbf{MEIP}} & "
        r"\multicolumn{2}{c|}{\textbf{TES}} & "
        r"\multicolumn{5}{c}{\textbf{ECD}} \\"
    )
    lines.append(
        r" & MRR & Hit@1 & NDCG@10 & MRR & L1 & L2 & L3 & L4 & Macro \\"
    )
    lines.append(r"\midrule")

    SKIP_SUFFIXES = ("_vision", "_v3fixed", "_v4clean", "_noleak", "_cot")
    all_models = [m for m in data.keys()
                  if not any(m.endswith(s) for s in SKIP_SUFFIXES)
                  and any(t.get("n_samples", 0) > 0 for t in data[m].values())
                  and m != "Qwen_Qwen2_5-7B-Instruct"]
    ordered = [m for m in MODEL_ORDER if m in all_models]
    ordered += [m for m in all_models if m not in ordered]

    for model in ordered:
        tasks = data[model]
        mr = tasks.get("meip", {})
        tr = tasks.get("tes", {})
        er = tasks.get("ecd", {})
        display = MODEL_DISPLAY.get(model, model)
        row = (
            f"{display} & "
            f"{fmt(mr.get('mrr'))} & {fmt(mr.get('hit@1'))} & "
            f"{fmt_tbc(tr.get('ndcg@10'), model, 'tes', shot)} & "
            f"{fmt_tbc(tr.get('mrr'), model, 'tes', shot)} & "
            f"{fmt(er.get('pairaccc_L1'))} & {fmt(er.get('pairaccc_L2'))} & "
            f"{fmt(er.get('pairaccc_L3'))} & {fmt(er.get('pairaccc_L4'))} & "
            f"{fmt(er.get('macro_pairaccc'))} \\\\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = RESULTS_DIR / f"latex_main_table_shot{shot}.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLaTeX table saved → {out}")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Compile ExhibitionBench SOTA results")
    parser.add_argument("--shot", type=int, nargs="+", default=[0],
                        help="Shot(s) to compile (default: 0)")
    parser.add_argument("--latex", action="store_true",
                        help="Export LaTeX table")
    parser.add_argument("--csv", action="store_true", default=True,
                        help="Export CSV (default: True)")
    parser.add_argument("--latency", action="store_true", default=True,
                        help="Show latency/cost breakdown (default: True)")
    args = parser.parse_args()

    for shot in args.shot:
        data = load_results(shot)
        if not data:
            print(f"No results found for shot={shot}")
            continue
        print_main_table(data, shot)
        if args.latency:
            print_latency_cost_table(data, shot)
        if args.csv:
            export_csv(data, shot)
        if args.latex:
            export_latex(data, shot)


if __name__ == "__main__":
    main()
