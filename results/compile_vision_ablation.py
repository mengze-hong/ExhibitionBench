"""
results/compile_vision_ablation.py
====================================
汇总 MEIP vision ablation 对比结果。

读取所有 results/meip_*_vision_shot0.json，找到对应 text-only 结果
（优先级：_v4clean → _v3fixed → _shot0），输出对比表 + LaTeX。

用法：
    python results/compile_vision_ablation.py
    python results/compile_vision_ablation.py --latex results/latex_vision_table.tex
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent

DISPLAY = {
    "gpt-5.2":                       "GPT-5.2",
    "gpt-5.1":                       "GPT-5.1",
    "claude-opus-4.6":               "Claude Opus 4.6",
    "claude-opus-4.5":               "Claude Opus 4.5",
    "claude-sonnet-4.5":             "Claude Sonnet 4.5",
    "gemini-2.5-pro":                "Gemini 2.5 Pro",
    "gemini-2.5-flash":              "Gemini 2.5 Flash",
    "doubao-seed-2.0-pro":           "Doubao Seed 2.0 Pro",
    "doubao-seed-1.6-vision-250815": "Doubao Seed 1.6 Vision",
    "deepseek-v3.2":                 "DeepSeek-V3.2",
    "kimi-k2.5":                     "Kimi-K2.5",
}

# 期望展示顺序（按能力从强到弱排列）
PREFERRED_ORDER = [
    "gpt-5.2", "claude-opus-4.6", "gemini-2.5-pro", "gemini-2.5-flash",
    "doubao-seed-2.0-pro", "doubao-seed-1.6-vision-250815",
    "gpt-5.1", "claude-sonnet-4.5", "deepseek-v3.2", "kimi-k2.5",
]

# vision model → text-only model name mapping (when names differ)
TEXT_MODEL_ALIAS = {
    "doubao-seed-1.6-vision-250815": "doubao-seed-1.6",
}


def find_text_result(model: str) -> dict | None:
    """按优先级查找 text-only 结果文件（考虑视觉模型名称别名）。"""
    text_model = TEXT_MODEL_ALIAS.get(model, model)
    for suffix in ["_v4clean", "_v3fixed", ""]:
        p = RESULTS_DIR / f"meip_{text_model}_shot0{suffix}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_vision_results() -> dict[str, dict]:
    """加载所有 vision 结果，返回 {model: result_dict}。"""
    out = {}
    for p in RESULTS_DIR.glob("meip_*_vision_shot0.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        model = d.get("model", p.stem.replace("meip_", "").replace("_vision_shot0", ""))
        out[model] = d
    return out


def build_rows(vision_results: dict[str, dict]) -> list[dict]:
    rows = []
    for model, vd in vision_results.items():
        td = find_text_result(model)
        if td is None:
            print(f"  [WARN] 找不到 text-only 结果：{model}，跳过")
            continue

        vis_mrr  = vd.get("mrr",   0.0)
        vis_h1   = vd.get("hit@1", 0.0)
        n_vis    = vd.get("n_samples", 0)

        if n_vis == 0:
            print(f"  [WARN] vision 结果为空（n=0）：{model}，跳过")
            continue
        if n_vis < 200:
            print(f"  [WARN] vision 样本数过少（n={n_vis}）：{model}，跳过（需要 n>=200 才可信）")
            continue

        txt_mrr = td.get("mrr",   0.0)
        txt_h1  = td.get("hit@1", 0.0)

        rows.append({
            "model":    model,
            "display":  DISPLAY.get(model, model),
            "txt_mrr":  txt_mrr,
            "vis_mrr":  vis_mrr,
            "d_mrr":    vis_mrr - txt_mrr,
            "txt_h1":   txt_h1,
            "vis_h1":   vis_h1,
            "d_h1":     vis_h1 - txt_h1,
            "n_txt":    td.get("n_samples", 0),
            "n_vis":    n_vis,
        })

    # 按 PREFERRED_ORDER 排序，未列出的追加到末尾
    order_map = {m: i for i, m in enumerate(PREFERRED_ORDER)}
    rows.sort(key=lambda r: order_map.get(r["model"], 999))
    return rows


def print_table(rows: list[dict]) -> None:
    sep  = "-" * 85
    hdr  = f"{'Model':<32} {'Text MRR':>9} {'Vis MRR':>9} {'Δ MRR':>8}  {'Text H@1':>9} {'Vis H@1':>9} {'Δ H@1':>8}"
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        dmrr = f"{r['d_mrr']:+.4f}" if r['d_mrr'] else " 0.0000"
        dh1  = f"{r['d_h1']:+.4f}"  if r['d_h1']  else " 0.0000"
        mrr_tag = " ▲" if r['d_mrr'] > 0.001 else (" ▼" if r['d_mrr'] < -0.001 else "  ")
        print(
            f"{r['display']:<32} {r['txt_mrr']:9.4f} {r['vis_mrr']:9.4f} {dmrr:>8}{mrr_tag}"
            f"  {r['txt_h1']:9.4f} {r['vis_h1']:9.4f} {dh1:>8}"
        )
    print(sep)

    # 总结
    pos_mrr = sum(1 for r in rows if r['d_mrr'] > 0.001)
    neg_mrr = sum(1 for r in rows if r['d_mrr'] < -0.001)
    avg_d   = sum(r['d_mrr'] for r in rows) / len(rows) if rows else 0
    print(f"\n总计 {len(rows)} 个模型：MRR 提升 {pos_mrr} 个，下降 {neg_mrr} 个，平均 Δ={avg_d:+.4f}")


def generate_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{MEIP Vision Ablation: Text-only vs.\ Text+Image. "
        r"$\Delta$ = Vision MRR $-$ Text MRR. "
        r"Most models degrade with image input, suggesting visual noise "
        r"rather than benefit on this cultural-anchoring task.}",
        r"\label{tab:vision-ablation}",
        r"\begin{tabular}{lccc|ccc}",
        r"\toprule",
        r"\textbf{Model} & \multicolumn{3}{c|}{\textbf{MRR}} & \multicolumn{3}{c}{\textbf{Hit@1}} \\",
        r" & Text & +Image & $\Delta$ & Text & +Image & $\Delta$ \\",
        r"\midrule",
    ]

    for r in rows:
        dname = r["display"].replace("_", r"\_")
        dmrr_str = f"{r['d_mrr']:+.3f}"
        dh1_str  = f"{r['d_h1']:+.3f}"
        # 高亮 delta
        if r['d_mrr'] > 0.001:
            dmrr_str = r"\textcolor{ForestGreen}{" + dmrr_str + "}"
        elif r['d_mrr'] < -0.001:
            dmrr_str = r"\textcolor{BrickRed}{" + dmrr_str + "}"
        if r['d_h1'] > 0.001:
            dh1_str = r"\textcolor{ForestGreen}{" + dh1_str + "}"
        elif r['d_h1'] < -0.001:
            dh1_str = r"\textcolor{BrickRed}{" + dh1_str + "}"

        lines.append(
            f"{dname} & {r['txt_mrr']:.3f} & {r['vis_mrr']:.3f} & {dmrr_str} & "
            f"{r['txt_h1']:.3f} & {r['vis_h1']:.3f} & {dh1_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="汇总 MEIP vision ablation 结果")
    parser.add_argument(
        "--latex",
        default=str(RESULTS_DIR / "latex_vision_table.tex"),
        help="LaTeX 输出路径（默认 results/latex_vision_table.tex）",
    )
    parser.add_argument("--no-latex", action="store_true", help="不生成 LaTeX 文件")
    args = parser.parse_args()

    vision = load_vision_results()
    print(f"找到 {len(vision)} 个模型的 vision 结果：{sorted(vision.keys())}\n")

    rows = build_rows(vision)
    if not rows:
        print("没有可对比的行，退出。")
        return

    print_table(rows)

    if not args.no_latex:
        latex = generate_latex(rows)
        out_path = Path(args.latex)
        out_path.write_text(latex, encoding="utf-8")
        print(f"\n[OK] LaTeX 表格已保存 -> {out_path}")
        print("\n--- LaTeX 预览 ---")
        print(latex)


if __name__ == "__main__":
    main()
