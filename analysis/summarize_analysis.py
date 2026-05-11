"""
analysis/summarize_analysis.py — Summarize all analysis experiment results
Prints formatted tables for H1/H2/H3 fewshot mechanism + metadata ablation.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FEWSHOT = BASE / "results" / "fewshot_analysis"
ABLATION = BASE / "results" / "metadata_ablation"

MODELS = ["gpt-5.2", "claude-opus-4.6", "gemini-2.5-pro", "deepseek-v3.2"]

# ── H2: Shot Ablation ─────────────────────────────────────────────────────────
print("\n" + "="*72)
print("H2: SHOT COUNT ABLATION (MEIP MRR)")
print("="*72)
header = f"{'Model':<22}" + "".join(f" {'Shot'+str(s):>8}" for s in [0,1,2,3,5])
print(header)
print("-"*72)
for model in MODELS:
    p = FEWSHOT / f"fewshot_{model}_meip.json"
    if not p.exists():
        print(f"{model:<22}  (missing)")
        continue
    d = json.loads(p.read_text())
    h2 = d.get("h2_shot_ablation", {})
    row = f"{model:<22}"
    for s in ["0","1","2","3","5"]:
        mrr = h2.get(s, {}).get("mrr")
        row += f" {mrr:>8.4f}" if mrr is not None else f" {'N/A':>8}"
    print(row)
print("="*72)

# ── H1: Cultural Anchoring ────────────────────────────────────────────────────
print("\n" + "="*90)
print("H1: CULTURAL ANCHORING (MEIP MRR by culture group)")
print("="*90)
print(f"{'Model':<22} {'Condition':<16} {'West MRR':>9} {'East MRR':>9} {'n_W':>5} {'n_E':>5} {'Gap':>8}")
print("-"*90)
for model in MODELS:
    p = FEWSHOT / f"fewshot_{model}_meip.json"
    if not p.exists():
        print(f"{model:<22} (missing)")
        continue
    d = json.loads(p.read_text())
    h1 = d.get("h1_cultural_anchoring", {})
    for cond, v in h1.items():
        w = v.get("western_mrr")
        e = v.get("eastern_mrr")
        nw = v.get("western_n", 0)
        ne = v.get("eastern_n", 0)
        gap = (e - w) if (e is not None and w is not None) else None
        w_s = f"{w:.4f}" if w is not None else "N/A"
        e_s = f"{e:.4f}" if e is not None else "N/A"
        gap_s = f"{gap:+.4f}" if gap is not None else "N/A"
        print(f"{model:<22} {cond:<16} {w_s:>9} {e_s:>9} {nw:>5} {ne:>5} {gap_s:>8}")
    print()
print("="*90)

# ── H3: Shuffled Label ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("H3: SHUFFLED LABEL (Format Conformity Bias)")
print("="*70)
print(f"{'Model':<22} {'ZeroShot':>10} {'2shot_correct':>14} {'2shot_shuffled':>15} {'delta':>8}")
print("-"*70)
for model in MODELS:
    p = FEWSHOT / f"fewshot_{model}_meip.json"
    if not p.exists():
        print(f"{model:<22} (missing)")
        continue
    d = json.loads(p.read_text())
    h3 = d.get("h3_shuffled_label", {})
    zs = h3.get("zero_shot", {}).get("mrr")
    c2 = h3.get("2shot_correct", {}).get("mrr")
    s2 = h3.get("2shot_shuffled", {}).get("mrr")
    delta = round(s2 - c2, 4) if (s2 is not None and c2 is not None) else None
    zs_s  = f"{zs:.4f}" if zs is not None else "N/A"
    c2_s  = f"{c2:.4f}" if c2 is not None else "N/A"
    s2_s  = f"{s2:.4f}" if s2 is not None else "N/A"
    dt_s  = f"{delta:+.4f}" if delta is not None else "N/A"
    print(f"{model:<22} {zs_s:>10} {c2_s:>14} {s2_s:>15} {dt_s:>8}")
print("="*70)

# ── Metadata Ablation ─────────────────────────────────────────────────────────
print("\n" + "="*80)
print("METADATA ABLATION (MEIP MRR at each level)")
print("="*80)
print(f"{'Model':<22}" + "".join(f" {'L'+str(i):>8}" for i in range(6)))
print("-"*80)
for model in MODELS:
    p = ABLATION / f"metadata_ablation_{model}.json"
    if not p.exists():
        print(f"{model:<22} (missing)")
        continue
    d = json.loads(p.read_text())
    rs = d.get("results", {})
    row = f"{model:<22}"
    for level in ["L0","L1","L2","L3","L4","L5"]:
        mrr = rs.get(level, {}).get("mrr")
        row += f" {mrr:>8.4f}" if mrr is not None else f" {'N/A':>8}"
    print(row)
print("="*80)
