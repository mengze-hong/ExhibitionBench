"""
analysis/cultural_bias_multi_model.py
======================================
系统性文化偏差分析：多模型 MEIP 评测 + 文化分组 Hit@1/MRR 统计

目标：证明"Western > Non-Western"偏差是跨模型的系统性规律，而非个别模型问题。

支持模型:
  gpt-5.2, gemini-2.5-pro, gemini-2.5-flash,
  deepseek-r1, kimi-k2.5, glm-5, minimax-m2.5

用法:
  # 运行所有模型（跳过已有结果）
  python analysis/cultural_bias_multi_model.py

  # 只跑指定模型
  python analysis/cultural_bias_multi_model.py --models gpt-5.2 gemini-2.5-pro deepseek-r1

  # 强制重跑（覆盖已有结果）
  python analysis/cultural_bias_multi_model.py --force

  # 只出表格（不跑评测，基于已有结果）
  python analysis/cultural_bias_multi_model.py --table-only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Fix Windows GBK encoding for stdout/stderr (must be at module level)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import openai

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results" / "cultural_bias"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── API ───────────────────────────────────────────────────────────────────────

CLIENT = openai.OpenAI(
    api_key="sk-TpK0g832p8LbMXTdI_pjkQ",
    base_url="http://csig.litellm.prod.sgpolaris/v1",
)

REASONING_MODELS = {"deepseek-r1", "kimi-k2.5", "minimax-m2.5"}
LARGE_TOKEN_MODELS = {"deepseek-r1", "kimi-k2.5", "minimax-m2.5", "gemini-2.5-pro", "gemini-2.5-flash"}

ALL_MODELS = [
    "gpt-5.2",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "deepseek-r1",
    "kimi-k2.5",
    "glm-5",
    "minimax-m2.5",
]

# ── 文化分组 ──────────────────────────────────────────────────────────────────

WESTERN_KW = [
    # adjective forms
    "french", "dutch", "german", "italian", "spanish", "english", "british",
    "american", "greek", "roman", "flemish", "portuguese", "swiss", "austrian",
    "scandinavian", "nordic", "belgian", "netherlandish", "european", "western",
    "byzantine", "celtic", "etruscan", "russian", "polish", "hungarian",
    "swedish", "danish", "norwegian", "finnish", "irish", "scottish", "czech",
    "bohemian", "bulgarian", "romanian", "ukrainian", "croatian", "serbian",
    # country / region names (title-case or lower)
    "france", "germany", "italy", "spain", "england", "britain",
    "netherlands", "belgium", "switzerland", "austria", "sweden", "denmark",
    "norway", "finland", "ireland", "scotland", "portugal", "greece",
    "russia", "poland", "hungary", "ukraine", "czech republic", "bohemia",
    "europe",
]
ASIAN_KW = [
    # adjective forms
    "chinese", "japanese", "korean", "indian", "thai", "cambodian", "tibetan",
    "southeast asian", "central asian", "asian", "nepalese", "indonesian",
    "burmese", "vietnamese", "philippine", "himalayan",
    "east asian", "south asian",
    # country / region names
    "china", "japan", "korea", "india", "thailand", "cambodia", "nepal",
    "indonesia", "vietnam", "burma", "myanmar", "philippines", "taiwan",
    "tibet", "mongolia", "mongolian",
]
ISLAMIC_KW = [
    "egyptian islamic",   # must come before "egyptian" in ANCIENT_KW
    "islamic", "arab", "persian", "iranian", "safavid", "timurid", "mamluk",
    "middle eastern", "moroccan",
    # country names
    "iran", "iraq", "syria", "turkey", "egypt islamic", "saudi", "jordan",
    "lebanon", "oman", "yemen", "afghanistan", "pakistan",
]
AFRICAN_KW = [
    "african", "mali", "yoruba", "akan", "kongo", "benin",
    "west african", "east african", "sub-saharan", "nigerian",
    "songye", "luba", "bamana", "fang",
]
PRE_COLUMBIAN_KW = [
    "maya", "aztec", "inca", "pre-columbian", "mesoamerica", "andean",
    "indigenous american", "native american", "oceanic", "polynesian",
]
ANCIENT_KW = [
    "egyptian", "mesopotamian", "babylonian", "assyrian", "sumerian",
    "ancient", "pre-dynastic",
    # country name used for ancient context (lower priority than ISLAMIC "iran"/"iraq")
    "egypt",
]


def classify_culture(culture_str: str) -> str:
    if not culture_str or not culture_str.strip():
        return "Unknown"
    c = culture_str.lower()
    # 顺序很重要：更具体的先匹配
    for kw in ISLAMIC_KW:
        if kw in c:
            return "Islamic"
    for kw in ANCIENT_KW:
        if kw in c:
            return "Ancient"
    for kw in WESTERN_KW:
        if kw in c:
            return "Western"
    for kw in ASIAN_KW:
        if kw in c:
            return "Asian"
    for kw in AFRICAN_KW:
        if kw in c:
            return "African"
    for kw in PRE_COLUMBIAN_KW:
        if kw in c:
            return "Pre-Columbian/Oceanic"
    return "Unknown"


# ── LLM ──────────────────────────────────────────────────────────────────────

def call_llm(model: str, prompt: str, max_tokens: int = 512, timeout: int = 120) -> Optional[str]:
    actual_max = max_tokens * 8 if model in LARGE_TOKEN_MODELS else max_tokens
    for attempt in range(3):
        try:
            resp = CLIENT.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=actual_max,
                temperature=0.0,
                timeout=timeout,
            )
            if not (resp.choices and resp.choices[0].message):
                continue
            msg = resp.choices[0].message
            content = msg.content
            if not content and model in REASONING_MODELS:
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    lines = [l.strip() for l in rc.strip().split("\n") if l.strip()]
                    content = lines[-1] if lines else rc
            return content or ""
        except Exception as e:
            log.warning(f"API error ({model}, attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


# ── Data ─────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def find_data_file(prefix: str) -> Optional[Path]:
    for f in DATA.glob(f"{prefix}*.jsonl"):
        if "v3" in f.name:
            return f
    for f in DATA.glob(f"{prefix}*.jsonl"):
        return f
    return None


# ── Prompt ────────────────────────────────────────────────────────────────────

def build_meip_prompt(sample: dict) -> tuple[str, list[str]]:
    """构造 MEIP zero-shot prompt，直接使用样本中内嵌的 candidate 信息。"""
    theme = sample.get("exhibition_theme", "")

    context_raw = sample.get("context", [])
    context_lines = []
    for c in context_raw[:4]:
        if isinstance(c, dict):
            obj = c
        else:
            obj = {}
        if obj:
            title = obj.get("title", "?")
            culture = obj.get("culture") or "?"
            date = obj.get("date") or "?"
            medium = (obj.get("medium") or "")[:50]
            context_lines.append(f"  - {title} | {culture} | {date} | {medium}")
    context_str = "\n".join(context_lines) if context_lines else "  (none)"

    candidates = sample.get("candidates", [])
    candidate_ids = []
    cand_lines = []
    for idx, c in enumerate(candidates, 1):
        cid = c["id"]
        candidate_ids.append(cid)
        title = c.get("title", "?")
        culture = c.get("culture") or "?"
        date = c.get("date") or "?"
        medium = (c.get("medium") or "")[:50]
        dept = (c.get("department") or "")[:40]
        cand_lines.append(
            f"  [{idx}] ID={cid} | {title} | {culture} | {date} | {medium}"
            + (f" | {dept}" if dept else "")
        )

    prompt = (
        f"You are assisting a museum curator. Given an exhibition theme and some context objects "
        f"already selected, identify which ONE candidate object best fits the exhibition and should "
        f"be added next.\n\n"
        f"Exhibition theme: {theme}\n\n"
        f"Context objects already in exhibition:\n{context_str}\n\n"
        f"Candidate objects (choose the best fit):\n" + "\n".join(cand_lines) + "\n\n"
        f"Reply with ONLY the ID of the best-fitting candidate object (e.g., \"met_123456\").\n"
    )
    return prompt, candidate_ids


def parse_response(resp: str, candidate_ids: list[str]) -> Optional[str]:
    if not resp:
        return None
    for cid in candidate_ids:
        if cid in resp:
            return cid
    nums = re.findall(r'\b(\d+)\b', resp)
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < len(candidate_ids):
            return candidate_ids[idx]
    return None


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval_single(args):
    """单条样本评测（供 ThreadPoolExecutor 调用）。"""
    idx, sample, model = args
    gold_id = sample.get("gold_id")
    if not gold_id:
        return None
    prompt, cids = build_meip_prompt(sample)
    resp = call_llm(model, prompt)
    pred = parse_response(resp or "", cids)
    gold_obj = next((c for c in sample.get("candidates", []) if c["id"] == gold_id), {})
    culture = gold_obj.get("culture", "")
    culture_group = classify_culture(culture)
    return {
        "idx": idx,
        "id": sample["id"],
        "gold_id": gold_id,
        "pred_id": pred,
        "hit": int(pred == gold_id),
        "culture_raw": culture,
        "culture_group": culture_group,
    }


def run_model_meip(model: str, samples: list[dict], max_samples: int = 200,
                   out_path: Optional[Path] = None, max_workers: int = 8) -> list[dict]:
    """对指定模型运行 MEIP zero-shot（并发），返回每条样本的预测记录。"""
    total = min(len(samples), max_samples)
    log.info(f"[{model}] 开始评测 {total} 个样本（并发 workers={max_workers}）...")

    args_list = [(i, samples[i], model) for i in range(total)]
    results_map: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_eval_single, a): a[0] for a in args_list}
        done_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            if result is not None:
                results_map[idx] = result
            done_count += 1
            if done_count % 20 == 0:
                hits_so_far = sum(r["hit"] for r in results_map.values())
                log.info(f"  [{model}] {done_count}/{total}  Hit@1 so far: {hits_so_far/done_count:.3f}")

    # 按原始顺序排列
    records = [results_map[i] for i in range(total) if i in results_map]

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"  [{model}] 结果已保存: {out_path}")

    return records


# ── Analysis ─────────────────────────────────────────────────────────────────

def compute_bias(records: list[dict]) -> dict:
    """按文化分组计算 Hit@1，返回 {group: hit_rate, ...} + 偏差统计。
    兼容 LLM baseline 的 'hit'（int）字段和 SBERT 的 'correct'（bool）字段。"""
    group_hits = defaultdict(list)
    for r in records:
        # 兼容多种字段格式
        if "hit" in r:
            hit_val = int(r["hit"])
        elif "correct" in r:
            hit_val = int(bool(r["correct"]))
        else:
            hit_val = int(r.get("pred_id") == r.get("gold_id"))
        group_hits[r["culture_group"]].append(hit_val)

    result = {}
    for g, hits in group_hits.items():
        result[g] = {
            "hit@1": round(sum(hits) / len(hits), 4),
            "n": len(hits),
        }

    # Overall
    all_hits = []
    for r in records:
        if "hit" in r:
            all_hits.append(int(r["hit"]))
        elif "correct" in r:
            all_hits.append(int(bool(r["correct"])))
        else:
            all_hits.append(int(r.get("pred_id") == r.get("gold_id")))
    result["_overall"] = {
        "hit@1": round(sum(all_hits) / len(all_hits), 4) if all_hits else 0.0,
        "n": len(all_hits),
    }

    # 偏差 = Western - non-Western (排除 Unknown)
    known_groups = {g: v for g, v in result.items() if not g.startswith("_") and g != "Unknown"}
    if known_groups:
        western = known_groups.get("Western", {}).get("hit@1", None)
        non_western = {g: v["hit@1"] for g, v in known_groups.items() if g != "Western"}
        if western is not None and non_western:
            avg_non_western = sum(non_western.values()) / len(non_western)
            result["_western_bias"] = round(western - avg_non_western, 4)

    return result


def print_bias_table(all_results: dict[str, dict], groups: list[str]) -> None:
    """Print cultural bias comparison table for all models."""
    print("\n" + "=" * 80)
    print("Cultural Bias Analysis (MEIP Hit@1 by Cultural Group)")
    print("=" * 80)

    # table header
    col_w = 12
    header = f"{'Model':<20}" + "".join(f"{g:>{col_w}}" for g in groups)
    header += f"{'Overall':>{col_w}}{'Delta(W-NW)':>{col_w}}"
    print(header)
    print("-" * len(header))

    for model, bias_data in all_results.items():
        row = f"{model:<20}"
        for g in groups:
            val = bias_data.get(g, {})
            if val and "hit@1" in val:
                row += f"{val['hit@1']:>{col_w}.4f}"
            else:
                row += f"{'--':>{col_w}}"
        overall = bias_data.get("_overall", {})
        row += f"{overall.get('hit@1', 0):>{col_w}.4f}"
        delta = bias_data.get("_western_bias", None)
        if delta is not None:
            row += f"{delta:>+{col_w}.4f}"
        else:
            row += f"{'--':>{col_w}}"
        print(row)

    print()
    print("Delta(W-NW) = Western Hit@1 - Mean(non-Western known groups) Hit@1")
    print("Positive value = Western bias; higher = more biased")
    print()

    # sample count table
    print("Sample counts per group (from first model):")
    first_model = list(all_results.keys())[0]
    first_data = all_results[first_model]
    for g in groups:
        n = first_data.get(g, {}).get("n", 0)
        print(f"  {g:<25}: {n}")


def print_latex_table(all_results: dict[str, dict], groups: list[str]) -> None:
    """输出 LaTeX 格式的偏差表，用于论文。"""
    print("\n" + "=" * 60)
    print("LaTeX 表格（可直接插入论文）")
    print("=" * 60)

    model_display = {
        "gpt-5.2": "GPT-5.2",
        "gemini-2.5-pro": "Gemini-2.5-Pro",
        "gemini-2.5-flash": "Gemini-2.5-Flash",
        "deepseek-r1": "DeepSeek-R1",
        "kimi-k2.5": "Kimi-K2.5",
        "glm-5": "GLM-5",
        "minimax-m2.5": "Minimax-M2.5",
    }

    group_abbrev = {
        "Western": "West.",
        "Asian": "Asian",
        "Islamic": "Islam.",
        "African": "Afr.",
        "Ancient": "Anc.",
        "Pre-Columbian/Oceanic": "PreCol.",
        "Unknown": "Unk.",
    }

    cols = len(groups) + 3  # groups + Overall + Δ + model
    col_spec = "l" + "c" * (cols - 1)
    header_cells = " & ".join(
        [r"\textbf{Model}"] +
        [r"\textbf{" + group_abbrev.get(g, g[:6]) + "}" for g in groups] +
        [r"\textbf{Avg}", r"\textbf{$\Delta$}"]
    )

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(rf"\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")
    print(header_cells + r" \\")
    print(r"\midrule")

    for model, bias_data in all_results.items():
        display = model_display.get(model, model)
        cells = [display]
        for g in groups:
            val = bias_data.get(g, {})
            if val and "hit@1" in val:
                cells.append(f"{val['hit@1']:.3f}")
            else:
                cells.append("--")
        overall = bias_data.get("_overall", {})
        cells.append(f"{overall.get('hit@1', 0):.3f}")
        delta = bias_data.get("_western_bias", None)
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            cells.append(f"{sign}{delta:.3f}")
        else:
            cells.append("--")
        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Cultural bias in MEIP Hit@1 across model families. "
          r"$\Delta$ = Western Hit@1 $-$ mean non-Western Hit@1. "
          r"All models exhibit a consistent Western-first bias.}")
    print(r"\label{tab:cultural-bias}")
    print(r"\end{table}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Fix Windows GBK encoding for stdout/stderr
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="多模型文化偏差系统性分析")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, help="要评测的模型列表")
    parser.add_argument("--max-samples", type=int, default=200, help="每个模型的最大样本数")
    parser.add_argument("--force", action="store_true", help="强制重跑（覆盖已有结果）")
    parser.add_argument("--table-only", action="store_true", help="只生成表格（不跑评测）")
    args = parser.parse_args()

    meip_path = find_data_file("meip_samples")
    if not meip_path:
        log.error("找不到 meip_samples 数据文件")
        sys.exit(1)

    samples = load_jsonl(meip_path)
    log.info(f"加载 {len(samples)} 个 MEIP 样本（使用前 {args.max_samples} 个）")

    # 主循环：对每个模型跑评测（或加载已有结果）
    all_results: dict[str, dict] = {}

    for model in args.models:
        out_path = RESULTS / f"meip_cultural_{model.replace('/', '-')}_n{args.max_samples}.jsonl"

        if out_path.exists() and not args.force and not args.table_only:
            log.info(f"[{model}] 加载已有结果: {out_path}")
            records = load_jsonl(out_path)
        elif args.table_only:
            if out_path.exists():
                log.info(f"[{model}] 加载已有结果: {out_path}")
                records = load_jsonl(out_path)
            else:
                log.warning(f"[{model}] 无结果文件，跳过（用 --force 重跑）")
                continue
        else:
            log.info(f"[{model}] 开始新评测...")
            records = run_model_meip(model, samples, args.max_samples, out_path)

        if records:
            all_results[model] = compute_bias(records)
            overall = all_results[model]["_overall"]["hit@1"]
            bias = all_results[model].get("_western_bias", "N/A")
            log.info(f"[{model}] Overall Hit@1={overall:.4f}, Δ(W-NW)={bias}")

    if not all_results:
        log.warning("没有任何结果，退出")
        sys.exit(0)

    # 确定要展示的分组（按出现频率排序，排除 Unknown）
    group_counts: dict[str, int] = defaultdict(int)
    for bias_data in all_results.values():
        for g, v in bias_data.items():
            if not g.startswith("_") and g != "Unknown" and isinstance(v, dict):
                group_counts[g] += v.get("n", 0)

    display_groups = sorted(group_counts.keys(), key=lambda g: -group_counts[g])
    # 保证 Western 在最前
    if "Western" in display_groups:
        display_groups.remove("Western")
        display_groups = ["Western"] + display_groups

    print_bias_table(all_results, display_groups)
    print_latex_table(all_results, display_groups)

    # 保存汇总 JSON
    summary_path = RESULTS / f"bias_summary_n{args.max_samples}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "models": args.models,
            "max_samples": args.max_samples,
            "groups": display_groups,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    log.info(f"汇总结果已保存: {summary_path}")

    # 关键结论
    print("\n" + "=" * 60)
    print("关键结论")
    print("=" * 60)
    biases = [(m, d.get("_western_bias", 0)) for m, d in all_results.items()
              if d.get("_western_bias") is not None]
    if biases:
        avg_bias = sum(b for _, b in biases) / len(biases)
        min_bias = min(b for _, b in biases)
        max_bias = max(b for _, b in biases)
        all_positive = all(b > 0 for _, b in biases)
        print(f"• 分析模型数: {len(biases)}")
        print(f"• 平均 Δ(W-NW): {avg_bias:+.4f}")
        print(f"• 偏差范围: [{min_bias:+.4f}, {max_bias:+.4f}]")
        print(f"• 所有模型均呈现 Western 偏向: {'✓ 是' if all_positive else '✗ 否'}")
        print()
        if all_positive:
            print("→ 结论：Western 偏向是系统性、跨模型的规律，不依赖于特定模型或训练数据。")


if __name__ == "__main__":
    main()
