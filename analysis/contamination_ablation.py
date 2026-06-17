"""
analysis/contamination_ablation.py — Data Contamination Ablation for ExhibitionBench
======================================================================================
Two ablation experiments to probe potential data contamination:

  C1: Institution Split (Met vs. non-Met)
      - If top models memorized Met Museum data (CC0, widely crawled),
        they should show higher relative performance on Met samples vs.
        performance of weaker models that cannot leverage memorization.
      - Compute MRR_met and MRR_nonmet per model; report the gap.

  C2: Title Masking (full titles vs. anonymized)
      - Replace exhibition/object titles with "[ARTWORK_N]" placeholders.
      - If a model relies on memorized name-exhibition associations, it will
        show a larger MRR drop than models doing pure semantic reasoning.
      - Compare delta_MRR = MRR_full - MRR_masked per model.

Usage:
  # C1 only (fast, uses cached sota_eval runs)
  python analysis/contamination_ablation.py --exp c1 --models gpt-5.2 claude-opus-4.6 gemini-2.5-pro doubao-seed-2.0-pro deepseek-v3.2

  # C2 only (reruns MEIP with masked titles, expensive)
  python analysis/contamination_ablation.py --exp c2 --models gpt-5.2 claude-opus-4.6 gemini-2.5-pro doubao-seed-2.0-pro deepseek-v3.2 --max-samples 200

  # Both experiments
  python analysis/contamination_ablation.py --exp all --models gpt-5.2 claude-opus-4.6 --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import openai

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
ANALYSIS_OUT = BASE / "results" / "contamination"
ANALYSIS_OUT.mkdir(exist_ok=True)

# ── API ───────────────────────────────────────────────────────────────────────
def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


INTERNAL_API_BASE = os.environ.get("LLM_API_BASE", "http://YOUR_LLM_API_BASE").rstrip("/")
INTERNAL_API_KEY = _require_env("LLM_API_KEY")

CLIENT = openai.OpenAI(
    api_key=INTERNAL_API_KEY,
    base_url=f"{INTERNAL_API_BASE}/v1",
)

LARGE_TOKEN_MODELS = {
    "deepseek-r1", "kimi-k2.5", "minimax-m2.5",
    "gemini-2.5-pro", "gemini-2.5-flash",
    "gpt-5", "gpt-5.1", "gpt-5-codex",
}
TEMP1_MODELS = {
    "gpt-5", "gpt-5.1", "gpt-5-codex", "gpt-5.2",
    "deepseek-r1", "kimi-k2.5",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_objects(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in load_jsonl(path)}


def find_data_file(name: str) -> Optional[Path]:
    for suffix in ["_v3_fixed", "_v3", "_v2", ""]:
        p = DATA / f"{name}{suffix}.jsonl"
        if p.exists():
            return p
    return None


def mrr(gold_id: str, ranked: list[str]) -> float:
    try:
        return 1.0 / (ranked.index(gold_id) + 1)
    except ValueError:
        return 0.0


def call_llm(model: str, prompt: str, max_tokens: int = 150) -> tuple[Optional[str], float, dict]:
    actual_max = max_tokens * 8 if model in LARGE_TOKEN_MODELS else max_tokens
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            temp = 1.0 if model in TEMP1_MODELS else 0.0
            resp = CLIENT.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=actual_max,
                temperature=temp,
                timeout=120,
            )
            latency = time.perf_counter() - t0
            usage = empty_usage.copy()
            if resp.usage:
                usage["prompt_tokens"]     = resp.usage.prompt_tokens or 0
                usage["completion_tokens"] = resp.usage.completion_tokens or 0
                usage["total_tokens"]      = resp.usage.total_tokens or 0
            if not (resp.choices and resp.choices[0].message):
                continue
            return (resp.choices[0].message.content or ""), latency, usage
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
        except Exception as e:
            log.warning(f"API error ({model}, attempt {attempt+1}): {e}")
            time.sleep(1)
    return None, 0.0, empty_usage


def parse_selection(response: str, candidates: list[str]) -> list[str]:
    resp = response.strip() if response else ""
    found = [c for c in candidates if c in resp]
    if found:
        return found + [c for c in candidates if c not in found]
    import re
    nums = re.findall(r'\b(\d+)\b', resp)
    idx_ranked = []
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < len(candidates) and idx not in idx_ranked:
            idx_ranked.append(idx)
    if idx_ranked:
        result = [candidates[i] for i in idx_ranked]
        result += [c for c in candidates if c not in result]
        return result
    return candidates


# ── MEIP Prompt (reuse from sota_eval) ────────────────────────────────────────

MEIP_ZEROSHOT_TEMPLATE = """\
You are assisting a museum curator. Given an exhibition theme and some context objects already selected, \
identify which ONE candidate object best fits the exhibition and should be added next.

Exhibition theme: {theme}

Context objects already in exhibition:
{context}

Candidate objects (choose the best fit):
{candidates}

Reply with ONLY the ID of the best-fitting candidate object (e.g., "met_123456").
"""


def build_meip_prompt_masked(sample: dict, objects: dict[str, dict], mask_titles: bool = False) -> str:
    """Build MEIP prompt, optionally masking all artwork titles."""
    theme = sample.get("exhibition_theme", "")

    context_raw = sample.get("context", [])
    context_objs = []
    for i, c in enumerate(context_raw[:4]):
        obj = c if isinstance(c, dict) else objects.get(c, {})
        if obj:
            title = f"[ARTWORK_{i+1}]" if mask_titles else obj.get('title', '?')
            context_objs.append(
                f"  - {title} ({obj.get('culture','?')}, {obj.get('date','?')})"
            )
    context_str = "\n".join(context_objs) if context_objs else "  (none)"

    candidates_raw = sample.get("candidates", sample.get("candidate_ids", []))
    candidate_ids = []
    cand_lines = []
    for idx, c in enumerate(candidates_raw, 1):
        cid = c["id"] if isinstance(c, dict) else c
        obj = c if isinstance(c, dict) else objects.get(cid, {})
        candidate_ids.append(cid)
        if obj:
            title = f"[ARTWORK_{idx}]" if mask_titles else obj.get('title', '?')
            cand_lines.append(
                f"  [{idx}] ID={cid} | {title} | "
                f"{obj.get('culture','?')} | {obj.get('date','?')} | "
                f"{(obj.get('medium') or '')[:60]}"
            )
        else:
            cand_lines.append(f"  [{idx}] ID={cid}")
    candidates_str = "\n".join(cand_lines)

    return MEIP_ZEROSHOT_TEMPLATE.format(
        theme=theme,
        context=context_str,
        candidates=candidates_str,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# C1: Institution Split
# ═══════════════════════════════════════════════════════════════════════════════

def run_c1_institution_split(models: list[str], max_samples_per_split: int = 200, workers: int = 100):
    """
    For each model, compute MRR separately on Met and non-Met MEIP samples.
    Uses ThreadPoolExecutor(max_workers=workers) for fast parallel inference.
    """
    log.info("=== C1: Institution Split Ablation ===")
    log.info("Met Museum is CC0, widely indexed -- tests if models memorized specific associations")

    # Load data
    meip_path = find_data_file("meip_samples")
    obj_path  = find_data_file("objects")
    if not meip_path or not obj_path:
        log.error("Cannot find MEIP/objects data files")
        return

    all_samples = load_jsonl(meip_path)
    objects     = load_objects(obj_path)

    # Load exhibition source map
    exh_src: dict[str, str] = {}
    exh_path = find_data_file("exhibitions")
    if exh_path:
        for e in load_jsonl(exh_path):
            exh_src[e["id"]] = e.get("source", "unknown")

    # Split samples by institution
    met_samples    = [s for s in all_samples if exh_src.get(s.get("exhibition_id", ""), "?") == "met"]
    nonmet_samples = [s for s in all_samples if exh_src.get(s.get("exhibition_id", ""), "?") != "met"
                      and exh_src.get(s.get("exhibition_id", ""), "?") not in ("unknown", "?", "NOT_FOUND")]

    log.info(f"Met samples: {len(met_samples)}, Non-Met samples: {len(nonmet_samples)}")

    # Cap to balance
    met_cap    = min(len(met_samples), max_samples_per_split)
    nonmet_cap = min(len(nonmet_samples), max_samples_per_split)
    met_subset    = met_samples[:met_cap]
    nonmet_subset = nonmet_samples[:nonmet_cap]

    all_results = {}

    for model in models:
        log.info(f"\n--- Model: {model} ---")

        # Check if already cached
        cache_path = ANALYSIS_OUT / f"c1_{model}.json"
        if cache_path.exists():
            log.info(f"  Loaded from cache: {cache_path}")
            all_results[model] = json.loads(cache_path.read_text(encoding="utf-8"))
            continue

        results = {}
        for split_name, split_samples in [("met", met_subset), ("nonmet", nonmet_subset)]:
            log.info(f"  Running {split_name} split ({len(split_samples)} samples) with {workers} workers ...")

            def _infer_one(sample, _model=model):
                gold_id = sample.get("gold_id", "")
                candidates_raw = sample.get("candidates", sample.get("candidate_ids", []))
                candidate_ids  = [c["id"] if isinstance(c, dict) else c for c in candidates_raw]
                if not gold_id or not candidate_ids:
                    return None
                prompt   = build_meip_prompt_masked(sample, objects, mask_titles=False)
                response, latency, usage = call_llm(_model, prompt)
                if response is None:
                    return None
                ranked = parse_selection(response, candidate_ids)
                return mrr(gold_id, ranked)

            mrr_scores = []
            done_count = [0]
            lock = Lock()

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_infer_one, s): s for s in split_samples}
                for fut in as_completed(futs):
                    res = fut.result()
                    with lock:
                        done_count[0] += 1
                        if res is not None:
                            mrr_scores.append(res)
                        if done_count[0] % 50 == 0:
                            cur_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
                            log.info(f"  C1 {split_name} {model}: {done_count[0]}/{len(split_samples)}, "
                                     f"MRR={cur_mrr:.4f}")

            n = len(mrr_scores)
            results[split_name] = {
                "n_samples": n,
                "mrr": round(sum(mrr_scores) / n, 4) if n else 0,
            }
            log.info(f"  {split_name}: n={n}, MRR={results[split_name]['mrr']:.4f}")

        results["mrr_gap"] = round(results["met"]["mrr"] - results["nonmet"]["mrr"], 4)
        log.info(f"  MRR gap (Met - NonMet): {results['mrr_gap']:+.4f}")
        all_results[model] = results
        cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary table
    print("\n" + "="*65)
    print("C1: INSTITUTION SPLIT -- Data Contamination Probe")
    print("="*65)
    print(f"{'Model':<25} {'MRR_Met':>8} {'MRR_NonMet':>11} {'Gap':>8}  Interpretation")
    print("-"*65)
    for model, r in all_results.items():
        gap = r.get("mrr_gap", 0)
        interp = "HIGH RISK" if gap > 0.05 else ("MODERATE" if gap > 0.02 else "LOW")
        print(f"{model:<25} {r['met']['mrr']:>8.4f} {r['nonmet']['mrr']:>11.4f} {gap:>+8.4f}  {interp}")
    print("="*65)
    print("Interpretation: Large positive gap (Met >> NonMet) = contamination risk")
    print("  Compare: top models should show LARGER gap than weaker models if contaminated")

    # Save overall results
    out_path = ANALYSIS_OUT / "c1_summary.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"\nSaved C1 results to {out_path}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# C2: Title Masking
# ═══════════════════════════════════════════════════════════════════════════════

def run_c2_title_masking(models: list[str], max_samples: int = 200, workers: int = 100):
    """
    Compare MRR with full titles vs. anonymized titles ([ARTWORK_N] placeholders).
    A large MRR drop signals the model relies on memorized title-exhibition associations.
    Uses ThreadPoolExecutor(max_workers=workers) for fast parallel inference.
    """
    log.info("\n=== C2: Title Masking Ablation ===")
    log.info("Tests if top models rely on memorized title associations vs. semantic reasoning")

    meip_path = find_data_file("meip_samples")
    obj_path  = find_data_file("objects")
    if not meip_path or not obj_path:
        log.error("Cannot find data files")
        return

    all_samples = load_jsonl(meip_path)[:max_samples]
    objects     = load_objects(obj_path)

    all_results = {}

    for model in models:
        log.info(f"\n--- Model: {model} ---")
        cache_path = ANALYSIS_OUT / f"c2_{model}.json"
        if cache_path.exists():
            log.info(f"  Loaded from cache: {cache_path}")
            all_results[model] = json.loads(cache_path.read_text(encoding="utf-8"))
            continue

        results = {}
        for mask_mode, mask_titles in [("full", False), ("masked", True)]:
            log.info(f"  Running {mask_mode} mode ({len(all_samples)} samples) with {workers} workers ...")

            def _infer_one(sample, _model=model, _mask=mask_titles):
                gold_id = sample.get("gold_id", "")
                candidates_raw = sample.get("candidates", sample.get("candidate_ids", []))
                candidate_ids  = [c["id"] if isinstance(c, dict) else c for c in candidates_raw]
                if not gold_id or not candidate_ids:
                    return None
                prompt   = build_meip_prompt_masked(sample, objects, mask_titles=_mask)
                response, latency, usage = call_llm(_model, prompt)
                if response is None:
                    return None
                ranked = parse_selection(response, candidate_ids)
                return mrr(gold_id, ranked)

            mrr_scores = []
            done_count = [0]
            lock = Lock()

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_infer_one, s): s for s in all_samples}
                for fut in as_completed(futs):
                    res = fut.result()
                    with lock:
                        done_count[0] += 1
                        if res is not None:
                            mrr_scores.append(res)
                        if done_count[0] % 50 == 0:
                            cur_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
                            log.info(f"  C2 {mask_mode} {model}: {done_count[0]}/{len(all_samples)}, "
                                     f"MRR={cur_mrr:.4f}")

            n = len(mrr_scores)
            results[mask_mode] = {
                "n_samples": n,
                "mrr": round(sum(mrr_scores) / n, 4) if n else 0,
            }
            log.info(f"  {mask_mode}: n={n}, MRR={results[mask_mode]['mrr']:.4f}")

        results["mrr_drop"] = round(results["full"]["mrr"] - results["masked"]["mrr"], 4)
        results["mrr_drop_pct"] = round(100 * results["mrr_drop"] / (results["full"]["mrr"] + 1e-9), 1)
        log.info(f"  MRR drop (full - masked): {results['mrr_drop']:+.4f} ({results['mrr_drop_pct']:.1f}%)")

        all_results[model] = results
        cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary table
    print("\n" + "="*70)
    print("C2: TITLE MASKING -- Data Contamination Probe")
    print("="*70)
    print(f"{'Model':<25} {'MRR_Full':>9} {'MRR_Masked':>11} {'Drop':>7} {'Drop%':>7}  Risk")
    print("-"*70)
    for model, r in all_results.items():
        drop = r.get("mrr_drop", 0)
        droppct = r.get("mrr_drop_pct", 0)
        risk = "HIGH" if droppct > 8 else ("MODERATE" if droppct > 4 else "LOW")
        print(f"{model:<25} {r['full']['mrr']:>9.4f} {r['masked']['mrr']:>11.4f} {drop:>+7.4f} {droppct:>6.1f}%  {risk}")
    print("="*70)
    print("Interpretation: Large drop (>8%) when titles masked = memorized associations")
    print("  If top models drop MORE than weaker models -> contamination evidence")

    out_path = ANALYSIS_OUT / "c2_summary.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"\nSaved C2 results to {out_path}")
    return all_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data contamination ablation experiments")
    parser.add_argument("--exp", choices=["c1", "c2", "all"], default="all",
                        help="Which experiment to run (default: all)")
    parser.add_argument("--models", nargs="+",
                        default=["gpt-5.2", "claude-opus-4.6", "gemini-2.5-pro",
                                 "doubao-seed-2.0-pro", "deepseek-v3.2", "gemini-2.5-flash"],
                        help="Models to evaluate")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max MEIP samples per split/condition (default: 200)")
    parser.add_argument("--workers", type=int, default=100,
                        help="ThreadPoolExecutor max_workers (default: 100)")
    args = parser.parse_args()

    log.info(f"Models: {args.models}")
    log.info(f"Max samples: {args.max_samples}")
    log.info(f"Workers: {args.workers}")
    log.info(f"Output dir: {ANALYSIS_OUT}")

    if args.exp in ("c1", "all"):
        run_c1_institution_split(args.models, max_samples_per_split=args.max_samples,
                                 workers=args.workers)

    if args.exp in ("c2", "all"):
        run_c2_title_masking(args.models, max_samples=args.max_samples,
                             workers=args.workers)


if __name__ == "__main__":
    main()
