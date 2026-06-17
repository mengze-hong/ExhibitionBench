"""
analysis/fewshot_mechanism.py — Few-shot Degradation Mechanism Analysis
=======================================================================
Tests three hypotheses explaining why GPT few-shot < zero-shot on MEIP/TES:

  H1 Cultural Anchoring: few-shot examples bias toward specific cultures
  H2 Context Overload:   more shots hurt performance (inverted-U curve)
  H3 Format Conformity:  model learns format, not content (shuffled label test)

Usage:
  python analysis/fewshot_mechanism.py --model gpt-5.2 --task meip
  python analysis/fewshot_mechanism.py --model gpt-5.2 --task meip --shots 0 1 2 3 5
  python analysis/fewshot_mechanism.py --model gpt-5.2 --task meip --h3-shuffled
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import openai

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results" / "fewshot_analysis"
RESULTS.mkdir(parents=True, exist_ok=True)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

# ── API Config ────────────────────────────────────────────────────────────────
API_BASE = os.environ.get("LLM_API_BASE", "http://YOUR_LLM_API_BASE").rstrip("/")
API_KEY = _require_env("LLM_API_KEY")

CLIENT = openai.OpenAI(
    api_key=API_KEY,
    base_url=f"{API_BASE}/v1",
)

REASONING_MODELS = {"deepseek-r1", "kimi-k2.5", "minimax-m2.5"}
LARGE_TOKEN_MODELS = {"deepseek-r1", "kimi-k2.5", "minimax-m2.5", "gemini-2.5-pro", "gemini-2.5-flash"}

# ── LLM Call ─────────────────────────────────────────────────────────────────

def call_llm(model: str, prompt: str, max_tokens: int = 1024, timeout: int = 120) -> Optional[str]:
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
            import time; time.sleep(1)
    return None

# ── Data Loaders ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_objects(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in load_jsonl(path)}

def find_data_file(prefix: str) -> Optional[Path]:
    for f in DATA.glob(f"{prefix}*.jsonl"):
        if "v3" in f.name:
            return f
    for f in DATA.glob(f"{prefix}*.jsonl"):
        return f
    return None

# ── MEIP Few-shot Examples ────────────────────────────────────────────────────

# Western-biased examples (for H1 cultural anchoring test)
WESTERN_EXAMPLES = [
    {
        "theme": "French Impressionism",
        "context": "Monet painting of water lilies; Renoir portrait of Parisians",
        "candidates": [
            ("[A]", "Degas bronze ballerina sculpture", True),
            ("[B]", "Chinese porcelain vase", False),
            ("[C]", "Egyptian hieroglyphic tablet", False),
        ],
        "answer": "A"
    },
    {
        "theme": "Renaissance Masterworks",
        "context": "Raphael Madonna fresco; Leonardo sketches",
        "candidates": [
            ("[A]", "Ottoman miniature painting", False),
            ("[B]", "Michelangelo marble bust", True),
            ("[C]", "Japanese folding screen", False),
        ],
        "answer": "B"
    },
]

# Asian-biased examples (for H1 cultural anchoring test)
ASIAN_EXAMPLES = [
    {
        "theme": "Tang Dynasty Treasures",
        "context": "Tang tri-color pottery horse; silk scroll painting",
        "candidates": [
            ("[A]", "Tang dynasty bronze mirror with floral design", True),
            ("[B]", "Roman marble toga statue", False),
            ("[C]", "Persian carpet fragment", False),
        ],
        "answer": "A"
    },
    {
        "theme": "Japanese Edo Period",
        "context": "Hiroshige woodblock print; Edo-period lacquerware",
        "candidates": [
            ("[A]", "European oil painting", False),
            ("[B]", "Hokusai ukiyo-e print", True),
            ("[C]", "Chinese ink painting", False),
        ],
        "answer": "B"
    },
]

def format_example(ex: dict) -> str:
    cands = "\n".join(f"  {tag} {desc}" for tag, desc, _ in ex["candidates"])
    return (
        f"Exhibition theme: {ex['theme']}\n"
        f"Context: {ex['context']}\n"
        f"Candidates:\n{cands}\n"
        f"Answer: {ex['answer']}\n\n"
    )

def build_meip_prompt_nshot(sample: dict, objects: dict, shot: int = 0,
                             example_set: str = "neutral", shuffled: bool = False,
                             rng: random.Random = None) -> tuple[str, list[str]]:
    """Build MEIP prompt with n-shot examples. Returns (prompt, candidate_ids)."""
    if rng is None:
        rng = random.Random(42)

    theme = sample.get("exhibition_theme", "")
    context_raw = sample.get("context", [])
    context_objs = []
    for c in context_raw[:4]:
        if isinstance(c, dict):
            obj = c
        else:
            obj = objects.get(c, {})
        if obj:
            context_objs.append(f"  - {obj.get('title','?')} ({obj.get('culture','?')}, {obj.get('date','?')})")
    context_str = "\n".join(context_objs) if context_objs else "  (none)"

    candidates_raw = sample.get("candidates", sample.get("candidate_ids", []))
    candidate_ids = []
    cand_lines = []
    for idx, c in enumerate(candidates_raw, 1):
        if isinstance(c, dict):
            cid = c["id"]
            obj = c
        else:
            cid = c
            obj = objects.get(cid, {})
        candidate_ids.append(cid)
        if obj:
            cand_lines.append(
                f"  [{idx}] ID={cid} | {obj.get('title','?')} | "
                f"{obj.get('culture','?')} | {obj.get('date','?')} | "
                f"{(obj.get('medium') or '')[:60]}"
            )
        else:
            cand_lines.append(f"  [{idx}] ID={cid}")

    # Build few-shot prefix
    prefix = ""
    if shot > 0:
        if example_set == "western":
            exs = WESTERN_EXAMPLES[:shot]
        elif example_set == "asian":
            exs = ASIAN_EXAMPLES[:shot]
        else:
            # Neutral: mix of both
            all_exs = WESTERN_EXAMPLES + ASIAN_EXAMPLES
            exs = all_exs[:shot]

        for ex in exs:
            if shuffled:
                # H3: Replace correct answer with random wrong answer
                wrong_answers = [tag for tag, _, correct in ex["candidates"] if not correct]
                if wrong_answers:
                    ex = dict(ex)
                    ex["answer"] = rng.choice(wrong_answers).strip("[]")
            prefix += format_example(ex)

    query = (
        f"You are assisting a museum curator. Given an exhibition theme and some context objects already selected, "
        f"identify which ONE candidate object best fits the exhibition and should be added next.\n\n"
        f"Exhibition theme: {theme}\n\n"
        f"Context objects already in exhibition:\n{context_str}\n\n"
        f"Candidate objects (choose the best fit):\n{'\n'.join(cand_lines)}\n\n"
        f"Reply with ONLY the ID of the best-fitting candidate object (e.g., \"met_123456\").\n"
    )

    return (prefix + query), candidate_ids


def parse_meip_response(resp: str, candidate_ids: list[str], objects: dict) -> Optional[str]:
    """Extract predicted object ID from response."""
    if not resp:
        return None
    resp = resp.strip()
    for cid in candidate_ids:
        if cid in resp:
            return cid
    # Try numbered response
    import re
    nums = re.findall(r'\b(\d+)\b', resp)
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < len(candidate_ids):
            return candidate_ids[idx]
    return None


def compute_mrr_hit1(pred: Optional[str], gold: str, candidate_ids: list[str]) -> tuple[float, float]:
    if pred is None:
        return 0.0, 0.0
    if pred == gold:
        return 1.0, 1.0
    return 0.0, 0.0  # 10-way ranking: if not rank-1 assume last


# ── H2: Shot Count Ablation ───────────────────────────────────────────────────

def run_shot_ablation(model: str, samples: list[dict], objects: dict,
                      shots: list[int], max_samples: int = 100, workers: int = 100) -> dict:
    """H2: Test performance vs. number of shots (0, 1, 2, 3, 5)."""
    results = {}
    for shot in shots:
        log.info(f"Shot ablation: {model}, shot={shot}, n={max_samples}")
        mrrs, hits = [], []
        lock = Lock()
        rng = random.Random(42)
        subset = [s for s in samples[:max_samples] if s.get("gold_id")]

        def _infer(sample, _shot=shot):
            gold = sample["gold_id"]
            prompt, cids = build_meip_prompt_nshot(sample, objects, shot=_shot, rng=random.Random(42))
            resp = call_llm(model, prompt)
            pred = parse_meip_response(resp, cids, objects)
            return compute_mrr_hit1(pred, gold, cids)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mrr_val, hit_val in ex.map(_infer, subset):
                with lock:
                    mrrs.append(mrr_val)
                    hits.append(hit_val)

        results[shot] = {
            "shot": shot,
            "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "hit@1": round(sum(hits) / len(hits), 4) if hits else 0.0,
            "n": len(mrrs),
        }
        log.info(f"  shot={shot}: MRR={results[shot]['mrr']:.4f}, Hit@1={results[shot]['hit@1']:.4f}")
    return results


# ── H1: Cultural Anchoring ────────────────────────────────────────────────────

def run_cultural_anchoring(model: str, samples: list[dict], objects: dict,
                            max_samples: int = 100, workers: int = 100) -> dict:
    """H1: Do Western examples bias toward Western objects?"""
    results = {}

    for condition, shot, example_set in [
        ("zero_shot", 0, "neutral"),
        ("western_2shot", 2, "western"),
        ("asian_2shot", 2, "asian"),
    ]:
        log.info(f"Cultural anchoring: {model}, condition={condition}")
        west_mrrs, east_mrrs = [], []
        lock = Lock()
        subset = [s for s in samples[:max_samples] if s.get("gold_id")]

        def _infer(sample, _shot=shot, _es=example_set):
            gold = sample["gold_id"]
            gold_obj = objects.get(gold, {})
            culture = (gold_obj.get("culture") or "").lower()
            is_western = any(w in culture for w in ["french", "italian", "british", "dutch",
                                                      "german", "american", "european", "greek",
                                                      "roman", "spanish", "france", "italy",
                                                      "germany", "spain", "netherlands", "england",
                                                      "united states", "united kingdom", "austria",
                                                      "belgium", "sweden", "ireland", "russia",
                                                      "netherlandish"])
            is_eastern = any(w in culture for w in ["chinese", "japanese", "korean", "asian",
                                                      "indian", "persian", "islamic", "thai",
                                                      "vietnamese", "tibetan", "china", "japan",
                                                      "korea", "india", "iran", "iranian",
                                                      "maya", "egypt", "egyptian"])
            prompt, cids = build_meip_prompt_nshot(sample, objects, shot=_shot,
                                                    example_set=_es, rng=random.Random(42))
            resp = call_llm(model, prompt)
            pred = parse_meip_response(resp, cids, objects)
            mrr_val, _ = compute_mrr_hit1(pred, gold, cids)
            return mrr_val, is_western, is_eastern

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mrr_val, is_w, is_e in ex.map(_infer, subset):
                with lock:
                    if is_w:
                        west_mrrs.append(mrr_val)
                    elif is_e:
                        east_mrrs.append(mrr_val)

        results[condition] = {
            "western_mrr": round(sum(west_mrrs) / len(west_mrrs), 4) if west_mrrs else None,
            "eastern_mrr": round(sum(east_mrrs) / len(east_mrrs), 4) if east_mrrs else None,
            "western_n": len(west_mrrs),
            "eastern_n": len(east_mrrs),
        }
        log.info(f"  {condition}: Western MRR={results[condition]['western_mrr']}, "
                 f"Eastern MRR={results[condition]['eastern_mrr']}")
    return results


# ── H3: Format Conformity (Shuffled Label) ────────────────────────────────────

def run_shuffled_label(model: str, samples: list[dict], objects: dict,
                        max_samples: int = 100, workers: int = 100) -> dict:
    """H3: Does shuffled label (wrong answer in example) hurt performance?"""
    results = {}
    for condition, shot, shuffled in [
        ("zero_shot", 0, False),
        ("2shot_correct", 2, False),
        ("2shot_shuffled", 2, True),
    ]:
        log.info(f"Shuffled label: {model}, condition={condition}")
        mrrs, hits = [], []
        lock = Lock()
        subset = [s for s in samples[:max_samples] if s.get("gold_id")]

        def _infer(sample, _shot=shot, _shuf=shuffled):
            gold = sample["gold_id"]
            rng = random.Random(42)
            prompt, cids = build_meip_prompt_nshot(sample, objects, shot=_shot,
                                                    shuffled=_shuf, rng=rng)
            resp = call_llm(model, prompt)
            pred = parse_meip_response(resp, cids, objects)
            return compute_mrr_hit1(pred, gold, cids)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mrr_val, hit_val in ex.map(_infer, subset):
                with lock:
                    mrrs.append(mrr_val)
                    hits.append(hit_val)

        results[condition] = {
            "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "hit@1": round(sum(hits) / len(hits), 4) if hits else 0.0,
            "n": len(mrrs),
        }
        log.info(f"  {condition}: MRR={results[condition]['mrr']:.4f}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)

    parser = argparse.ArgumentParser(description="Few-shot Degradation Mechanism Analysis")
    parser.add_argument("--model", default="gpt-5.2", help="Model to evaluate")
    parser.add_argument("--task", default="meip", choices=["meip"])
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 1, 2, 3, 5],
                        help="Shot counts for ablation")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--h1-cultural", action="store_true", help="Run H1 cultural anchoring")
    parser.add_argument("--h2-shots", action="store_true", help="Run H2 shot count ablation")
    parser.add_argument("--h3-shuffled", action="store_true", help="Run H3 shuffled label")
    parser.add_argument("--all", action="store_true", help="Run all 3 hypotheses")
    parser.add_argument("--workers", type=int, default=100, help="Thread pool workers")
    args = parser.parse_args()

    # Load data
    meip_path = find_data_file("meip_samples")
    obj_path = find_data_file("objects")
    if not meip_path or not obj_path:
        log.error("Could not find data files")
        sys.exit(1)

    samples = load_jsonl(meip_path)
    objects = load_objects(obj_path)
    log.info(f"Loaded {len(samples)} MEIP samples, {len(objects)} objects")

    run_h1 = args.h1_cultural or args.all
    run_h2 = args.h2_shots or args.all
    run_h3 = args.h3_shuffled or args.all

    if not (run_h1 or run_h2 or run_h3):
        # Default: run all
        run_h1 = run_h2 = run_h3 = True

    # Load existing results so we can update individual experiments without losing others
    out_path = RESULTS / f"fewshot_{args.model}_{args.task}.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results["model"] = args.model
    all_results["max_samples"] = args.max_samples

    if run_h2:
        log.info("=" * 60)
        log.info("H2: Shot Count Ablation")
        log.info("=" * 60)
        h2 = run_shot_ablation(args.model, samples, objects, args.shots, args.max_samples, args.workers)
        all_results["h2_shot_ablation"] = h2

        print("\n--- H2: Shot Count Ablation ---")
        print(f"{'Shot':>5} | {'MRR':>7} | {'Hit@1':>7} | {'N':>5}")
        print("-" * 35)
        for shot in sorted(h2.keys()):
            r = h2[shot]
            print(f"{shot:>5} | {r['mrr']:>7.4f} | {r['hit@1']:>7.4f} | {r['n']:>5}")

    if run_h1:
        log.info("=" * 60)
        log.info("H1: Cultural Anchoring Analysis")
        log.info("=" * 60)
        h1 = run_cultural_anchoring(args.model, samples, objects, args.max_samples, args.workers)
        all_results["h1_cultural_anchoring"] = h1

        print("\n--- H1: Cultural Anchoring ---")
        print(f"{'Condition':<20} | {'Western MRR':>12} | {'Eastern MRR':>12} | {'West N':>7} | {'East N':>7}")
        print("-" * 65)
        for cond, r in h1.items():
            print(f"{cond:<20} | {str(r['western_mrr']):>12} | {str(r['eastern_mrr']):>12} | "
                  f"{r['western_n']:>7} | {r['eastern_n']:>7}")

    if run_h3:
        log.info("=" * 60)
        log.info("H3: Format Conformity (Shuffled Label)")
        log.info("=" * 60)
        h3 = run_shuffled_label(args.model, samples, objects, args.max_samples, args.workers)
        all_results["h3_shuffled_label"] = h3

        print("\n--- H3: Shuffled Label ---")
        print(f"{'Condition':<20} | {'MRR':>7} | {'Hit@1':>7} | {'N':>5}")
        print("-" * 45)
        for cond, r in h3.items():
            print(f"{cond:<20} | {r['mrr']:>7.4f} | {r['hit@1']:>7.4f} | {r['n']:>5}")

    # Save results
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
