"""
analysis/metadata_ablation.py — Metadata Field Ablation Study
=============================================================
Tests which metadata fields matter most for MEIP performance.

Ablation levels (from minimal to full):
  L0: title only
  L1: title + date
  L2: title + date + culture
  L3: title + date + culture + medium
  L4: title + date + culture + medium + department
  L5: title + date + culture + medium + department + description (full)

Usage:
  python analysis/metadata_ablation.py --model gpt-5.2 --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

import openai

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results" / "metadata_ablation"
RESULTS.mkdir(parents=True, exist_ok=True)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_BASE = os.environ.get("LLM_API_BASE", "http://YOUR_LLM_API_BASE").rstrip("/")
API_KEY = _require_env("LLM_API_KEY")

CLIENT = openai.OpenAI(
    api_key=API_KEY,
    base_url=f"{API_BASE}/v1",
)

REASONING_MODELS = {"deepseek-r1", "kimi-k2.5", "minimax-m2.5"}
LARGE_TOKEN_MODELS = {"deepseek-r1", "kimi-k2.5", "minimax-m2.5", "gemini-2.5-pro", "gemini-2.5-flash"}

# Metadata levels (what fields are included at each level)
METADATA_LEVELS = {
    "L0": ["title"],
    "L1": ["title", "date"],
    "L2": ["title", "date", "culture"],
    "L3": ["title", "date", "culture", "medium"],
    "L4": ["title", "date", "culture", "medium", "department"],
    "L5": ["title", "date", "culture", "medium", "department", "description"],
}


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


def format_obj_at_level(obj: dict, fields: list[str]) -> str:
    """Format object metadata using only specified fields."""
    parts = []
    for field in fields:
        val = obj.get(field)
        if val and str(val).strip():
            if field == "description":
                parts.append(f"desc: {str(val)[:120]}")
            else:
                parts.append(f"{str(val)[:60]}")
    return " | ".join(parts) if parts else "?"


def build_meip_prompt_ablation(sample: dict, objects: dict, fields: list[str]) -> tuple[str, list[str]]:
    """Build MEIP prompt with specific metadata fields only."""
    theme = sample.get("exhibition_theme", "")

    # Context objects
    context_raw = sample.get("context", [])
    context_lines = []
    for c in context_raw[:4]:
        if isinstance(c, dict):
            obj = c
        else:
            obj = objects.get(c, {})
        if obj:
            context_lines.append(f"  - {format_obj_at_level(obj, fields)}")
    context_str = "\n".join(context_lines) if context_lines else "  (none)"

    # Candidates
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
            cand_lines.append(f"  [{idx}] ID={cid} | {format_obj_at_level(obj, fields)}")
        else:
            cand_lines.append(f"  [{idx}] ID={cid}")

    prompt = (
        f"You are assisting a museum curator. Given an exhibition theme and some context objects already selected, "
        f"identify which ONE candidate object best fits the exhibition and should be added next.\n\n"
        f"Exhibition theme: {theme}\n\n"
        f"Context objects already in exhibition:\n{context_str}\n\n"
        f"Candidate objects (choose the best fit):\n{chr(10).join(cand_lines)}\n\n"
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


def run_ablation(model: str, samples: list[dict], objects: dict,
                 max_samples: int = 100, workers: int = 100) -> dict:
    """Run metadata ablation across all 6 levels using ThreadPoolExecutor."""
    results = {}
    for level, fields in METADATA_LEVELS.items():
        log.info(f"Metadata ablation: {model}, level={level}, fields={fields}")
        mrrs, hits = [], []
        lock = Lock()
        subset = [s for s in samples[:max_samples] if s.get("gold_id")]

        def _infer(sample, _fields=fields):
            gold = sample["gold_id"]
            prompt, cids = build_meip_prompt_ablation(sample, objects, _fields)
            resp = call_llm(model, prompt)
            pred = parse_response(resp, cids)
            mrr_val = 1.0 if pred == gold else 0.0
            return mrr_val

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mrr_val in ex.map(_infer, subset):
                with lock:
                    mrrs.append(mrr_val)
                    hits.append(mrr_val)

        results[level] = {
            "level": level,
            "fields": fields,
            "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "hit@1": round(sum(hits) / len(hits), 4) if hits else 0.0,
            "n": len(mrrs),
        }
        log.info(f"  {level}: MRR={results[level]['mrr']:.4f}, Hit@1={results[level]['hit@1']:.4f}")

    # Compute delta from full (L5)
    if "L5" in results:
        full_mrr = results["L5"]["mrr"]
        for level in results:
            results[level]["delta_from_L5"] = round(results[level]["mrr"] - full_mrr, 4)

    return results


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=100, help="ThreadPoolExecutor max_workers")
    args = parser.parse_args()

    meip_path = find_data_file("meip_samples")
    obj_path = find_data_file("objects")
    if not meip_path or not obj_path:
        log.error("Data files not found")
        sys.exit(1)

    samples = load_jsonl(meip_path)
    objects = load_objects(obj_path)
    log.info(f"Loaded {len(samples)} MEIP samples, {len(objects)} objects")

    results = run_ablation(args.model, samples, objects, args.max_samples, args.workers)

    print(f"\n{'Metadata Ablation Results':^55}")
    print(f"Model: {args.model}, N={args.max_samples}")
    print("=" * 55)
    print(f"{'Level':<6} | {'Fields':<35} | {'MRR':>7} | {'Δ vs L5':>8}")
    print("-" * 55)
    for level in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        if level not in results:
            continue
        r = results[level]
        field_str = "+".join(r["fields"])
        print(f"{level:<6} | {field_str:<35} | {r['mrr']:>7.4f} | {r.get('delta_from_L5', 0):>+8.4f}")

    out_path = RESULTS / f"metadata_ablation_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
