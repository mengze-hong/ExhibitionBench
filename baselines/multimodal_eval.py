"""
baselines/multimodal_eval.py — ExhibitionBench Multimodal (Vision) Evaluator
=============================================================================
Evaluates vision-language LLMs on the MEIP task using both text AND image inputs.
Images are embedded as image_url content parts in the OpenAI-compatible message format.

Supported models (all confirmed vision-capable in LiteLLM API):
  - gpt-5.2                          (OpenAI)
  - claude-opus-4.6                  (Anthropic)
  - gemini-2.5-pro                   (Google)
  - gemini-2.5-flash                 (Google)
  - doubao-seed-1.6-vision-250815    (ByteDance)

Image coverage (MEIP v3):
  - Candidate objects: ~91% have image_url
  - Context objects:   ~94% have image_url
  Missing images fall back to text-only for that item.

Usage:
  python baselines/multimodal_eval.py --models gpt-5.2 gemini-2.5-flash
  python baselines/multimodal_eval.py --models all
  python baselines/multimodal_eval.py --models gpt-5.2 --max-samples 50 --workers 50
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import openai

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# ── API Config ────────────────────────────────────────────────────────────────

INTERNAL_API_BASE = "http://csig.litellm.prod.sgpolaris"
INTERNAL_API_KEY = "sk-TpK0g832p8LbMXTdI_pjkQ"

CLIENT = openai.OpenAI(
    api_key=INTERNAL_API_KEY,
    base_url=f"{INTERNAL_API_BASE}/v1",
)

# ── Vision Model Registry ─────────────────────────────────────────────────────

VISION_MODELS = {
    "gpt-5.2":                        "gpt-5.2",
    "claude-opus-4.6":                "claude-opus-4.6",
    "gemini-2.5-pro":                 "gemini-2.5-pro",
    "gemini-2.5-flash":               "gemini-2.5-flash",
    "doubao-seed-1.6-vision-250815":  "doubao-seed-1.6-vision-250815",
    "claude-sonnet-4.5":              "claude-sonnet-4.5",
    "deepseek-v3.2":                  "deepseek-v3.2",
    "kimi-k2.5":                      "kimi-k2.5",
}

ALL_VISION_MODELS = list(VISION_MODELS.keys())

# Models that need larger token budgets (internal thinking)
LARGE_TOKEN_MODELS = {"gemini-2.5-pro", "gemini-2.5-flash", "kimi-k2.5"}
TEMP1_MODELS: set[str] = {"kimi-k2.5"}  # kimi requires temperature=1

# Domains known to be reliably accessible by the LiteLLM proxy.
# artic.edu (Art Institute of Chicago IIIF) times out from the proxy — skip those.
TRUSTED_IMG_DOMAINS = {
    "images.metmuseum.org",
    "api.europeana.eu",
    "openaccess-cdn.clevelandart.org",
    "framemark.vam.ac.uk",
    "media.vam.ac.uk",
    "lh3.googleusercontent.com",
    "upload.wikimedia.org",
    "collectionapi.metmuseum.org",
    "collections.vam.ac.uk",
}

# Gemini via Vertex AI is more restrictive — MetMuseum CDN sometimes returns
# URL_ERROR-ERROR_NOT_FOUND on Vertex AI's fetch. Use only CDN-agnostic sources.
GEMINI_TRUSTED_IMG_DOMAINS = {
    "api.europeana.eu",
    "openaccess-cdn.clevelandart.org",
    "lh3.googleusercontent.com",
    "upload.wikimedia.org",
}

# Models routed through Vertex AI (Gemini) need stricter filtering
VERTEX_MODELS = {"gemini-2.5-pro", "gemini-2.5-flash"}


def is_trusted_image(url: str, model: str = "") -> bool:
    """Return True only for image URLs we know the LiteLLM proxy can fetch."""
    if not url:
        return False
    try:
        domain = url.split("/")[2]
    except IndexError:
        return False
    if model in VERTEX_MODELS:
        return domain in GEMINI_TRUSTED_IMG_DOMAINS
    return domain in TRUSTED_IMG_DOMAINS

# ── LLM Vision Call ───────────────────────────────────────────────────────────

def call_llm_vision(
    model: str,
    content_parts: list[dict],
    max_tokens: int = 200,
    retries: int = 3,
    timeout: int = 180,
) -> tuple[Optional[str], float, dict]:
    """
    Calls a vision LLM with a multipart content message.

    content_parts is a list of dicts following OpenAI content part format:
      {"type": "text", "text": "..."}
      {"type": "image_url", "image_url": {"url": "https://..."}}

    Returns: (content, latency_sec, usage_dict)
    """
    actual_max = max_tokens * 8 if model in LARGE_TOKEN_MODELS else max_tokens
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    temp = 1.0 if model in TEMP1_MODELS else 0.0

    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = CLIENT.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_parts}],
                max_tokens=actual_max,
                temperature=temp,
                timeout=timeout,
            )
            latency = time.perf_counter() - t0

            usage = empty_usage.copy()
            if resp.usage:
                usage["prompt_tokens"]     = resp.usage.prompt_tokens or 0
                usage["completion_tokens"] = resp.usage.completion_tokens or 0
                usage["total_tokens"]      = resp.usage.total_tokens or 0

            if not (resp.choices and resp.choices[0].message):
                continue
            content = resp.choices[0].message.content
            return (content or ""), latency, usage

        except openai.RateLimitError:
            wait = 2 ** attempt
            log.warning(f"Rate limit for {model}, waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            log.warning(f"Vision API error ({model}, attempt {attempt+1}): {e}")
            time.sleep(1)
    return None, 0.0, empty_usage


# ── Data Loaders ──────────────────────────────────────────────────────────────

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
    for suffix in ["_v3", "_v2", ""]:
        p = DATA / f"{name}{suffix}.jsonl"
        if p.exists():
            return p
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def mrr(gold_id: str, ranked_ids: list[str]) -> float:
    for i, rid in enumerate(ranked_ids, 1):
        if rid == gold_id:
            return 1.0 / i
    return 0.0


def parse_selection(response: str, candidates: list[str]) -> list[str]:
    """Parse model output into a ranked list of candidate IDs."""
    import re
    resp = response.strip() if response else ""

    found = []
    for cid in candidates:
        if cid in resp:
            found.append(cid)
    if found:
        return found + [c for c in candidates if c not in found]

    lines = resp.split("\n")
    ranked = []
    for line in lines:
        line = line.strip()
        for cid in candidates:
            if cid in line and cid not in ranked:
                ranked.append(cid)
                break
    if ranked:
        return ranked + [c for c in candidates if c not in ranked]

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


# ── Multimodal Prompt Builder ─────────────────────────────────────────────────

def build_meip_vision_prompt(
    sample: dict,
    objects: dict[str, dict],
    model: str = "",
) -> tuple[list[dict], list[str]]:
    """
    Build a multimodal MEIP prompt (text + images).

    Returns:
        (content_parts, candidate_ids)
        content_parts: list of {"type":"text"/"image_url", ...} dicts
        candidate_ids: ordered list of candidate IDs (for parse_selection)
    """
    theme = sample.get("exhibition_theme", "")
    parts: list[dict] = []

    # ── Text header ──────────────────────────────────────────────────────────
    parts.append({
        "type": "text",
        "text": (
            "You are assisting a museum curator. Given an exhibition theme and "
            "context objects already selected, identify which ONE candidate object "
            "best fits the exhibition and should be added next.\n\n"
            f"Exhibition theme: {theme}\n\n"
            "Context objects already in exhibition:"
        )
    })

    # ── Context objects: text + optional image ────────────────────────────────
    context_raw = sample.get("context", [])
    for i, c in enumerate(context_raw[:4]):
        obj = c if isinstance(c, dict) else objects.get(c, {})
        if not obj:
            continue
        title   = obj.get("title", "?")
        culture = obj.get("culture", "?")
        date    = obj.get("date", "?")
        medium  = (obj.get("medium") or "")[:60]
        img_url = obj.get("image_url", "")

        parts.append({
            "type": "text",
            "text": f"\n  [{i+1}] {title} | {culture} | {date} | {medium}"
        })
        if is_trusted_image(img_url, model):
            parts.append({
                "type": "image_url",
                "image_url": {"url": img_url, "detail": "low"}
            })

    # ── Separator ─────────────────────────────────────────────────────────────
    parts.append({
        "type": "text",
        "text": "\n\nCandidate objects (choose the best fit):"
    })

    # ── Candidates: text + optional image ────────────────────────────────────
    candidates_raw = sample.get("candidates", sample.get("candidate_ids", []))
    candidate_ids: list[str] = []
    for idx, c in enumerate(candidates_raw, 1):
        obj = c if isinstance(c, dict) else objects.get(c, {})
        if isinstance(c, dict):
            cid = c["id"]
        else:
            cid = c
        candidate_ids.append(cid)

        title   = obj.get("title", "?") if obj else "?"
        culture = obj.get("culture", "?") if obj else "?"
        date    = obj.get("date", "?") if obj else "?"
        medium  = (obj.get("medium") or "")[:60] if obj else ""
        img_url = obj.get("image_url", "") if obj else ""

        parts.append({
            "type": "text",
            "text": f"\n  [{idx}] ID={cid} | {title} | {culture} | {date} | {medium}"
        })
        if is_trusted_image(img_url, model):
            parts.append({
                "type": "image_url",
                "image_url": {"url": img_url, "detail": "low"}
            })

    # ── Final instruction ─────────────────────────────────────────────────────
    parts.append({
        "type": "text",
        "text": (
            "\n\nReply with ONLY the ID of the best-fitting candidate object "
            "(e.g., \"met_123456\")."
        )
    })

    return parts, candidate_ids


# ── Evaluation Function ───────────────────────────────────────────────────────

def evaluate_meip_vision(
    model: str,
    samples: list[dict],
    objects: dict[str, dict],
    max_samples: int = 1409,
    workers: int = 150,
    checkpoint_path: Optional[Path] = None,
) -> dict:
    """
    Evaluate a vision-language model on the MEIP task with image inputs.
    Uses ThreadPoolExecutor with `workers` parallel threads.
    Supports checkpoint resume: pass checkpoint_path to save/resume progress.
    """
    samples_used = samples[:max_samples]

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_results: dict[int, tuple] = {}  # idx -> (score, hit1, latency, usage)
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    ckpt_results[r["idx"]] = (r["score"], r["hit1"], r["latency"], r["usage"])
                except Exception:
                    pass
        log.info(f"[{model}] Resuming from checkpoint: {len(ckpt_results)} samples already done")

    ckpt_lock = Lock()
    ckpt_f = open(checkpoint_path, "a", encoding="utf-8") if checkpoint_path else None

    def _run_one(item):
        i, sample = item
        # Skip already done
        if i in ckpt_results:
            return (i, *ckpt_results[i])
        gold_id = sample.get("gold_id", "")
        if not gold_id:
            return None
        try:
            content_parts, candidate_ids = build_meip_vision_prompt(sample, objects, model=model)
        except Exception as e:
            log.warning(f"Prompt build error at sample {i}: {e}")
            return None
        if not candidate_ids:
            return None

        response, latency, usage = call_llm_vision(model, content_parts, max_tokens=200)
        if response is None:
            return None
        ranked = parse_selection(response, candidate_ids)
        score = mrr(gold_id, ranked)
        hit1 = 1.0 if ranked and ranked[0] == gold_id else 0.0

        # Save to checkpoint
        if ckpt_f:
            with ckpt_lock:
                ckpt_f.write(json.dumps({"idx": i, "score": score, "hit1": hit1,
                                         "latency": latency, "usage": usage}) + "\n")
                ckpt_f.flush()

        return (i, score, hit1, latency, usage)

    mrr_scores = []
    hit1_scores = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    done = 0
    lock = Lock()

    log.info(f"[{model}] Starting MEIP vision eval on {len(samples_used)} samples "
             f"with {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, (i, s)): i for i, s in enumerate(samples_used)}
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                continue
            _, score, hit1, latency, usage = res
            with lock:
                mrr_scores.append(score)
                hit1_scores.append(hit1)
                total_latency += latency
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                total_tokens += usage["total_tokens"]
                done += 1
                if done % 100 == 0:
                    log.info(f"  [{model}] {done}/{len(samples_used)} done | "
                             f"MRR={sum(mrr_scores)/len(mrr_scores):.4f} "
                             f"Hit@1={sum(hit1_scores)/len(hit1_scores):.4f}")

    if ckpt_f:
        ckpt_f.close()

    n = len(mrr_scores)
    result = {
        "task": "meip_vision",
        "model": model,
        "modality": "text+image",
        "n_samples": n,
        "mrr": round(sum(mrr_scores) / n, 4) if n else 0,
        "hit@1": round(sum(hit1_scores) / n, 4) if n else 0,
        "total_latency_sec": round(total_latency, 2),
        "avg_latency_sec": round(total_latency / n, 3) if n else 0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(BASE / "logs" / "multimodal_eval.log", encoding="utf-8"),
        ],
    )
    (BASE / "logs").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="ExhibitionBench Multimodal (Vision) Evaluator")
    parser.add_argument(
        "--models", nargs="+", default=["gpt-5.2", "gemini-2.5-flash"],
        help=f"Models to evaluate. Use 'all' for all vision models. "
             f"Available: {list(VISION_MODELS.keys())}",
    )
    parser.add_argument("--max-samples", type=int, default=1409,
                        help="Max MEIP samples to evaluate (default: all 1409)")
    parser.add_argument("--workers", type=int, default=150,
                        help="ThreadPool workers for parallel API calls (default: 150)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing result files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (auto-detected per model)")
    args = parser.parse_args()

    # Resolve model list
    model_list = ALL_VISION_MODELS if "all" in args.models else args.models
    for m in model_list:
        if m not in VISION_MODELS:
            log.warning(f"Unknown model '{m}', skipping. Available: {list(VISION_MODELS.keys())}")
    model_list = [m for m in model_list if m in VISION_MODELS]

    # Load data
    meip_path = find_data_file("meip_samples")
    if not meip_path:
        log.error("No MEIP data file found in data/. Expected meip_samples_v3.jsonl")
        sys.exit(1)
    obj_path = find_data_file("objects")
    if not obj_path:
        log.error("No objects data file found in data/. Expected objects_v3.jsonl")
        sys.exit(1)

    log.info(f"Loading MEIP samples from {meip_path.name}...")
    samples = load_jsonl(meip_path)
    log.info(f"Loading objects from {obj_path.name}...")
    objects = load_objects(obj_path)
    log.info(f"Loaded {len(samples)} samples, {len(objects)} objects")

    # Quick image URL coverage stats (trusted domains only)
    has_img = sum(1 for s in samples[:200] for c in s.get("candidates", [])
                  if isinstance(c, dict) and is_trusted_image(c.get("image_url", "")))
    total_cands = sum(len(s.get("candidates", [])) for s in samples[:200])
    log.info(f"Trusted image URL coverage (first 200 samples, candidates): "
             f"{has_img}/{total_cands} = {has_img/total_cands*100:.1f}%")

    # Run evaluation per model
    for model in model_list:
        out_path = RESULTS / f"meip_{model}_vision_shot0.json"
        if out_path.exists() and not args.overwrite:
            log.info(f"[{model}] Result already exists at {out_path.name}, skipping. "
                     f"Use --overwrite to re-run.")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Evaluating: {model}  (max_samples={args.max_samples}, workers={args.workers})")
        log.info(f"{'='*60}")

        ckpt_path = RESULTS / f"meip_{model}_vision_ckpt.jsonl" if args.resume else None
        result = evaluate_meip_vision(
            model=model,
            samples=samples,
            objects=objects,
            max_samples=args.max_samples,
            workers=args.workers,
            checkpoint_path=ckpt_path,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        log.info(f"[{model}] DONE → {out_path.name}")
        log.info(f"  MRR  = {result['mrr']:.4f}")
        log.info(f"  Hit@1= {result['hit@1']:.4f}")
        log.info(f"  n    = {result['n_samples']}")

    log.info("\nAll models complete.")


if __name__ == "__main__":
    main()
