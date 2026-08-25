#!/usr/bin/env python3
# evaluation/openllm_baseline.py — Open-Weight LLM Evaluation for ExhibitionBench
# ==============================================================================
# Reproducibility artifact for evaluating open-weight models on ExhibitionBench.
# Supports multiple backends: HuggingFace Serverless API, Together AI, Groq,
# local Ollama, or any vLLM-compatible endpoint.
#
# Usage examples:
#   # Groq (free tier, Llama-3.3-70B)
#   python evaluation/openllm_baseline.py \
#       --api-base https://api.groq.com/openai/v1 \
#       --api-key YOUR_GROQ_KEY \
#       --model llama-3.3-70b-versatile \
#       --tasks meip tes ecd
#
#   # Together AI (Llama-3.1-8B)
#   python evaluation/openllm_baseline.py \
#       --api-base https://api.together.xyz/v1 \
#       --api-key YOUR_TOGETHER_KEY \
#       --model meta-llama/Llama-3.1-8B-Instruct-Turbo \
#       --tasks meip tes ecd
#
#   # Local Ollama
#   python evaluation/openllm_baseline.py \
#       --api-base http://localhost:11434/v1 \
#       --api-key ollama \
#       --model llama3:70b \
#       --tasks meip tes ecd
#
#   # Local vLLM server
#   python evaluation/openllm_baseline.py \
#       --api-base http://localhost:8000/v1 \
#       --api-key vllm \
#       --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#       --tasks meip tes ecd
#
# Results are saved to results/{task}_{slug}_shot0.json in the same format
# as sota_eval.py, fully compatible with compile_sota_results.py.

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import openai

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE    = Path(__file__).resolve().parent.parent
DATA    = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


# ── Backend client ────────────────────────────────────────────────────────────

def make_client(api_base: str, api_key: str) -> openai.OpenAI:
    """Create an OpenAI-compatible client for any backend."""
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return openai.OpenAI(api_key=api_key, base_url=base)


def call_llm(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> tuple[Optional[str], float, dict]:
    """Call LLM with up to 3 retries. Returns (content, latency_sec, usage_dict)."""
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
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
            wait = 2 ** attempt
            log.warning(f"Rate limit; sleeping {wait}s")
            time.sleep(wait)
        except Exception as exc:
            log.warning(f"API error (attempt {attempt+1}): {exc}")
            time.sleep(1)
    return None, 0.0, empty_usage


# ── Data helpers ──────────────────────────────────────────────────────────────

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


def model_slug(model: str) -> str:
    """Convert model name to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", model).strip("_")


# ── MEIP helpers ──────────────────────────────────────────────────────────────

MEIP_PROMPT_TEMPLATE = """\
You are assisting a museum curator. Given an exhibition theme and some context objects already selected, \
identify which ONE candidate object best fits the exhibition and should be added next.

Exhibition theme: {theme}

Context objects already in exhibition:
{context}

Candidate objects (choose the best fit):
{candidates}

Reply with ONLY the ID of the best-fitting candidate object (e.g., "met_123456").
"""


def build_meip_prompt(sample: dict, objects: dict) -> tuple[str, list[str]]:
    theme = sample.get("exhibition_theme", "")
    context_raw = sample.get("context", [])
    ctx_lines = []
    for c in context_raw[:4]:
        obj = c if isinstance(c, dict) else objects.get(c, {})
        if obj:
            ctx_lines.append(
                f"  - {obj.get('title', '?')} "
                f"({obj.get('culture', '?')}, {obj.get('date', '?')})"
            )
    context_str = "\n".join(ctx_lines) if ctx_lines else "  (none)"

    candidates_raw = sample.get("candidates", sample.get("candidate_ids", []))
    cand_lines, cand_ids = [], []
    for idx, c in enumerate(candidates_raw, 1):
        cid = c["id"] if isinstance(c, dict) else c
        obj = c if isinstance(c, dict) else objects.get(cid, {})
        cand_ids.append(cid)
        if obj:
            cand_lines.append(
                f"  [{idx}] ID={cid} | {obj.get('title', '?')}"
                f" | {obj.get('culture', '?')}"
                f" | {obj.get('date', '?')}"
                f" | {(obj.get('medium') or '')[:60]}"
            )
        else:
            cand_lines.append(f"  [{idx}] ID={cid}")
    prompt = MEIP_PROMPT_TEMPLATE.format(
        theme=theme,
        context=context_str,
        candidates="\n".join(cand_lines),
    )
    return prompt, cand_ids


def parse_id_selection(response: str, candidates: list[str]) -> list[str]:
    resp = (response or "").strip()
    found = sorted((c for c in candidates if c in resp), key=resp.find)
    if found:
        return found + [c for c in candidates if c not in found]
    nums = re.findall(r"\b(\d+)\b", resp)
    idx_ranked: list[int] = []
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < len(candidates) and idx not in idx_ranked:
            idx_ranked.append(idx)
    if idx_ranked:
        result = [candidates[i] for i in idx_ranked]
        result += [c for c in candidates if c not in result]
        return result
    return candidates


def mrr_score(gold_id: str, ranked: list[str]) -> float:
    try:
        return 1.0 / (ranked.index(gold_id) + 1)
    except ValueError:
        return 0.0


# ── MEIP evaluation ───────────────────────────────────────────────────────────

def evaluate_meip(
    client: openai.OpenAI,
    model: str,
    samples: list[dict],
    objects: dict,
    workers: int = 100,
) -> dict:
    log.info(f"MEIP: {len(samples)} samples, {workers} workers")
    mrr_scores: list[float] = []
    total_latency = 0.0
    usage_agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    done = [0]
    lock = Lock()

    def _one(sample):
        gold_id = sample.get("gold_id", "")
        prompt, cand_ids = build_meip_prompt(sample, objects)
        if not gold_id or not cand_ids:
            return None
        response, lat, usage = call_llm(client, model, prompt)
        if response is None:
            return None
        ranked = parse_id_selection(response, cand_ids)
        return mrr_score(gold_id, ranked), lat, usage

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, s): s for s in samples}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                done[0] += 1
                if res is not None:
                    m, lat, usage = res
                    mrr_scores.append(m)
                    total_latency += lat
                    for k in usage_agg:
                        usage_agg[k] += usage.get(k, 0)
                if done[0] % 100 == 0:
                    cur = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
                    log.info(f"  MEIP {done[0]}/{len(samples)}, MRR={cur:.4f}")

    n = len(mrr_scores)
    final_mrr = sum(mrr_scores) / n if n else 0.0
    log.info(f"MEIP done: n={n}, MRR={final_mrr:.4f}")
    return {
        "task": "meip",
        "model": model,
        "shot": 0,
        "n_samples": n,
        "mrr": round(final_mrr, 4),
        "total_latency_sec": round(total_latency, 2),
        "avg_latency_sec": round(total_latency / max(n, 1), 3),
        **usage_agg,
    }


# ── TES evaluation ────────────────────────────────────────────────────────────

TES_PROMPT_TEMPLATE = """\
You are a museum curator. Given an exhibition theme, rank the following candidate objects \
from most to least relevant to the theme. Consider cultural fit, historical period, artistic \
style, and thematic coherence.

Exhibition theme: {theme}

Candidate objects (rank ALL of them):
{candidates}

Reply with ONLY the IDs in ranked order, one per line, most relevant first.
"""


def ndcg_at_k(gold_id: str, ranked: list[str], k: int = 10) -> float:
    for i, cid in enumerate(ranked[:k]):
        if cid == gold_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def evaluate_tes(
    client: openai.OpenAI,
    model: str,
    exhibitions: list[dict],
    objects: dict,
    workers: int = 100,
    k: int = 10,
) -> dict:
    log.info(f"TES: {len(exhibitions)} queries, {workers} workers")
    ndcg_scores: list[float] = []
    mrr_scores_tes: list[float] = []
    total_latency = 0.0
    usage_agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    done = [0]
    lock = Lock()

    def _one(exh):
        theme = exh.get("query_theme", exh.get("theme", exh.get("exhibition_theme", "")))
        gold_ids = exh.get("relevant_ids", exh.get("gold_ids", []))
        if not gold_ids:
            gid = exh.get("gold_id")
            if gid:
                gold_ids = [gid]
        candidates = exh.get("candidates", exh.get("candidate_ids", []))
        candidate_ids = [c["id"] if isinstance(c, dict) else c for c in candidates]
        if not theme or not gold_ids or not candidate_ids:
            return None
        cand_lines = []
        for idx, (cid, candidate) in enumerate(zip(candidate_ids, candidates), 1):
            if isinstance(candidate, dict) and candidate.get("sample_objects"):
                sample_text = "; ".join(
                    f"{item.get('title', '?')} ({item.get('culture', '?')}, {item.get('date', '?')})"
                    for item in candidate["sample_objects"][:5]
                )
                cand_lines.append(f"  [{idx}] ID={cid} | {sample_text}")
            else:
                obj = candidate if isinstance(candidate, dict) else objects.get(cid, {})
                cand_lines.append(
                    f"  [{idx}] ID={cid} | {obj.get('title', '?')}"
                    f" | {obj.get('culture', '?')} | {obj.get('date', '?')}"
                )
        prompt = TES_PROMPT_TEMPLATE.format(
            theme=theme, candidates="\n".join(cand_lines)
        )
        response, lat, usage = call_llm(client, model, prompt, max_tokens=500)
        if response is None:
            return None
        resp = response.strip()
        found = sorted((c for c in candidate_ids if c in resp), key=resp.find)
        if not found:
            nums = re.findall(r"\b(\d+)\b", resp)
            idx_order: list[int] = []
            for n in nums:
                idx = int(n) - 1
                if 0 <= idx < len(candidate_ids) and idx not in idx_order:
                    idx_order.append(idx)
            found = [candidate_ids[i] for i in idx_order]
            found += [c for c in candidate_ids if c not in found]
        primary = gold_ids[0]
        return ndcg_at_k(primary, found, k=k), mrr_score(primary, found), lat, usage

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, e): e for e in exhibitions}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                done[0] += 1
                if res is not None:
                    nd, m, lat, usage = res
                    ndcg_scores.append(nd)
                    mrr_scores_tes.append(m)
                    total_latency += lat
                    for kk in usage_agg:
                        usage_agg[kk] += usage.get(kk, 0)
                if done[0] % 50 == 0:
                    cur_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
                    log.info(f"  TES {done[0]}/{len(exhibitions)}, NDCG@10={cur_ndcg:.4f}")

    n = len(ndcg_scores)
    log.info(f"TES done: n={n}, NDCG@10={sum(ndcg_scores)/max(n,1):.4f}")
    return {
        "task": "tes",
        "model": model,
        "shot": 0,
        "n_samples": n,
        "ndcg@10": round(sum(ndcg_scores) / max(n, 1), 4),
        "mrr": round(sum(mrr_scores_tes) / max(n, 1), 4),
        "total_latency_sec": round(total_latency, 2),
        "avg_latency_sec": round(total_latency / max(n, 1), 3),
        **usage_agg,
    }


# ── ECD evaluation ────────────────────────────────────────────────────────────

ECD_PROMPT_TEMPLATE = """\
You are an expert museum curator. Two exhibition sequences are shown below.
One is coherent (real), the other contains a disruptive artifact that breaks coherence.
Identify which sequence is the COHERENT one.

Sequence A:
{seq_a}

Sequence B:
{seq_b}

Reply with ONLY "A" or "B".
"""


def format_obj(obj: dict) -> str:
    parts = [obj.get("title", "?")]
    for field in ("culture", "date", "medium", "department"):
        val = obj.get(field)
        if val:
            parts.append(str(val)[:60])
    return " | ".join(parts)


def evaluate_ecd(
    client: openai.OpenAI,
    model: str,
    samples: list[dict],
    objects: dict,
    workers: int = 100,
) -> dict:
    log.info(f"ECD: {len(samples)} samples, {workers} workers")
    level_correct: dict[str, list[int]] = defaultdict(list)
    total_latency = 0.0
    usage_agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    done = [0]
    lock = Lock()

    def _format_seq(seq) -> str:
        if not seq:
            return "(empty)"
        lines = []
        for i, item in enumerate(seq, 1):
            if item is None:
                continue
            obj = item if isinstance(item, dict) else objects.get(item, {})
            if obj:
                lines.append(f"  {i}. {format_obj(obj)}")
        return "\n".join(lines) if lines else "(empty)"

    def _one(sample):
        level_raw = str(sample.get("level", "1"))
        level = level_raw if level_raw.startswith("L") else f"L{level_raw}"
        pos_seq = sample.get("positive_sequence", sample.get("pos_seq"))
        neg_seq = sample.get("negative_sequence", sample.get("neg_seq"))
        if not pos_seq:
            pos_seq = sample.get("positive", {}).get("items")
        if not neg_seq:
            neg_seq = sample.get("negative", {}).get("items")
        if not pos_seq or not neg_seq:
            return None
        # Randomly assign A/B to avoid position bias
        if random.random() < 0.5:
            seq_a, seq_b, gold = pos_seq, neg_seq, "A"
        else:
            seq_a, seq_b, gold = neg_seq, pos_seq, "B"
        prompt = ECD_PROMPT_TEMPLATE.format(
            seq_a=_format_seq(seq_a),
            seq_b=_format_seq(seq_b),
        )
        response, lat, usage = call_llm(client, model, prompt, max_tokens=10)
        if response is None:
            return None
        resp = response.strip().upper()
        pred = "A" if "A" in resp else ("B" if "B" in resp else "?")
        return level, (1 if pred == gold else 0), lat, usage

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, s): s for s in samples}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                done[0] += 1
                if res is not None:
                    lev, corr, lat, usage = res
                    level_correct[lev].append(corr)
                    total_latency += lat
                    for kk in usage_agg:
                        usage_agg[kk] += usage.get(kk, 0)
                if done[0] % 100 == 0:
                    all_c = [c for v in level_correct.values() for c in v]
                    cur = sum(all_c) / len(all_c) if all_c else 0
                    log.info(f"  ECD {done[0]}/{len(samples)}, Macro={cur:.4f}")

    per_level = {}
    for lev in ["L1", "L2", "L3", "L4"]:
        vals = level_correct.get(lev, [])
        per_level[f"pairaccc_{lev}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

    all_vals = [c for v in level_correct.values() for c in v]
    macro = round(sum(all_vals) / len(all_vals), 4) if all_vals else 0.0
    n = len(all_vals)
    log.info(f"ECD done: n={n}, Macro={macro:.4f}, per-level={per_level}")
    return {
        "task": "ecd",
        "model": model,
        "shot": 0,
        "n_samples": n,
        **per_level,
        "macro_pairaccc": macro,
        "total_latency_sec": round(total_latency, 2),
        "avg_latency_sec": round(total_latency / max(n, 1), 3),
        **usage_agg,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Open-weight LLM evaluation for ExhibitionBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-base", required=True,
        help="OpenAI-compatible API base URL "
             "(e.g. https://api.groq.com/openai/v1 or http://localhost:11434/v1)"
    )
    parser.add_argument("--api-key", required=True,
                        help="API key for the backend (use 'ollama' or 'vllm' for local)")
    parser.add_argument("--model", required=True,
                        help="Model name as recognised by the backend "
                             "(e.g. llama-3.3-70b-versatile, meta-llama/Llama-3.1-8B-Instruct-Turbo)")
    parser.add_argument("--tasks", nargs="+", choices=["meip", "tes", "ecd"],
                        default=["meip", "tes", "ecd"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per task (default: all)")
    parser.add_argument("--workers", type=int, default=100,
                        help="ThreadPoolExecutor max_workers (default: 100)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing result files")
    args = parser.parse_args()

    slug = model_slug(args.model)
    log.info(f"Model: {args.model}  (slug: {slug})")
    log.info(f"API base: {args.api_base}")
    log.info(f"Tasks: {args.tasks}")
    log.info(f"Workers: {args.workers}")

    client = make_client(args.api_base, args.api_key)

    # Load shared data
    obj_path = find_data_file("objects")
    objects  = load_objects(obj_path) if obj_path else {}
    log.info(f"Loaded {len(objects)} objects from {obj_path}")

    for task in args.tasks:
        out_path = RESULTS / f"{task}_{slug}_shot0.json"
        if out_path.exists() and not args.force:
            log.info(f"Already exists: {out_path} (use --force to overwrite)")
            continue

        if task == "meip":
            meip_path = find_data_file("meip_samples")
            if not meip_path:
                log.error("Cannot find meip_samples data file")
                continue
            samples = load_jsonl(meip_path)
            if args.max_samples:
                samples = samples[: args.max_samples]
            result = evaluate_meip(client, args.model, samples, objects,
                                   workers=args.workers)

        elif task == "tes":
            tes_path = find_data_file("tes_samples")
            if not tes_path:
                log.error("Cannot find tes_samples data file")
                continue
            exhibitions = load_jsonl(tes_path)
            if args.max_samples:
                exhibitions = exhibitions[: args.max_samples]
            result = evaluate_tes(client, args.model, exhibitions, objects,
                                  workers=args.workers)

        elif task == "ecd":
            ecd_path = find_data_file("ecd_samples")
            if not ecd_path:
                log.error("Cannot find ecd_samples data file")
                continue
            samples = load_jsonl(ecd_path)
            if args.max_samples:
                samples = samples[: args.max_samples]
            result = evaluate_ecd(client, args.model, samples, objects,
                                  workers=args.workers)

        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"Saved: {out_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
