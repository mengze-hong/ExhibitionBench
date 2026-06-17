"""
baselines/sota_eval.py — ExhibitionBench SOTA Multi-Model Evaluator
====================================================================
Evaluates all SOTA LLMs from diverse families on all three benchmark tasks:
  - MEIP  (Museum Exhibition Item Prediction) — 10-way ranking, MRR metric
  - TES   (Thematic Exhibition Selection)      — 50-way ranking, NDCG@10 metric
  - ECD   (Exhibition Coherence Discrimination) — pairwise, PairAcc metric

Models evaluated (one strong representative per family):
  OpenAI    : gpt-5.2
  Anthropic : claude-opus-4.6
  Google    : gemini-2.5-flash
  DeepSeek  : deepseek-r1
  Kimi      : kimi-k2.5
  Doubao    : doubao-seed-1.6
  GLM       : glm-5
  Minimax   : minimax-m2.5

Usage:
  python baselines/sota_eval.py --task meip --model gpt-5.2
  python baselines/sota_eval.py --task all --model all
  python baselines/sota_eval.py --task meip --model all --max-samples 200
  python baselines/sota_eval.py --task all --model all --max-samples 100 --shot 0
"""

from __future__ import annotations

import argparse
import hashlib
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
from typing import Any, Optional

import openai

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RAW_RESPONSES = RESULTS / "raw_responses"

# ── API Config ────────────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Please export it before running evaluations."
        )
    return value


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


INTERNAL_API_BASE = os.environ.get(
    "LLM_API_BASE",
    "http://YOUR_LLM_API_BASE",
).rstrip("/")
INTERNAL_API_KEY = _require_env("LLM_API_KEY")

CLIENT = openai.OpenAI(
    api_key=INTERNAL_API_KEY,
    base_url=f"{INTERNAL_API_BASE}/v1",
)

# Open-weight models via an OpenAI-compatible gateway (Llama / Qwen / Mistral)
XTY_API_BASE = os.environ.get("LLM_OPENWEIGHT_API_BASE", "https://YOUR_OPENWEIGHT_API_BASE").rstrip("/")
XTY_API_KEY  = _require_env("LLM_OPENWEIGHT_API_KEY")

XTY_CLIENT = openai.OpenAI(
    api_key=XTY_API_KEY,
    base_url=XTY_API_BASE,
)

# ── Model Registry ────────────────────────────────────────────────────────────

MODELS = {
    # Family         : model_id
    "gpt-5.2":            "gpt-5.2",            # OpenAI flagship
    "gpt-5.1":            "gpt-5.1",            # OpenAI strong v1
    "gpt-5":              "gpt-5",              # OpenAI strong
    "claude-opus-4.6":    "claude-opus-4.6",    # Anthropic flagship
    "claude-opus-4.5":    "claude-opus-4.5",    # Anthropic prev flagship
    "claude-sonnet-4.5":  "claude-sonnet-4.5",  # Anthropic fast
    "gemini-2.5-pro":     "gemini-2.5-pro",     # Google flagship
    "gemini-2.5-flash":   "gemini-2.5-flash",   # Google fast
    "gemini-3-pro-preview":   "gemini-3-pro-preview",     # Google Gemini-3 flagship preview
    "gemini-3-flash-preview": "gemini-3-flash-preview",   # Google Gemini-3 fast preview
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",   # Google Gemini-3.1 flagship preview
    "deepseek-r1":        "deepseek-r1",         # DeepSeek reasoning
    "deepseek-v3.2":      "deepseek-v3.2",       # DeepSeek chat v3.2
    # "deepseek-v3.1":      "deepseek-v3.1",       # DeepSeek chat v3.1 -- NOT AVAILABLE on this endpoint
    "deepseek-v3":        "deepseek-v3",         # DeepSeek chat v3.0
    "kimi-k2.5":          "kimi-k2.5",           # Moonshot
    "doubao-seed-2.0-pro":    "doubao-seed-2.0-pro",    # ByteDance flagship
    "doubao-seed-2.0-lite":   "doubao-seed-2.0-lite",   # ByteDance lite
    "doubao-seed-1.6":        "doubao-seed-1.6",        # ByteDance v1.6 standard
    # "doubao-seed-1.6-thinking":"doubao-seed-1.6-thinking", # REMOVED: endpoint currently closed
    # "doubao-seed-1.6-lite":   "doubao-seed-1.6-lite",   # REMOVED: endpoint currently closed
    # "doubao-seed-1.6-flash":  "doubao-seed-1.6-flash-250715", # REMOVED: endpoint currently closed
    "glm-5":              "glm-5",               # Zhipu AI (reasoning model)
    "minimax-m2.5":       "minimax-m2.5",        # Minimax
    # "gpt-5-mini":       "gpt-5-mini",          # REMOVED: content often empty (pure reasoning), 已有旧结果 n=0
    # "qwen-plus-latest": "qwen-plus-latest",    # REMOVED: 401 team无权限访问
    # ── Open-weight models via an OpenAI-compatible gateway ──────────────────────────────────────
    "llama-3.3-70b":          "llama-3.3-70b",          # Meta Llama-3.3 70B
    "llama-3.1-70b-instruct": "llama-3.1-70b-instruct", # Meta Llama-3.1 70B
    "llama-3.1-8b-instruct":  "llama-3.1-8b-instruct",  # Meta Llama-3.1 8B
    "qwen2.5-72b-instruct":   "qwen2.5-72b-instruct",   # Qwen2.5 72B
    "qwen2.5-7b-instruct":    "qwen2.5-7b-instruct",    # Qwen2.5 7B
    "qwen3-8b":               "qwen3-8b",               # Qwen3 8B
    "qwen3-14b":              "qwen3-14b",              # Qwen3 14B
}

ALL_MODELS = list(MODELS.keys())

# Default model selection (strongest one per family for paper main table)
DEFAULT_MODELS = [
    "gpt-5.2",            # OpenAI
    "claude-sonnet-4.5",  # Anthropic (claude-opus-4.6 times out)
    "gemini-2.5-pro",     # Google
    "deepseek-r1",        # DeepSeek
    "kimi-k2.5",          # Moonshot
    "doubao-seed-2.0-pro",# ByteDance
    "glm-5",              # Zhipu
    "minimax-m2.5",       # Minimax
]

# Models routed via an OpenAI-compatible gateway (open-weight)
XTY_MODELS = {
    "llama-3.3-70b", "llama-3.1-70b-instruct", "llama-3.1-8b-instruct",
    "qwen2.5-72b-instruct", "qwen2.5-7b-instruct",
    "qwen3-8b", "qwen3-14b",
}

# Reasoning models that put answer in reasoning_content, not content
REASONING_MODELS = {"deepseek-r1", "kimi-k2.5", "minimax-m2.5", "glm-5",
                    "doubao-seed-1.6-thinking"}
# Models that need larger max_tokens because they use thinking tokens
LARGE_TOKEN_MODELS = {"deepseek-r1", "kimi-k2.5", "minimax-m2.5",
                      "gemini-2.5-pro", "gemini-2.5-flash",
                      "gemini-3-pro-preview", "gemini-3-flash-preview",
                      "gemini-3.1-pro-preview",
                      "gpt-5", "gpt-5.1", "gpt-5-codex", "glm-5",
                      "doubao-seed-1.6-thinking"}
# Models that require temperature=1 (OpenAI reasoning/codex models)
TEMP1_MODELS = {"gpt-5", "gpt-5.1", "gpt-5-codex"}

# ── LLM Call ─────────────────────────────────────────────────────────────────

def call_llm(model: str, prompt: str, max_tokens: int = 2048,
             retries: int = 12, timeout: int = 300) -> tuple[Optional[str], float, dict]:
    """
    Returns: (content, latency_sec, usage_dict)
      - content: model response string, or None on failure
      - latency_sec: wall-clock seconds for the API call (0 if failed)
      - usage_dict: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    """
    # Reasoning models need much more token budget for thinking
    actual_max = max_tokens * 8 if model in LARGE_TOKEN_MODELS else max_tokens
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # Route open-weight models to an OpenAI-compatible gateway, others to primary LLM endpoint
    client = XTY_CLIENT if model in XTY_MODELS else CLIENT
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            # gpt-5 and similar reasoning models require temperature=1
            temp = 1.0 if model in TEMP1_MODELS else 0.0
            # Qwen3 defaults to thinking mode on an OpenAI-compatible gateway → triggers Cloudflare 524 timeout.
            # Inject /no_think system message to disable thinking and keep responses short.
            QWEN3_MODELS = {"qwen3-8b", "qwen3-14b", "qwen3-32b", "qwen3-72b"}
            messages: list[dict] = []
            if model in QWEN3_MODELS:
                messages.append({"role": "system", "content": "/no_think"})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=actual_max,
                temperature=temp,
                timeout=timeout,
            )
            latency = time.perf_counter() - t0

            # Extract token usage
            usage = empty_usage.copy()
            if resp.usage:
                usage["prompt_tokens"]     = resp.usage.prompt_tokens or 0
                usage["completion_tokens"] = resp.usage.completion_tokens or 0
                usage["total_tokens"]      = resp.usage.total_tokens or 0

            if not (resp.choices and resp.choices[0].message):
                continue
            msg = resp.choices[0].message
            content = msg.content
            # Reasoning models: answer may be empty while thinking is in reasoning_content
            if not content and model in REASONING_MODELS:
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    # Extract last line or answer from reasoning
                    lines = [l.strip() for l in rc.strip().split("\n") if l.strip()]
                    content = lines[-1] if lines else rc
            return (content or ""), latency, usage
        except openai.RateLimitError:
            wait = min(2 ** attempt, 60)  # cap at 60s
            log.warning(f"Rate limit for {model}, waiting {wait}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
        except Exception as e:
            log.warning(f"API error ({model}, attempt {attempt+1}): {e}")
            time.sleep(1)
    return None, 0.0, empty_usage


# ── Data Loaders ─────────────────────────────────────────────────────────────

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
    """Find data file, preferring v3_fixed > v3 > v2 > base."""
    for suffix in ["_v3_fixed", "_v3", "_v2", ""]:
        p = DATA / f"{name}{suffix}.jsonl"
        if p.exists():
            return p
    return None


class JsonlRawLogger:
    """Thread-safe JSONL logger for per-sample raw traces."""

    def __init__(self, path: Path, run_meta: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")
        self._lock = Lock()
        self.write(
            {
                "event": "run_start",
                "timestamp": time.time(),
                "run_meta": run_meta,
            }
        )

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.write(
                json.dumps(
                    {"event": "run_end", "timestamp": time.time()},
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._fh.close()


def build_tes_fewshot_prefix(shot: int) -> str:
    if shot <= 0:
        return ""
    examples = [
        (
            "Japanese Woodblock Prints",
            "Edo-period ukiyo-e prints with Japanese subjects.",
            "EX_001: French Impressionist landscapes; EX_002: Japanese ukiyo-e beauty print; EX_003: Ming porcelain",
            "ANSWER: EX_002",
        ),
        (
            "Ancient Egyptian Funerary Arts",
            "Objects associated with burial and afterlife in ancient Egypt.",
            "EX_001: Greek marble statue; EX_002: Egyptian shabtis and funerary object set; EX_003: Roman floor mosaic",
            "ANSWER: EX_002",
        ),
        (
            "Islamic Geometric Decoration",
            "Works emphasizing geometric Islamic ornament.",
            "EX_001: Viking silver brooch; EX_002: Mamluk geometric inlay Quran stand; EX_003: Tang horse figure",
            "ANSWER: EX_002",
        ),
    ]
    use_n = min(shot, len(examples))
    blocks = []
    for i in range(use_n):
        q_theme, q_desc, cands, ans = examples[i]
        blocks.append(
            "\n".join(
                [
                    f"EXAMPLE {i+1}:",
                    f"Query theme: {q_theme}",
                    f"Query description: {q_desc}",
                    f"Candidates: {cands}",
                    f"{ans}",
                ]
            )
        )
    return "Here are reference examples:\n\n" + "\n\n".join(blocks) + "\n\nNow solve the real query:\n\n"


def build_ecd_fewshot_prefix(shot: int) -> str:
    if shot <= 0:
        return ""
    examples = [
        (
            "Impressionist Landscapes",
            "A: Monet/Haystacks/Pissarro style landscape continuity",
            "B: Adds unrelated baroque religious painting into impressionist sequence",
            "ANSWER: A",
        ),
        (
            "Ancient Egyptian Funerary Arts",
            "A: coffin, shabti, canopic jar in coherent funerary context",
            "B: inserts modern abstract sculpture among ancient funerary objects",
            "ANSWER: A",
        ),
        (
            "Japanese Woodblock Prints",
            "A: Edo ukiyo-e prints with shared style/period",
            "B: inserts Renaissance oil portrait into ukiyo-e sequence",
            "ANSWER: A",
        ),
    ]
    use_n = min(shot, len(examples))
    blocks = []
    for i in range(use_n):
        theme, seq_a, seq_b, ans = examples[i]
        blocks.append(
            "\n".join(
                [
                    f"EXAMPLE {i+1}:",
                    f"Theme: {theme}",
                    f"Sequence A: {seq_a}",
                    f"Sequence B: {seq_b}",
                    ans,
                ]
            )
        )
    return "Here are reference examples:\n\n" + "\n\n".join(blocks) + "\n\nNow solve the real sample:\n\n"


# ── Metrics ──────────────────────────────────────────────────────────────────

def mrr(gold_id: str, ranked_ids: list[str]) -> float:
    for i, rid in enumerate(ranked_ids, 1):
        if rid == gold_id:
            return 1.0 / i
    return 0.0


def ndcg_at_k(gold_ids: set, ranked_ids: list[str], k: int = 10) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(ranked_ids[:k])
        if rid in gold_ids
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def parse_selection(response: str, candidates: list[str]) -> list[str]:
    """Parse model output into a ranked list of candidate IDs."""
    resp = response.strip() if response else ""

    # Try to find IDs directly mentioned
    found = []
    for cid in candidates:
        if cid in resp:
            found.append(cid)
    if found:
        return found + [c for c in candidates if c not in found]

    # Try to parse a numbered list (e.g., "1. ..., 2. ...")
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

    # Try to find first number mentioned
    import re
    nums = re.findall(r'\b(\d+)\b', resp)
    idx_ranked = []
    for n in nums:
        idx = int(n) - 1  # 1-indexed to 0-indexed
        if 0 <= idx < len(candidates) and idx not in idx_ranked:
            idx_ranked.append(idx)
    if idx_ranked:
        result = [candidates[i] for i in idx_ranked]
        result += [c for c in candidates if c not in result]
        return result

    return candidates  # fallback: original order


def parse_meip_best_id(response: str, candidate_ids: list[str]) -> Optional[str]:
    text = (response or "").strip()
    if not text:
        return None
    for cid in candidate_ids:
        if cid in text:
            return cid
    import re
    nums = re.findall(r"\b(\d+)\b", text)
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < len(candidate_ids):
            return candidate_ids[idx]
    return None


def parse_tes_ranked_ids(response: str, anon_candidate_ids: list[str]) -> Optional[list[str]]:
    text = (response or "").strip()
    if not text:
        return None
    ranked = []
    for aid in anon_candidate_ids:
        if aid in text and aid not in ranked:
            ranked.append(aid)
    if ranked:
        ranked += [x for x in anon_candidate_ids if x not in ranked]
        return ranked
    import re
    nums = re.findall(r"\b(\d+)\b", text)
    idx_ranked = []
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < len(anon_candidate_ids) and idx not in idx_ranked:
            idx_ranked.append(idx)
    if idx_ranked:
        ranked = [anon_candidate_ids[i] for i in idx_ranked]
        ranked += [x for x in anon_candidate_ids if x not in ranked]
        return ranked
    return None


# ── MEIP Evaluation ───────────────────────────────────────────────────────────

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

# CoT variant: prompts the model to reason before answering
MEIP_COT_TEMPLATE = """\
You are assisting a museum curator. Given an exhibition theme and some context objects already selected, \
identify which ONE candidate object best fits the exhibition and should be added next.

Exhibition theme: {theme}

Context objects already in exhibition:
{context}

Candidate objects (choose the best fit):
{candidates}

Think step by step:
1. What is the core cultural/period/medium focus of this exhibition theme?
2. How do the context objects reinforce that focus?
3. Which candidate best continues that narrative?

After your reasoning, output your final answer on the last line as:
ANSWER: <object_id>
"""

# Static few-shot bank: 5 hand-crafted examples covering diverse cultures/periods
# Each example is a self-contained (theme, context, candidates, answer) block
MEIP_FEWSHOT_BANK = [
    """\
EXAMPLE:
Exhibition theme: Japanese Woodblock Prints
Context objects:
  - Thirty-Six Views of Mt. Fuji (Japanese, 1830-1833)
  - The Great Wave off Kanagawa (Japanese, ca. 1831)
Candidates:
  [1] ID=ex_001 | Portrait of a Lady | European | 17th century | Oil on canvas
  [2] ID=ex_002 | Beauty Arranging Her Hair | Japanese | ca. 1795 | Woodblock print
  [3] ID=ex_003 | Dragon Vase | Chinese | Ming dynasty | Porcelain
Best fit: ex_002
Reason: ukiyo-e woodblock print of a Japanese beauty fits the Japanese Woodblock Prints theme.

""",
    """\
EXAMPLE:
Exhibition theme: Ancient Egyptian Funerary Arts
Context objects:
  - Canopic Jar of Neskhons (Egyptian, ca. 1069-945 BCE)
  - Book of the Dead of Hunefer (Egyptian, ca. 1275 BCE)
Candidates:
  [1] ID=ex_004 | Marble Statue of Athena | Greek | 5th century BCE | Marble
  [2] ID=ex_005 | Set of Four Shabtis | Egyptian | ca. 1550-1070 BCE | Faience
  [3] ID=ex_006 | Roman Mosaic Floor Fragment | Roman | 2nd century CE | Stone mosaic
Best fit: ex_005
Reason: shabtis are quintessential Egyptian funerary objects, directly matching the theme.

""",
    """\
EXAMPLE:
Exhibition theme: Renaissance Portraiture
Context objects:
  - Portrait of a Young Man (Italian, ca. 1480-1490)
  - Portrait of Giovanna Tornabuoni (Italian, 1488)
Candidates:
  [1] ID=ex_007 | Landscape with Figures | Dutch | 17th century | Oil on panel
  [2] ID=ex_008 | Portrait of a Noblewoman | Italian | ca. 1515 | Oil on panel
  [3] ID=ex_009 | Abstract Composition | American | 1950 | Oil on canvas
Best fit: ex_008
Reason: Italian Renaissance portrait of a noblewoman directly matches the portraiture theme and period.

""",
    """\
EXAMPLE:
Exhibition theme: Islamic Geometric Decoration
Context objects:
  - Mihrab (Prayer Niche) (Iranian, 1354-55)
  - Star-Shaped Tile Panel (Iranian, 13th century)
Candidates:
  [1] ID=ex_010 | Viking Brooch | Norse | 9th century | Silver
  [2] ID=ex_011 | Mamluk Quran Stand with Geometric Inlay | Egyptian | 14th century | Wood, ivory
  [3] ID=ex_012 | Tang Dynasty Horse | Chinese | 8th century | Glazed earthenware
Best fit: ex_011
Reason: the Mamluk Quran stand features intricate Islamic geometric inlay, perfectly matching the theme.

""",
    """\
EXAMPLE:
Exhibition theme: Impressionist Landscapes
Context objects:
  - Water Lilies (French, ca. 1906)
  - Haystacks at Sunset (French, 1890-1891)
Candidates:
  [1] ID=ex_013 | Still Life with Apples | French | 1890s | Oil on canvas
  [2] ID=ex_014 | The Poppy Field near Argenteuil | French | 1873 | Oil on canvas
  [3] ID=ex_015 | Self-Portrait with Bandaged Ear | Dutch | 1889 | Oil on canvas
Best fit: ex_014
Reason: Monet's Poppy Field is an Impressionist landscape, directly fitting the theme. The self-portrait (L3) is not a landscape.

""",
]


def build_meip_prompt(sample: dict, objects: dict[str, dict], shot: int = 0,
                      cot: bool = False) -> str:
    theme = sample.get("exhibition_theme", "")

    # Context objects — stored as list of dicts OR list of ids
    context_raw = sample.get("context", [])
    context_objs = []
    for c in context_raw[:4]:
        if isinstance(c, dict):
            obj = c
        else:
            obj = objects.get(c, {})
        if obj:
            context_objs.append(
                f"  - {obj.get('title','?')} ({obj.get('culture','?')}, {obj.get('date','?')})"
            )
    context_str = "\n".join(context_objs) if context_objs else "  (none)"

    # Candidates — stored as list of dicts OR list of ids
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
    candidates_str = "\n".join(cand_lines)

    # True N-shot: use exactly `shot` examples from MEIP_FEWSHOT_BANK (max 5)
    prefix = ""
    if shot > 0:
        n_ex = min(shot, len(MEIP_FEWSHOT_BANK))
        prefix = "".join(MEIP_FEWSHOT_BANK[:n_ex])
        prefix = f"Here are {n_ex} example(s) to guide you:\n\n" + prefix + "Now solve the following:\n\n"

    template = MEIP_COT_TEMPLATE if cot else MEIP_ZEROSHOT_TEMPLATE
    return prefix + template.format(
        theme=theme,
        context=context_str,
        candidates=candidates_str,
    )


def _get_meip_candidate_ids(sample: dict) -> list[str]:
    """Extract candidate IDs from MEIP sample (handles both dict and id formats)."""
    candidates_raw = sample.get("candidates", sample.get("candidate_ids", []))
    ids = []
    for c in candidates_raw:
        if isinstance(c, dict):
            ids.append(c["id"])
        else:
            ids.append(c)
    return ids


def _load_cached_results(raw_path: Path) -> dict:
    """Load previously successful results from raw_responses jsonl. Returns {sample_id: {mrr, hit1, latency, usage}}."""
    cache = {}
    if not raw_path or not raw_path.exists():
        return cache
    with open(raw_path, errors='ignore') as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get('event') == 'sample' and d.get('status') == 'ok':
                    sid = d['sample_id']
                    metrics = d.get('metrics', {})
                    cache[sid] = {
                        'mrr': metrics.get('mrr', 0),
                        'hit1': metrics.get('hit@1', 0),
                        'latency': d.get('latency_sec', 0),
                        'usage': d.get('usage_tokens', {}),
                    }
            except:
                pass
    return cache


def evaluate_meip(model: str, samples: list[dict], objects: dict[str, dict],
                  max_samples: int = 500, shot: int = 0, workers: int = 100,
                  cot: bool = False, raw_logger: Optional[JsonlRawLogger] = None,
                  resume_cache: Optional[dict] = None) -> dict:
    samples_used = samples[:max_samples]

    # Per-sample worker function
    def _run_one(item):
        i, sample = item
        sample_id = sample.get("id", f"meip_{i:06d}")
        # Resume: skip if already have cached result
        if resume_cache and sample_id in resume_cache:
            cached = resume_cache[sample_id]
            return (i, cached['mrr'], cached['hit1'], cached['latency'], cached['usage'])
        gold_id = sample.get("gold_id", "")
        candidate_ids = _get_meip_candidate_ids(sample)
        if not gold_id or not candidate_ids:
            if raw_logger:
                raw_logger.write(
                    {
                        "event": "sample",
                        "task": "meip",
                        "sample_id": sample_id,
                        "status": "invalid_input",
                        "reason": "missing_gold_or_candidates",
                    }
                )
            return None
        prompt = build_meip_prompt(sample, objects, shot=shot, cot=cot)
        max_tok = 400 if cot else 150
        response_text, latency, usage = call_llm(model, prompt, max_tokens=max_tok)
        if response_text is None:
            if raw_logger:
                raw_logger.write(
                    {
                        "event": "sample",
                        "task": "meip",
                        "sample_id": sample_id,
                        "status": "api_failure",
                        "gold_id": gold_id,
                        "candidate_ids": candidate_ids,
                        "prompt": prompt,
                        "latency_sec": latency,
                        "usage_tokens": usage,
                    }
                )
            return None
        raw_response_original = response_text
        # CoT: extract answer from "ANSWER: <id>" on the last line(s)
        if cot:
            lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
            cot_answer = None
            for line in reversed(lines):
                if line.upper().startswith("ANSWER:"):
                    cot_answer = line.split(":", 1)[1].strip()
                    break
            if cot_answer:
                response_text = cot_answer
        best_id = parse_meip_best_id(response_text, candidate_ids)
        if not best_id:
            if raw_logger:
                raw_logger.write(
                    {
                        "event": "sample",
                        "task": "meip",
                        "sample_id": sample_id,
                        "status": "invalid_output",
                        "reason": "cannot_parse_candidate_id",
                        "gold_id": gold_id,
                        "candidate_ids": candidate_ids,
                        "prompt": prompt,
                        "raw_response_original": raw_response_original,
                        "parsed_response": response_text,
                        "latency_sec": latency,
                        "usage_tokens": usage,
                    }
                )
            return None
        ranked = [best_id] + [x for x in candidate_ids if x != best_id]
        score = mrr(gold_id, ranked)
        hit1 = 1.0 if ranked and ranked[0] == gold_id else 0.0
        if raw_logger:
            raw_logger.write(
                {
                    "event": "sample",
                    "task": "meip",
                    "sample_id": sample_id,
                    "status": "ok",
                    "gold_id": gold_id,
                    "candidate_ids": candidate_ids,
                    "prompt": prompt,
                    "raw_response_original": raw_response_original,
                    "parsed_response": response_text,
                    "parsed_output": {"ranked_ids": ranked},
                    "metrics": {"mrr": score, "hit@1": hit1},
                    "latency_sec": latency,
                    "usage_tokens": usage,
                }
            )
        return (i, score, hit1, latency, usage)

    mrr_scores = []
    hit1_scores = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    done = 0
    lock = Lock()

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
                if done % 50 == 0:
                    log.info(f"  MEIP {model}: {done}/{len(samples_used)}, "
                             f"MRR={sum(mrr_scores)/len(mrr_scores):.4f}")

    n = len(mrr_scores)
    return {
        "task": "meip",
        "model": model,
        "shot": shot,
        "n_samples": n,
        "mrr": round(sum(mrr_scores) / n, 4) if n else 0,
        "hit@1": round(sum(hit1_scores) / n, 4) if n else 0,
        "total_latency_sec": round(total_latency, 2),
        "avg_latency_sec": round(total_latency / n, 3) if n else 0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }


# ── TES Evaluation ────────────────────────────────────────────────────────────

TES_TEMPLATE = """\
You are a museum curator selecting an exhibition. Given a thematic query, \
rank the provided exhibition candidates from most to least relevant.

Query theme: {theme}
Query description: {description}

Exhibition candidates (each identified by an anonymous ID; judge solely by the artworks inside):
{candidates}

Return ONLY the anonymous IDs of the top 10 most relevant exhibitions, in order from best to worst.
Format: EX_001, EX_002, EX_003, ... (comma-separated)
"""

TES_COT_TEMPLATE = """\
You are a museum curator selecting an exhibition. Given a thematic query, \
rank the provided exhibition candidates from most to least relevant.

Query theme: {theme}
Query description: {description}

Exhibition candidates (each identified by an anonymous ID; judge solely by the artworks inside):
{candidates}

Think step by step:
1. What is the core cultural/period/medium focus implied by the query theme?
2. Scan each candidate's artworks: which ones closely match the theme in culture, period, or subject?
3. Rank candidates from best to worst thematic match.

After your reasoning, output your final answer on a new line as:
ANSWER: EX_001, EX_002, EX_003, ... (top-10 comma-separated anonymous IDs, best first)
"""


def build_tes_prompt(sample: dict, cot: bool = False) -> tuple[str, dict[str, str]]:
    """Build a leak-free TES prompt.

    Removes all theme/title/description/real-id from candidates.
    Exposes only the sample artworks.
    Returns (prompt_str, anon_to_real_id mapping).
    """
    theme = sample.get("query_theme", "")
    desc = sample.get("query_description", "")
    cands = sample.get("candidates", [])
    cand_lines = []
    anon_to_real: dict[str, str] = {}
    for i, c in enumerate(cands[:50], 1):  # limit to 50 candidates in prompt
        anon_id = f"EX_{i:03d}"
        anon_to_real[anon_id] = c["id"]
        sample_objs = c.get("sample_objects", [])[:5]
        if sample_objs:
            obj_str = "; ".join(
                f"{o.get('title','?')} ({o.get('culture','?')}, {o.get('date','?') or '?'})"
                for o in sample_objs
            )
        else:
            obj_str = "(no sample objects)"
        cand_lines.append(f"  [{anon_id}] {obj_str}")
    template = TES_COT_TEMPLATE if cot else TES_TEMPLATE
    prompt = template.format(
        theme=theme,
        description=desc[:300],
        candidates="\n".join(cand_lines),
    )
    return prompt, anon_to_real


def evaluate_tes(model: str, samples: list[dict],
                 max_samples: int = 300, shot: int = 0, workers: int = 100,
                 cot: bool = False, raw_logger: Optional[JsonlRawLogger] = None) -> dict:
    samples_used = samples[:max_samples]

    def _run_one(item):
        i, sample = item
        sample_id = sample.get("id", f"tes_{i:06d}")
        gold_ids = set(sample.get("gold_ids", [sample.get("gold_id", "")]))
        all_candidate_ids = [c["id"] for c in sample.get("candidates", [])]
        if not gold_ids or not all_candidate_ids:
            if raw_logger:
                raw_logger.write(
                    {
                        "event": "sample",
                        "task": "tes",
                        "sample_id": sample_id,
                        "status": "invalid_input",
                        "reason": "missing_gold_or_candidates",
                    }
                )
            return None
        # build_tes_prompt now returns (prompt, anon_to_real mapping)
        prompt, anon_to_real = build_tes_prompt(sample, cot=cot)
        prompt = build_tes_fewshot_prefix(shot) + prompt
        real_to_anon = {v: k for k, v in anon_to_real.items()}
        anon_gold_ids = {real_to_anon.get(g, g) for g in gold_ids}
        all_anon_ids = [real_to_anon.get(rid, rid) for rid in all_candidate_ids]
        max_tok = 600 if cot else 300
        response_text, latency, usage = call_llm(model, prompt, max_tokens=max_tok)
        if response_text is None:
            if raw_logger:
                raw_logger.write(
                    {
                        "event": "sample",
                        "task": "tes",
                        "sample_id": sample_id,
                        "status": "api_failure",
                        "gold_ids": sorted(gold_ids),
                        "candidate_ids": all_candidate_ids,
                        "prompt": prompt,
                        "latency_sec": latency,
                        "usage_tokens": usage,
                    }
                )
            return None
        raw_response_original = response_text
        # CoT: extract ranked list from "ANSWER: EX_001, ..." line
        if cot:
            lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
            for line in reversed(lines):
                if line.upper().startswith("ANSWER:"):
                    response_text = line.split(":", 1)[1].strip()
                    break
        # parse from anon space, then map back to real ids
        ranked_anon = parse_tes_ranked_ids(response_text, all_anon_ids)
        if not ranked_anon:
            if raw_logger:
                raw_logger.write(
                    {
                        "event": "sample",
                        "task": "tes",
                        "sample_id": sample_id,
                        "status": "invalid_output",
                        "reason": "cannot_parse_ranking",
                        "gold_ids": sorted(gold_ids),
                        "candidate_ids": all_candidate_ids,
                        "prompt": prompt,
                        "raw_response_original": raw_response_original,
                        "parsed_response": response_text,
                        "latency_sec": latency,
                        "usage_tokens": usage,
                    }
                )
            return None
        ranked_real = [anon_to_real.get(aid, aid) for aid in ranked_anon]
        nd = ndcg_at_k(gold_ids, ranked_real, k=10)
        mr = mrr(next(iter(gold_ids)), ranked_real)
        if raw_logger:
            raw_logger.write(
                {
                    "event": "sample",
                    "task": "tes",
                    "sample_id": sample_id,
                    "status": "ok",
                    "gold_ids": sorted(gold_ids),
                    "candidate_ids": all_candidate_ids,
                    "prompt": prompt,
                    "raw_response_original": raw_response_original,
                    "parsed_response": response_text,
                    "parsed_output": {
                        "ranked_anon_ids": ranked_anon,
                        "ranked_ids": ranked_real,
                    },
                    "metrics": {"ndcg@10": nd, "mrr": mr},
                    "latency_sec": latency,
                    "usage_tokens": usage,
                }
            )
        return (i,
                nd,
                mr,
                latency, usage)

    ndcg_scores = []
    mrr_scores = []
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    done = 0
    lock = Lock()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, (i, s)): i for i, s in enumerate(samples_used)}
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                continue
            _, nd, mr, latency, usage = res
            with lock:
                ndcg_scores.append(nd)
                mrr_scores.append(mr)
                total_latency += latency
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                total_tokens += usage["total_tokens"]
                done += 1
                if done % 30 == 0:
                    log.info(f"  TES {model}: {done}/{len(samples_used)}, "
                             f"NDCG@10={sum(ndcg_scores)/len(ndcg_scores):.4f}")

    n = len(ndcg_scores)
    return {
        "task": "tes",
        "model": model,
        "shot": shot,
        "n_samples": n,
        "ndcg@10": round(sum(ndcg_scores) / n, 4) if n else 0,
        "mrr": round(sum(mrr_scores) / n, 4) if n else 0,
        "total_latency_sec": round(total_latency, 2),
        "avg_latency_sec": round(total_latency / n, 3) if n else 0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }


# ── ECD Evaluation ────────────────────────────────────────────────────────────

ECD_TEMPLATE = """\
You are a museum curator evaluating exhibition coherence.

Exhibition theme: {theme}

Two exhibition sequences are shown below. One is the ORIGINAL coherent sequence; \
the other has been DISRUPTED by swapping one item.

Sequence A:
{seq_a}

Sequence B:
{seq_b}

Which sequence is more coherent and fits the exhibition theme better?
Reply with ONLY "A" or "B".
"""

ECD_COT_TEMPLATE = """\
You are a museum curator evaluating exhibition coherence.

Exhibition theme: {theme}

Two exhibition sequences are shown below. One is the ORIGINAL coherent sequence; \
the other has been DISRUPTED by swapping one item.

Sequence A:
{seq_a}

Sequence B:
{seq_b}

Think step by step:
1. What cultural/period/medium coherence does the exhibition theme demand?
2. Check Sequence A: do all items fit the theme and flow logically?
3. Check Sequence B: do all items fit the theme and flow logically?
4. Which sequence is clearly more coherent?

After your reasoning, output your final answer on the last line as:
ANSWER: A
or
ANSWER: B
"""


def format_seq(obj_list: list[dict]) -> str:
    lines = []
    for i, obj in enumerate(obj_list, 1):
        if obj is None or not isinstance(obj, dict):
            lines.append(f"  {i}. [unknown item]")
            continue
        medium = obj.get('medium') or '?'
        lines.append(
            f"  {i}. {obj.get('title','?')} | "
            f"{obj.get('culture','?')} | {obj.get('date','?')} | "
            f"{str(medium)[:60]}"
        )
    return "\n".join(lines)


def evaluate_ecd(model: str, samples: list[dict], objects: dict[str, dict],
                 max_samples: int = 800, shot: int = 0, workers: int = 100,
                 cot: bool = False, raw_logger: Optional[JsonlRawLogger] = None) -> dict:
    import random
    rng = random.Random(42)
    samples_used = samples[:max_samples]

    # Pre-assign A/B for each sample deterministically
    assignments = [rng.random() > 0.5 for _ in samples_used]

    def _run_one(item):
        i, sample, gold_is_a = item
        sample_id = sample.get("id", f"ecd_{i:06d}")
        positive_entry = sample.get("positive", {})
        negative_entry = sample.get("negative", {})
        level = sample.get("level", 1)
        pos_items = positive_entry.get("items", [])
        neg_items = negative_entry.get("items", [])
        theme = positive_entry.get("theme", sample.get("theme", ""))
        if not pos_items or not neg_items:
            if raw_logger:
                raw_logger.write(
                    {
                        "event": "sample",
                        "task": "ecd",
                        "sample_id": sample_id,
                        "status": "invalid_input",
                        "reason": "missing_pos_or_neg_items",
                    }
                )
            return None
        if gold_is_a:
            seq_a, seq_b = pos_items, neg_items
            correct_answer = "A"
        else:
            seq_a, seq_b = neg_items, pos_items
            correct_answer = "B"
        template = ECD_COT_TEMPLATE if cot else ECD_TEMPLATE
        prompt = template.format(
            theme=theme,
            seq_a=format_seq(seq_a),
            seq_b=format_seq(seq_b),
        )
        prompt = build_ecd_fewshot_prefix(shot) + prompt
        max_tok = 300 if cot else 50
        response_text, latency, usage = call_llm(model, prompt, max_tokens=max_tok)
        if response_text is None:
            if raw_logger:
                raw_logger.write(
                    {
                        "event": "sample",
                        "task": "ecd",
                        "sample_id": sample_id,
                        "status": "api_failure",
                        "level": level,
                        "gold_answer": correct_answer,
                        "prompt": prompt,
                        "latency_sec": latency,
                        "usage_tokens": usage,
                    }
                )
            return None
        raw_response_original = response_text
        # CoT: extract answer from "ANSWER: A/B" on last line(s)
        if cot:
            lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
            for line in reversed(lines):
                if line.upper().startswith("ANSWER:"):
                    response_text = line.split(":", 1)[1].strip()
                    break
        resp_upper = (response_text or "").strip().upper()
        if resp_upper.startswith("A"):
            predicted = "A"
        elif resp_upper.startswith("B"):
            predicted = "B"
        else:
            if raw_logger:
                raw_logger.write(
                    {
                        "event": "sample",
                        "task": "ecd",
                        "sample_id": sample_id,
                        "status": "invalid_output",
                        "reason": "cannot_parse_ab_choice",
                        "level": level,
                        "gold_answer": correct_answer,
                        "prompt": prompt,
                        "raw_response_original": raw_response_original,
                        "parsed_response": response_text,
                        "latency_sec": latency,
                        "usage_tokens": usage,
                    }
                )
            return None
        if raw_logger:
            raw_logger.write(
                {
                    "event": "sample",
                    "task": "ecd",
                    "sample_id": sample_id,
                    "status": "ok",
                    "level": level,
                    "gold_answer": correct_answer,
                    "raw_response_original": raw_response_original,
                    "parsed_response": response_text,
                    "parsed_output": {"predicted_answer": predicted},
                    "is_correct": predicted == correct_answer,
                    "prompt": prompt,
                    "latency_sec": latency,
                    "usage_tokens": usage,
                }
            )
        return (i, level, predicted == correct_answer, latency, usage)

    level_correct = defaultdict(int)
    level_total   = defaultdict(int)
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    n_calls = 0
    done = 0
    lock = Lock()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, (i, s, assignments[i])): i
                for i, s in enumerate(samples_used)}
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                continue
            _, level, is_correct, latency, usage = res
            level_str = f"L{level}"
            with lock:
                level_total[level_str] += 1
                if is_correct:
                    level_correct[level_str] += 1
                total_latency += latency
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                total_tokens += usage["total_tokens"]
                n_calls += 1
                done += 1
                if done % 100 == 0:
                    levels_seen = [l for l in level_total if level_total[l] > 0]
                    macro = sum(level_correct[l] / level_total[l] for l in levels_seen) / len(levels_seen)
                    log.info(f"  ECD {model}: {done}/{len(samples_used)}, "
                             f"Macro PairAcc={macro:.4f}")

    result = {
        "task": "ecd",
        "model": model,
        "shot": shot,
        "n_samples": sum(level_total.values()),
        "total_latency_sec": round(total_latency, 2),
        "avg_latency_sec": round(total_latency / n_calls, 3) if n_calls else 0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }
    levels = ["L1", "L2", "L3", "L4"]
    pair_accs = []
    for lv in levels:
        if level_total[lv] > 0:
            acc = level_correct[lv] / level_total[lv]
        else:
            acc = 0.0
        result[f"pairaccc_{lv}"] = round(acc, 4)
        pair_accs.append(acc)

    valid_accs = [a for a in pair_accs if a > 0]
    result["macro_pairaccc"] = round(sum(valid_accs) / len(valid_accs) if valid_accs else 0, 4)
    return result


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation(
    task: str,
    model: str,
    max_samples: int = 500,
    shot: int = 0,
    force: bool = False,
    resume: bool = False,
    meip_data: Optional[Path] = None,
    tag: str = "",
    workers: int = 100,
    cot: bool = False,
    save_raw: bool = False,
    tes_noleak: bool = False,
) -> Optional[dict]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    normalized_model = model.replace("/", "-").replace(":", "-")
    suffix_parts: list[str] = []
    if task == "tes" and tes_noleak:
        suffix_parts.append("noleak")
    if tag:
        suffix_parts.append(tag)
    suffix = "".join(f"_{part}" for part in suffix_parts)
    out_path = RESULTS / f"{task}_{normalized_model}_shot{shot}{suffix}.json"

    if out_path.exists() and not force:
        log.info(f"Already exists: {out_path}, skipping (use --force to rerun)")
        with open(out_path) as f:
            return json.load(f)

    log.info(f"Running {task} evaluation with {model} (shot={shot}, max={max_samples})")

    # Load objects
    obj_path = find_data_file("objects")
    if not obj_path:
        log.error("No objects file found!")
        return None
    objects = load_objects(obj_path)
    data_fingerprint = {
        "objects_path": str(obj_path.relative_to(BASE)),
        "objects_sha256": _file_sha256(obj_path),
    }

    raw_logger: Optional[JsonlRawLogger] = None
    raw_path = RAW_RESPONSES / f"{task}_{normalized_model}_shot{shot}{suffix}.jsonl"
    if save_raw:
        raw_logger = JsonlRawLogger(
            raw_path,
            run_meta={
                "task": task,
                "model": model,
                "shot": shot,
                "tag": tag,
                "tes_noleak": tes_noleak,
                "max_samples": max_samples,
            },
        )

    # Load resume cache if --resume
    resume_cache = None
    if resume and raw_path.exists():
        resume_cache = _load_cached_results(raw_path)
        log.info(f"Resume mode: loaded {len(resume_cache)} cached results from {raw_path.name}")

    result = None

    try:
        if task == "meip":
            meip_path = meip_data if meip_data else find_data_file("meip_samples")
            if not meip_path:
                log.error("No MEIP samples found!")
                return None
            samples = load_jsonl(meip_path)
            data_fingerprint["task_data_path"] = str(meip_path.relative_to(BASE))
            data_fingerprint["task_data_sha256"] = _file_sha256(meip_path)
            result = evaluate_meip(
                model,
                samples,
                objects,
                max_samples=max_samples,
                shot=shot,
                workers=workers,
                cot=cot,
                raw_logger=raw_logger,
                resume_cache=resume_cache,
            )

        elif task == "tes":
            tes_path = find_data_file("tes_samples")
            if not tes_path:
                log.error("No TES samples found!")
                return None
            samples = load_jsonl(tes_path)
            data_fingerprint["task_data_path"] = str(tes_path.relative_to(BASE))
            data_fingerprint["task_data_sha256"] = _file_sha256(tes_path)
            result = evaluate_tes(
                model,
                samples,
                max_samples=max_samples,
                shot=shot,
                workers=workers,
                cot=cot,
                raw_logger=raw_logger,
            )

        elif task == "ecd":
            ecd_path = find_data_file("ecd_samples")
            if not ecd_path:
                log.error("No ECD samples found!")
                return None
            samples = load_jsonl(ecd_path)
            data_fingerprint["task_data_path"] = str(ecd_path.relative_to(BASE))
            data_fingerprint["task_data_sha256"] = _file_sha256(ecd_path)
            result = evaluate_ecd(
                model,
                samples,
                objects,
                max_samples=max_samples,
                shot=shot,
                workers=workers,
                cot=cot,
                raw_logger=raw_logger,
            )
    finally:
        if raw_logger is not None:
            raw_logger.close()

    if result:
        result["run_meta"] = {
            "api_base_internal": INTERNAL_API_BASE,
            "api_base_xty": XTY_API_BASE,
            "workers": workers,
            "cot": cot,
            "save_raw": save_raw,
            "tes_noleak": tes_noleak,
            "tag": tag,
            "data_fingerprint": data_fingerprint,
            "timestamp": time.time(),
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"Saved results to {out_path}")
        log.info(f"Result: {result}")

    return result


# ── Summary Table ─────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    meip_results = {r["model"]: r for r in results if r.get("task") == "meip"}
    tes_results  = {r["model"]: r for r in results if r.get("task") == "tes"}
    ecd_results  = {r["model"]: r for r in results if r.get("task") == "ecd"}

    all_models_in_results = sorted(set(
        list(meip_results.keys()) + list(tes_results.keys()) + list(ecd_results.keys())
    ))

    header = (
        f"\n{'Model':<22} | {'MEIP MRR':>9} {'Hit@1':>7} | "
        f"{'TES NDCG@10':>11} {'MRR':>7} | "
        f"{'ECD L1':>7} {'L2':>7} {'L3':>7} {'L4':>7} {'Macro':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print("ExhibitionBench Results Summary")
    print(sep)
    print(header)
    print(sep)

    for m in all_models_in_results:
        mr = meip_results.get(m, {})
        tr = tes_results.get(m, {})
        er = ecd_results.get(m, {})
        line = (
            f"{m:<22} | "
            f"{mr.get('mrr', '-'):>9} {mr.get('hit@1', '-'):>7} | "
            f"{tr.get('ndcg@10', '-'):>11} {tr.get('mrr', '-'):>7} | "
            f"{er.get('pairaccc_L1', '-'):>7} {er.get('pairaccc_L2', '-'):>7} "
            f"{er.get('pairaccc_L3', '-'):>7} {er.get('pairaccc_L4', '-'):>7} "
            f"{er.get('macro_pairaccc', '-'):>7}"
        )
        print(line)

    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="ExhibitionBench SOTA Evaluation")
    parser.add_argument("--task", "--tasks", dest="task", default="all",
                        nargs="+",
                        help="Task(s) to evaluate: meip tes ecd all (space separated)")
    parser.add_argument("--model", "--models", dest="model", default="all",
                        nargs="+",
                        help=f"Model(s) or 'all'. Available: {', '.join(ALL_MODELS)}")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max samples per task per model")
    parser.add_argument("--shot", type=int, nargs="+", default=[0],
                        help="Number of few-shot examples (0=zero-shot). Multiple values allowed, e.g. --shot 0 1 3")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results file exists")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: only run samples that failed/missing in previous run (reads raw_responses)")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of existing results")
    parser.add_argument("--meip-data", type=str, default=None,
                        help="Path to custom MEIP samples file (e.g. data/meip_samples_v4.jsonl)")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag appended to result filename (e.g. 'v4clean')")
    parser.add_argument("--workers", type=int, default=100,
                        help="Number of concurrent worker threads (default: 100). "
                             "Lower for rate-limited/reasoning models, e.g. --workers 10")
    parser.add_argument("--cot", action="store_true",
                        help="Use Chain-of-Thought prompting (adds _cot tag to output file)")
    parser.add_argument("--save-raw", action="store_true",
                        help="Save per-sample prompt/raw response traces to results/raw_responses")
    parser.add_argument("--tes-noleak", action="store_true",
                        help="Add _noleak marker to TES output filename for leak-free protocol")
    args = parser.parse_args()

    if args.summary:
        # Load all existing results and print summary
        all_results = []
        for p in RESULTS.glob("*.json"):
            with open(p) as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_results.extend(data)
                    elif isinstance(data, dict):
                        all_results.append(data)
                except Exception:
                    pass
        print_summary_table(all_results)
        return

    # Determine models  (args.model is now a list thanks to nargs="+")
    raw_models = args.model if isinstance(args.model, list) else [args.model]
    flat_models = []
    for m in raw_models:
        for part in m.replace(",", " ").split():
            flat_models.append(part)
    if flat_models == ["all"] or flat_models == ["default"]:
        models = DEFAULT_MODELS
    else:
        models = flat_models
    unknown_models = [m for m in models if m not in MODELS]
    if unknown_models:
        raise ValueError(
            f"Unknown model(s): {unknown_models}. "
            f"Allowed models: {sorted(MODELS.keys())}"
        )

    # Determine tasks  (args.task is now a list thanks to nargs="+")
    raw_tasks = args.task if isinstance(args.task, list) else [args.task]
    all_valid = {"meip", "tes", "ecd", "all"}
    tasks = []
    for t in raw_tasks:
        for part in t.replace(",", " ").split():
            if part == "all":
                tasks = ["meip", "tes", "ecd"]
                break
            elif part in all_valid:
                tasks.append(part)
    if not tasks:
        tasks = ["meip", "tes", "ecd"]

    shots = args.shot  # now a list, e.g. [0] or [1, 3]
    log.info(f"Evaluating {len(models)} model(s) on {len(tasks)} task(s), shots={shots}")
    log.info(f"Models: {models}")
    log.info(f"Tasks:  {tasks}")

    # Auto-add cot tag if --cot is specified
    cot_tag = "cot" if args.cot else ""
    auto_save_raw = args.save_raw or (args.tag == "fullrun")

    all_results = []
    for shot in shots:
        for model in models:
            for task in tasks:
                effective_tag = args.tag or cot_tag
                result = run_evaluation(
                    task=task,
                    model=model,
                    max_samples=args.max_samples,
                    shot=shot,
                    force=args.force or args.resume,
                    resume=args.resume,
                    meip_data=Path(args.meip_data) if args.meip_data else None,
                    tag=effective_tag,
                    workers=args.workers,
                    cot=args.cot,
                    save_raw=auto_save_raw,
                    tes_noleak=args.tes_noleak,
                )
                if result:
                    all_results.append(result)

    if all_results:
        print_summary_table(all_results)

        # Save aggregated summary (use last shot value if multiple shots given)
        summary_path = RESULTS / f"summary_shot{shots[-1]}.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
