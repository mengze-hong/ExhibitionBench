"""
scripts/generate_human_eval_outputs.py
=======================================
为 Human Eval tab 预生成模型输出。

选取 20 个代表性样本（每任务）× 3 个强模型并行调用，
结果存到 human_eval/ 目录，供 nicegui_app.py 的 Human Eval tab 加载。

用法：
    python scripts/generate_human_eval_outputs.py              # 全部任务
    python scripts/generate_human_eval_outputs.py --tasks meip ecd
    python scripts/generate_human_eval_outputs.py --n 10       # 每任务只选 10 条（快速测试）
"""

from __future__ import annotations
import argparse, json, logging, random, re, time
import concurrent.futures
from pathlib import Path

BASE     = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
OUT_DIR  = BASE / "human_eval"
OUT_DIR.mkdir(exist_ok=True)

API_KEY  = "sk-TpK0g832p8LbMXTdI_pjkQ"
API_BASE = "http://csig.litellm.prod.sgpolaris/v1"

# 三个强模型（已验证可用）
DEFAULT_MODELS = ["gpt-5.2", "deepseek-v3.2", "kimi-k2.5"]
N_PER_TASK     = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# LLM 调用（带重试）
# ──────────────────────────────────────────────
def call_llm(model: str, prompt: str, max_tokens: int = 400, retries: int = 3) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=30,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            log.warning(f"[{model}] attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return ""


# ──────────────────────────────────────────────
# 数据文件选择
# ──────────────────────────────────────────────
def _pick_file(stem: str) -> Path:
    for suffix in ["_v4", "_v3_fixed", "_v3", "_v2", ""]:
        p = DATA_DIR / f"{stem}{suffix}.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 {stem}*.jsonl")


# ──────────────────────────────────────────────
# Artwork 字典（用于 join candidate_ids → 完整对象）
# ──────────────────────────────────────────────
def _load_objects() -> dict:
    """加载 data/objects.jsonl，返回 {id: artwork_dict}"""
    obj_path = DATA_DIR / "objects.jsonl"
    if not obj_path.exists():
        log.warning("objects.jsonl 未找到，context/candidates 可能只有 id")
        return {}
    objs = {}
    with open(obj_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            objs[r["id"]] = r
    log.info(f"加载 {len(objs):,} 件展品（用于 join）")
    return objs


def _enrich_meip_sample(sample: dict, objects: dict) -> dict:
    """把 meip sample 的 context id-only 和 candidate_ids 展开成完整对象列表。"""
    import copy
    s = copy.deepcopy(sample)
    # context: 可能是 [{id:...}, ...] 或已完整
    ctx = s.get("context", [])
    enriched_ctx = []
    for item in ctx:
        if isinstance(item, dict) and len(item) <= 2 and "id" in item:
            enriched_ctx.append(objects.get(item["id"], item))
        else:
            enriched_ctx.append(item)
    s["context"] = enriched_ctx
    # candidates: 可能是 candidate_ids 列表 或 已有 candidates key
    if not s.get("candidates") and s.get("candidate_ids"):
        s["candidates"] = [objects.get(cid, {"id": cid, "title": cid}) for cid in s["candidate_ids"]]
    return s


# ──────────────────────────────────────────────
# 样本选取（代表性采样）
# ──────────────────────────────────────────────
def select_samples(n: int = N_PER_TASK) -> dict[str, list[dict]]:
    rng = random.Random(42)
    objects = _load_objects()  # 用于 enrich meip context/candidates

    # MEIP：只选 context 和 candidates 已有完整信息的样本，按展览主题分层
    meip_all_raw = [json.loads(l) for l in open(_pick_file("meip_samples"), encoding="utf-8")]
    # 过滤出 context 有 title（完整对象）且 candidates 非空的样本
    def _is_full_meip(s: dict) -> bool:
        ctx = s.get("context", [])
        cands = s.get("candidates", [])
        if not ctx or not cands:
            return False
        return isinstance(ctx[0], dict) and "title" in ctx[0] and \
               isinstance(cands[0], dict) and "title" in cands[0]
    meip_all = [s for s in meip_all_raw if _is_full_meip(s)]
    log.info(f"MEIP 完整样本: {len(meip_all)}/{len(meip_all_raw)}")
    by_theme: dict[str, list] = {}
    for s in meip_all:
        t = s.get("exhibition_theme", "other")
        by_theme.setdefault(t, []).append(s)
    theme_list = sorted(by_theme.keys())
    rng.shuffle(theme_list)
    meip_sel: list[dict] = []
    for t in theme_list:
        if len(meip_sel) >= n:
            break
        meip_sel.append(rng.choice(by_theme[t]))
    log.info(f"MEIP: 选取 {len(meip_sel)} 条（来自 {len(theme_list)} 个主题）")

    # ECD：L1–L4 各 n/4 条，确保难度分布均匀
    ecd_all = [json.loads(l) for l in open(_pick_file("ecd_samples"), encoding="utf-8")]
    by_level: dict[int, list] = {i: [] for i in range(1, 5)}
    for s in ecd_all:
        by_level[s.get("level", 1)].append(s)
    ecd_sel: list[dict] = []
    each = max(1, n // 4)
    for lvl in range(1, 5):
        pool = by_level[lvl][:]
        rng.shuffle(pool)
        ecd_sel.extend(pool[:each])
    log.info(f"ECD: 选取 {len(ecd_sel)} 条（L1-L4 各 {each} 条）")

    # TES：随机多元主题
    tes_all = [json.loads(l) for l in open(_pick_file("tes_samples"), encoding="utf-8")]
    rng.shuffle(tes_all)
    tes_sel = tes_all[:n]
    log.info(f"TES: 选取 {len(tes_sel)} 条")

    return {"meip": meip_sel, "ecd": ecd_sel, "tes": tes_sel}


# ──────────────────────────────────────────────
# 任务专用生成函数
# ──────────────────────────────────────────────
def gen_meip(model: str, sample: dict) -> dict:
    context    = sample.get("context", [])
    candidates = sample.get("candidates", [])
    gold_id    = sample.get("gold_id", "")
    theme      = sample.get("exhibition_theme", "")

    ctx_block  = "\n".join(
        f"  {i+1}. {it.get('title','')} | {it.get('culture','')} | {it.get('date','')} | {it.get('medium','')}"
        for i, it in enumerate(context)
    )
    cand_block = "\n".join(
        f"  [{i+1}] {c.get('title','')} | {c.get('culture','')} | {c.get('date','')} | {c.get('medium','')}"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"You are an expert museum curator.\n"
        f"Exhibition theme: \"{theme}\"\n\n"
        f"Artworks already in the exhibition:\n{ctx_block}\n\n"
        f"Candidates to complete the exhibition (pick the single best):\n{cand_block}\n\n"
        "Reply with:\n"
        "1. The candidate number [1-10]\n"
        "2. A 2-3 sentence curatorial justification explaining why this artwork best "
        "completes the exhibition's narrative."
    )
    raw       = call_llm(model, prompt)
    num_match = re.search(r'\b([1-9]|10)\b', raw)
    choice    = int(num_match.group(1)) if num_match else -1
    pred      = candidates[choice - 1] if 1 <= choice <= len(candidates) else {}
    return {
        "raw_response":    raw,
        "choice_num":      choice,
        "predicted_id":    pred.get("id", ""),
        "predicted_title": pred.get("title", ""),
        "correct":         pred.get("id", "") == gold_id,
        "gold_id":         gold_id,
        "gold_title":      next((c.get("title","") for c in candidates if c.get("id","") == gold_id), ""),
    }


def gen_ecd(model: str, sample: dict) -> dict:
    pos   = sample.get("positive", {})
    neg   = sample.get("negative", {})
    theme = pos.get("theme", "")
    gold  = sample.get("label", 0)   # 0 = A(positive) 是真实序列

    def fmt(items: list) -> str:
        return "\n".join(
            f"  {i+1}. {it.get('title','')} | {it.get('culture','')} | "
            f"{it.get('date','')} | {it.get('medium','')}"
            for i, it in enumerate(items)
        )

    prompt = (
        f"You are an expert museum curator evaluating exhibition coherence.\n"
        f"Theme: \"{theme}\"\n\n"
        f"Sequence A:\n{fmt(pos.get('items',[]))}\n\n"
        f"Sequence B:\n{fmt(neg.get('items',[]))}\n\n"
        "Which sequence is more thematically coherent for this exhibition?\n"
        "Reply with:\n"
        "1. Your verdict: 'A' or 'B'\n"
        "2. A 2-3 sentence explanation identifying the specific reason one sequence "
        "is weaker (e.g., anachronism, cultural mismatch, thematic drift)."
    )
    raw    = call_llm(model, prompt)
    choice = 0 if re.search(r'\bA\b', raw) else 1
    return {
        "raw_response": raw,
        "choice":       choice,          # 0=chose A(positive), 1=chose B(negative)
        "correct":      choice == gold,
        "gold_label":   gold,            # 0=A is correct
        "level":        sample.get("level", "?"),
        "perturbation": sample.get("perturbation_type", ""),
    }


def gen_tes(model: str, sample: dict) -> dict:
    q_theme    = sample.get("query_theme", "")
    q_desc     = sample.get("query_description", "")
    candidates = sample.get("candidates", [])
    gold_id    = sample.get("gold_id", "")
    gold_ids   = sample.get("gold_ids", [gold_id] if gold_id else [])

    cand_block = "\n".join(
        f"  [{i+1}] {c.get('title', c.get('theme',''))} — {str(c.get('description',''))[:80]}"
        for i, c in enumerate(candidates[:50])
    )
    prompt = (
        f"You are an expert museum curator selecting relevant exhibitions.\n"
        f"Query theme: \"{q_theme}\"\n"
        f"Description: {q_desc}\n\n"
        f"Rate the following {min(50, len(candidates))} exhibition themes by relevance.\n\n"
        f"Candidates:\n{cand_block}\n\n"
        "Reply with:\n"
        "1. A JSON array of the top-10 most relevant IDs (1-based), e.g. [3,1,7,...]\n"
        "2. One sentence explaining your ranking criteria."
    )
    raw       = call_llm(model, prompt, max_tokens=300)
    arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
    ranked_ids, ranked_titles = [], []
    if arr_match:
        try:
            for n in json.loads(arr_match.group()):
                if 1 <= int(n) <= len(candidates):
                    c = candidates[int(n) - 1]
                    ranked_ids.append(c.get("id", ""))
                    ranked_titles.append(c.get("title", c.get("theme", "")))
        except Exception:
            pass
    hit      = any(gid in ranked_ids for gid in gold_ids)
    hit_rank = next((i + 1 for i, rid in enumerate(ranked_ids) if rid in gold_ids), None)
    return {
        "raw_response":  raw,
        "ranked_ids":    ranked_ids[:10],
        "ranked_titles": ranked_titles[:10],
        "hit":           hit,
        "hit_rank":      hit_rank,
        "gold_ids":      gold_ids,
    }


# ──────────────────────────────────────────────
# 批量生成（并行）
# ──────────────────────────────────────────────
def generate_task(task: str, samples: list, gen_fn, models: list) -> list[dict]:
    results = []
    for i, sample in enumerate(samples):
        sid = sample.get("id", f"{task}_{i:04d}")
        log.info(f"[{task}] {i+1}/{len(samples)}: {sid}")
        entry = {"sample_id": sid, "task": task, "sample": sample}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
            futs = {ex.submit(gen_fn, m, sample): m for m in models}
            for fut in concurrent.futures.as_completed(futs):
                m = futs[fut]
                try:
                    entry[m] = fut.result()
                    c = entry[m].get("correct", "?")
                    log.info(f"  [{m}] correct={c}")
                except Exception as e:
                    log.error(f"  [{m}] error: {e}")
                    entry[m] = {"error": str(e), "raw_response": ""}
        results.append(entry)
    return results


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="生成 Human Eval 模型输出")
    parser.add_argument("--tasks",  nargs="+", default=["meip", "ecd", "tes"],
                        choices=["meip", "ecd", "tes"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n",      type=int,  default=N_PER_TASK,
                        help="每任务选取样本数（默认 20）")
    parser.add_argument("--force",  action="store_true",
                        help="强制重新生成（覆盖已有文件）")
    args = parser.parse_args()

    log.info(f"模型: {args.models}  任务: {args.tasks}  每任务样本数: {args.n}")
    samples = select_samples(args.n)

    task_cfg = {
        "meip": (samples["meip"], gen_meip),
        "ecd":  (samples["ecd"],  gen_ecd),
        "tes":  (samples["tes"],  gen_tes),
    }

    for task in args.tasks:
        out_file = OUT_DIR / f"human_eval_{task}.jsonl"
        if out_file.exists() and not args.force:
            existing = sum(1 for _ in open(out_file, encoding="utf-8"))
            log.info(f"[{task}] 已有 {existing} 条结果，跳过（用 --force 重新生成）")
            continue

        task_samples, gen_fn = task_cfg[task]
        log.info(f"=== {task.upper()} | {len(task_samples)} 样本 × {len(args.models)} 模型 ===")
        results = generate_task(task, task_samples, gen_fn, args.models)

        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"[{task}] ✓ 已保存 {len(results)} 条 → {out_file}")

    log.info("=== 生成完成！运行 nicegui_app.py 打开 Human Eval tab ===")


if __name__ == "__main__":
    main()
