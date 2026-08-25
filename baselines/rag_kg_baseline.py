"""
baselines/rag_kg_baseline.py
=============================
RAG + 知识图谱增强 baseline。

构建一个以 CIDOC CRM 字段为骨架的 mini 知识图谱：
  artwork → creator, period, style, culture, medium, subject

推理时，检索与上下文相关的 KG 三元组，拼入 LLM prompt，增强推理能力。

使用方法：
  python baselines/rag_kg_baseline.py build \\
      --objects data/objects.jsonl \\
      --output data/kg.json

  python baselines/rag_kg_baseline.py meip \\
      --input data/meip_samples.jsonl \\
      --kg data/kg.json \\
      --output results/rag_kg_meip_pred.jsonl \\
      --model gpt-5.2
"""

from __future__ import annotations
import json
import re
import time
import argparse
import logging
import random
import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

try:
    from .data_utils import meip_candidates, meip_context
except ImportError:  # Direct execution: python baselines/rag_kg_baseline.py
    from data_utils import meip_candidates, meip_context

log = logging.getLogger(__name__)

INTERNAL_API_BASE = os.environ.get("LLM_API_BASE", "http://YOUR_LLM_API_BASE").rstrip("/")
INTERNAL_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
DEFAULT_MODEL = "gpt-5.2"


def _require_api_key() -> str:
    if not INTERNAL_API_KEY:
        raise RuntimeError("Missing LLM_API_KEY environment variable")
    return INTERNAL_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# KG 构建
# ─────────────────────────────────────────────────────────────────────────────

def build_kg(objects_path: Path, output_path: Path) -> dict:
    """
    从 objects.jsonl 构建 mini 知识图谱。

    结构：
    {
      "obj_id": {
        "triples": [
          ("artwork_title", "hasCreator", "artist_name"),
          ("artwork_title", "hasCulture", "French"),
          ("artwork_title", "hasMedium", "oil on canvas"),
          ("artwork_title", "hasDate", "1890"),
          ...
        ],
        "text": "flat text representation for retrieval"
      }
    }
    """
    kg = {}
    with open(objects_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            oid = obj["id"]
            title = obj.get("title", "Unknown")
            triples = []

            if obj.get("culture"):
                triples.append((title, "hasCulture", obj["culture"]))
            if obj.get("medium"):
                triples.append((title, "hasMedium", obj["medium"]))
            if obj.get("date"):
                triples.append((title, "hasDate", obj["date"]))
            if obj.get("description"):
                # 从描述里提取艺术家（简单规则：描述里常含 "by ARTIST"）
                m = re.search(r'\bby\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)', obj["description"])
                if m:
                    triples.append((title, "hasCreator", m.group(1)))

            # 推断风格（粗略规则，实际可替换为 LLM 分类）
            text_lower = (title + " " + obj.get("description", "")).lower()
            style_keywords = {
                "Impressionism": ["impression", "monet", "renoir", "pissarro", "light effect"],
                "Baroque": ["baroque", "caravaggio", "rembrandt", "dramatic light"],
                "Renaissance": ["renaissance", "raphael", "da vinci", "perspective"],
                "Ancient": ["ancient", "bc", "bce", "dynasty", "ritual", "tomb"],
                "Modern": ["modern", "abstract", "cubism", "picasso", "matisse"],
                "Asian Traditional": ["chinese", "japanese", "korean", "silk", "porcelain", "jade"],
                "Islamic Art": ["islamic", "mosque", "calligraphy", "arabesque", "quran"],
            }
            for style, keywords in style_keywords.items():
                if any(k in text_lower for k in keywords):
                    triples.append((title, "hasStyle", style))
                    break

            kg[oid] = {
                "triples": triples,
                "text": " | ".join(f"{p} {o}" for _, p, o in triples),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    log.info(f"KG 已写入 {output_path}，共 {len(kg)} 个实体")
    return kg


def load_kg(kg_path: Path) -> dict:
    with open(kg_path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# KG 检索：找与上下文相关的三元组
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_kg_context(context_ids: list[str], candidate_ids: list[str], kg: dict) -> str:
    """
    为推理检索相关 KG 三元组。
    返回格式化的字符串，直接拼入 prompt。
    """
    lines = []
    for oid in context_ids + candidate_ids:
        if oid in kg and kg[oid]["triples"]:
            for subj, pred, obj in kg[oid]["triples"]:
                lines.append(f"  ({subj}, {pred}, {obj})")
    if not lines:
        return ""
    return "Relevant knowledge graph facts:\n" + "\n".join(lines[:40])  # 最多 40 条，控制长度


# ─────────────────────────────────────────────────────────────────────────────
# MEIP RAG+KG baseline
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert museum curator with deep knowledge of art history.
You have access to structured knowledge about artworks. Use the provided knowledge graph facts
to reason about which candidate artwork best fits the exhibition's curatorial logic.
"""

def run_meip_rag_kg(client: OpenAI, sample: dict, kg: dict,
                    objects: dict[str, dict], model: str = DEFAULT_MODEL) -> dict:
    context = meip_context(sample, objects)
    candidates = meip_candidates(sample, objects)
    context_ids = [c["id"] for c in context]
    candidate_ids = [c["id"] for c in candidates]

    kg_text = retrieve_kg_context(context_ids, candidate_ids, kg)

    def obj_line(obj: dict) -> str:
        return (
            f'ID:{obj["id"]} | {obj.get("title","")} | {obj.get("culture","")} | '
            f'{obj.get("medium","")} | {obj.get("date","")}'
        )

    ctx_block = "\n".join(f"- {obj_line(c)}" for c in context)
    cand_block = "\n".join(f"- {obj_line(c)}" for c in candidates)

    user_content = (
        f"## Exhibition Context\nThese artworks are already in the exhibition:\n{ctx_block}\n\n"
        f"## Candidates\n{cand_block}\n\n"
    )
    if kg_text:
        user_content += f"## Knowledge Graph Facts\n{kg_text}\n\n"

    user_content += (
        "Rank ALL candidates from most to least suitable to complete this exhibition. "
        "Return ONLY a JSON array of candidate IDs, e.g. [\"id1\", \"id2\", ...]"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not arr_match:
            raise ValueError(f"No JSON array in: {raw[:200]}")
        ranked_ids = [str(x) for x in json.loads(arr_match.group())]
        ranked_ids = [r for r in ranked_ids if r in candidate_ids]
        missing = [cid for cid in candidate_ids if cid not in ranked_ids]
        ranked_ids += missing
        return {"id": sample["id"], "ranked_ids": ranked_ids, "status": "ok"}
    except Exception as e:
        log.warning(f"[{sample['id']}] 失败: {e}")
        fallback = candidate_ids[:]
        random.shuffle(fallback)
        return {"id": sample["id"], "ranked_ids": fallback, "status": f"error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="RAG+KG baseline")
    subparsers = parser.add_subparsers(dest="cmd")

    build_p = subparsers.add_parser("build", help="构建知识图谱")
    build_p.add_argument("--objects", default="data/objects.jsonl")
    build_p.add_argument("--output", default="data/kg.json")

    meip_p = subparsers.add_parser("meip", help="运行 MEIP RAG+KG baseline")
    meip_p.add_argument("--input", required=True)
    meip_p.add_argument("--kg", default="data/kg.json")
    meip_p.add_argument("--output", required=True)
    meip_p.add_argument("--model", default=DEFAULT_MODEL)
    meip_p.add_argument("--objects", default="data/objects.jsonl")
    meip_p.add_argument("--max-samples", type=int, default=None)
    meip_p.add_argument("--sleep", type=float, default=0.3)

    args = parser.parse_args()
    base = Path(__file__).resolve().parent.parent

    if args.cmd == "build":
        build_kg(base / args.objects, base / args.output)

    elif args.cmd == "meip":
        kg = load_kg(base / args.kg)
        with open(base / args.objects, encoding="utf-8") as handle:
            objects = {obj["id"]: obj for line in handle if (obj := json.loads(line))}
        api_base = INTERNAL_API_BASE if INTERNAL_API_BASE.endswith("/v1") else f"{INTERNAL_API_BASE}/v1"
        client = OpenAI(api_key=_require_api_key(), base_url=api_base)

        samples = []
        with open(base / args.input, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        if args.max_samples:
            samples = samples[:args.max_samples]

        out_path = base / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done_ids = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass

        with open(out_path, "a", encoding="utf-8") as out:
            for sample in tqdm(samples, desc=f"RAG+KG MEIP [{args.model}]"):
                if sample["id"] in done_ids:
                    continue
                result = run_meip_rag_kg(client, sample, kg, objects, model=args.model)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                time.sleep(args.sleep)
        log.info(f"完成 → {out_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
