"""
meip_hard_generator.py
======================
为 ExhibitionBench 生成两个增强版 MEIP 评测任务：

  MEIP-Hard : 9 个候选改用 SBERT 硬负例（同文化/主题最相似展品），替代原随机负例
  MEIP-Open : 移除候选列表，模型需从全库（23 658 件）自主检索正确展品

用法
----
    python benchmark/meip_hard_generator.py

依赖
----
    pip install sentence-transformers numpy tqdm
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\mengzehong\Desktop\展览馆llm")
DATA_DIR = BASE_DIR / "data"

OBJECTS_PATH    = DATA_DIR / "objects.jsonl"
MEIP_PATH       = DATA_DIR / "meip_samples.jsonl"
OUT_HARD_PATH   = DATA_DIR / "meip_hard_samples.jsonl"
OUT_OPEN_PATH   = DATA_DIR / "meip_open_samples.jsonl"

# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------
SBERT_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE   = 256     # SBERT 编码批大小
TOP_K        = 50      # 从最相似的 TOP_K 个展品中采硬负例
N_HARD_NEG   = 9       # 每个样本的硬负例数量
TOTAL_CANDS  = N_HARD_NEG + 1   # 候选总数 = 9 硬负例 + 1 gold
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 工具函数
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """按行读取 JSONL，返回 list[dict]。"""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  [加载] {len(records):>6,} 条  <-  {path.name}")
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    """将 list[dict] 写出为 JSONL（utf-8）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [写出] {len(records):>6,} 条  ->  {path.name}")


def make_obj_text(obj: dict) -> str:
    """
    将展品核心字段拼成 SBERT 编码文本。
    格式：title | culture | date | medium
    空字段自动跳过。
    """
    fields = [
        obj.get("title",   "") or "",
        obj.get("culture", "") or "",
        obj.get("date",    "") or "",
        obj.get("medium",  "") or "",
    ]
    return " | ".join(f for f in fields if f.strip())


# ============================================================
# Step 1 — 加载数据
# ============================================================

def load_data() -> tuple[list[dict], dict[str, dict], list[str], list[dict]]:
    """
    返回：
        objects    : 全部展品列表
        obj_by_id  : id -> 展品 dict
        all_ids    : 展品 id 列表（与 objects 行对齐）
        meip_easy  : 原始 MEIP-Easy 样本列表
    """
    print("[1/5] 加载数据文件 ...")
    objects   = load_jsonl(OBJECTS_PATH)
    meip_easy = load_jsonl(MEIP_PATH)

    obj_by_id: dict[str, dict] = {o["id"]: o for o in objects}
    all_ids   = [o["id"] for o in objects]
    return objects, obj_by_id, all_ids, meip_easy


# ============================================================
# Step 2 — SBERT 全库编码
# ============================================================

def encode_corpus(objects: list[dict]) -> np.ndarray:
    """
    用 SBERT 对所有展品编码。
    返回 shape=(N, D) float32 矩阵，**已 L2 归一化**（点积即余弦相似度）。
    """
    print(f"\n[2/5] SBERT 编码 {len(objects):,} 件展品 ...")
    print(f"      model={SBERT_MODEL}  batch_size={BATCH_SIZE}")

    model = SentenceTransformer(SBERT_MODEL)
    texts = [make_obj_text(o) for o in objects]

    t0 = time.time()
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 归一化，点积 = 余弦相似度
    ).astype(np.float32)
    elapsed = time.time() - t0

    print(f"  [完成] 编码耗时 {elapsed:.1f}s，向量维度={vecs.shape[1]}")
    return vecs                      # shape: (N, D)


# ============================================================
# Step 3 — 生成 MEIP-Hard
# ============================================================

def build_hard_samples(
    meip_easy : list[dict],
    obj_by_id : dict[str, dict],
    all_ids   : list[str],
    corpus_vecs: np.ndarray,
) -> list[dict]:
    """
    对每个 MEIP 样本：
      1. 用 gold 的向量计算与全库的余弦相似度
      2. 排除 gold 本身及 context 中已出现的展品
      3. 取 top-50 最相似展品作为候选池，随机采样 9 个作为硬负例
      4. 候选列表 = gold + 9 硬负例，打乱顺序

    返回 hard_records 列表，同时打印相似度统计。
    """
    print(f"\n[3/5] 生成 MEIP-Hard（TOP_K={TOP_K}, N_HARD_NEG={N_HARD_NEG}）...")

    id_to_idx: dict[str, int] = {oid: i for i, oid in enumerate(all_ids)}

    hard_records: list[dict] = []
    sim_hard_per_sample: list[float] = []   # 每个样本硬负例均值
    sim_easy_per_sample: list[float] = []   # 每个样本原随机负例均值
    culture_counter: Counter = Counter()
    skipped = 0

    for sample in tqdm(meip_easy, desc="  硬负例生成", ncols=80, unit="样本"):
        sample_id = sample["id"]
        gold_id   = sample["gold_id"]

        # ── context 中已用的展品 id（排除出候选池）──
        ctx_ids: set[str] = set()
        for c in sample.get("context", []):
            if isinstance(c, dict):
                ctx_ids.add(c.get("id", ""))
            elif isinstance(c, str):
                ctx_ids.add(c)
        exclude_ids = ctx_ids | {gold_id}

        # ── 检查 gold 是否在展品库 ──
        if gold_id not in id_to_idx:
            print(f"\n  [WARN] gold_id={gold_id!r} 不在展品库，跳过 {sample_id}")
            skipped += 1
            continue

        # ── 计算余弦相似度（已归一化，点积即余弦）──
        gold_vec = corpus_vecs[id_to_idx[gold_id]]  # shape (D,)
        sims     = corpus_vecs @ gold_vec            # shape (N,)

        # ── 按相似度降序排列，筛出候选池（排除 gold + context）──
        sorted_idx = np.argsort(-sims)
        pool_idx: list[int] = []
        for idx in sorted_idx:
            if len(pool_idx) >= TOP_K:
                break
            if all_ids[idx] not in exclude_ids:
                pool_idx.append(int(idx))

        if len(pool_idx) < N_HARD_NEG:
            # 候选池不足（理论上极少，23k 展品中只需 9 个）
            # 放宽策略：只排除 gold 本身
            pool_idx = []
            for idx in sorted_idx:
                if len(pool_idx) >= TOP_K * 2:
                    break
                if all_ids[idx] != gold_id:
                    pool_idx.append(int(idx))
            if len(pool_idx) < N_HARD_NEG:
                print(f"\n  [WARN] {sample_id} 候选池仍不足 {N_HARD_NEG}，跳过")
                skipped += 1
                continue

        # ── 随机采样 N_HARD_NEG 个硬负例 ──
        chosen    = random.sample(pool_idx, N_HARD_NEG)
        hard_ids  = [all_ids[i] for i in chosen]
        hard_sims = [float(sims[i]) for i in chosen]

        # ── 计算原 MEIP-Easy 随机负例的相似度（用于对比统计）──
        easy_neg_sims: list[float] = []
        for c in sample.get("candidates", []):
            cid = c["id"] if isinstance(c, dict) else c
            if cid != gold_id and cid in id_to_idx:
                easy_neg_sims.append(float(sims[id_to_idx[cid]]))

        avg_hard_sim = float(np.mean(hard_sims))
        avg_easy_sim = float(np.mean(easy_neg_sims)) if easy_neg_sims else 0.0
        sim_hard_per_sample.append(avg_hard_sim)
        sim_easy_per_sample.append(avg_easy_sim)

        # ── 构造候选列表（内嵌完整展品 dict，打乱顺序）──
        candidates: list[dict] = [obj_by_id[gold_id]] + [obj_by_id[h] for h in hard_ids]
        random.shuffle(candidates)

        # ── culture 统计 ──
        gold_culture = (obj_by_id[gold_id].get("culture") or "Unknown").split(",")[0].strip()
        culture_counter[gold_culture] += 1

        # ── 组装输出 record ──
        record: dict[str, Any] = {
            "id"                      : sample_id,
            "exhibition_id"           : sample.get("exhibition_id", ""),
            "exhibition_theme"        : sample.get("exhibition_theme", ""),
            "context"                 : sample.get("context", []),
            "candidates"              : candidates,
            "gold_id"                 : gold_id,
            "hard_negative"           : True,
            "avg_candidate_similarity": round(avg_hard_sim, 6),
        }
        hard_records.append(record)

    print(f"  [完成] 生成 {len(hard_records):,} 条，跳过 {skipped} 条")
    _print_sim_stats(sim_hard_per_sample, sim_easy_per_sample)
    _print_culture_stats(culture_counter)
    return hard_records


def _print_sim_stats(sim_hard: list[float], sim_easy: list[float]) -> None:
    """打印硬负例 vs 随机负例的相似度对比统计。"""
    print("\n  ── SBERT 候选相似度对比（候选与 gold 的余弦相似度）──")
    if sim_easy:
        arr = np.array(sim_easy)
        print(f"  MEIP-Easy  随机负例  均值={arr.mean():.4f}  "
              f"std={arr.std():.4f}  [min={arr.min():.4f}, max={arr.max():.4f}]")
    if sim_hard:
        arr = np.array(sim_hard)
        print(f"  MEIP-Hard  SBERT硬例  均值={arr.mean():.4f}  "
              f"std={arr.std():.4f}  [min={arr.min():.4f}, max={arr.max():.4f}]")
    if sim_hard and sim_easy:
        delta = np.mean(sim_hard) - np.mean(sim_easy)
        verdict = "[OK] 硬负例确实更相似 gold（难度更高）" if delta > 0 else "[WARN] 差值为负，请检查"
        print(f"  Delta (Hard - Easy) = {delta:+.4f}   {verdict}")


def _print_culture_stats(counter: Counter) -> None:
    """打印各文化分组样本数。"""
    print("\n  ── 各文化分组样本数（gold 展品文化，top-20）──")
    total = sum(counter.values())
    max_cnt = max(counter.values()) if counter else 1
    for culture, cnt in counter.most_common(20):
        bar_len = int(cnt / max_cnt * 30)
        bar = "█" * bar_len
        pct = cnt / total * 100
        print(f"  {culture:<35s}  {cnt:4d} ({pct:5.1f}%)  {bar}")
    if len(counter) > 20:
        rest_cnt = sum(v for _, v in counter.most_common()[20:])
        print(f"  {'（其他 ' + str(len(counter) - 20) + ' 个文化）':<35s}  {rest_cnt:4d}")
    print(f"  {'合计':<35s}  {total:4d}")


# ============================================================
# Step 4 — 生成 MEIP-Open
# ============================================================

def build_open_samples(meip_easy: list[dict], pool_size: int) -> list[dict]:
    """
    MEIP-Open：移除 candidates 字段，添加 pool_size 字段。
    模型需从 pool_size 件展品中直接输出一个展品 ID。
    """
    print(f"\n[4/5] 生成 MEIP-Open（移除候选列表，pool_size={pool_size:,}）...")
    open_records: list[dict] = []
    for s in meip_easy:
        rec: dict[str, Any] = {
            "id"              : s["id"],
            "exhibition_id"   : s.get("exhibition_id", ""),
            "exhibition_theme": s.get("exhibition_theme", ""),
            "context"         : s.get("context", []),
            # candidates 字段故意省略，强制开放检索
            "gold_id"         : s["gold_id"],
            "pool_size"       : pool_size,
        }
        open_records.append(rec)
    print(f"  [完成] 生成 {len(open_records):,} 条")
    return open_records


# ============================================================
# Step 5 — 写出文件 & 汇总
# ============================================================

def save_and_summarize(
    hard_records: list[dict],
    open_records: list[dict],
) -> None:
    print("\n[5/5] 写出结果文件 ...")
    write_jsonl(hard_records, OUT_HARD_PATH)
    write_jsonl(open_records, OUT_OPEN_PATH)

    print("\n" + "=" * 60)
    print("  生成汇总")
    print("=" * 60)
    hard_ok  = sum(1 for r in hard_records if r.get("hard_negative"))
    hard_avg = np.mean([r["avg_candidate_similarity"] for r in hard_records
                        if r.get("avg_candidate_similarity") is not None])
    print(f"  MEIP-Hard  : {len(hard_records):,} 条  "
          f"（含硬负例: {hard_ok:,}）  "
          f"全局平均候选相似度: {hard_avg:.4f}")
    print(f"  MEIP-Open  : {len(open_records):,} 条  "
          f"pool_size={open_records[0]['pool_size']:,}")
    print(f"\n  输出路径   : {DATA_DIR}")
    print(f"    {OUT_HARD_PATH.name}")
    print(f"    {OUT_OPEN_PATH.name}")
    print("\n  完成！\n")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("  ExhibitionBench  —  MEIP-Hard / MEIP-Open 生成器")
    print("=" * 60)

    # 1. 加载
    objects, obj_by_id, all_ids, meip_easy = load_data()

    # 2. 编码
    corpus_vecs = encode_corpus(objects)

    # 3. 生成 MEIP-Hard
    hard_records = build_hard_samples(meip_easy, obj_by_id, all_ids, corpus_vecs)

    # 4. 生成 MEIP-Open
    open_records = build_open_samples(meip_easy, pool_size=len(objects))

    # 5. 写出 & 汇总
    save_and_summarize(hard_records, open_records)


if __name__ == "__main__":
    main()
