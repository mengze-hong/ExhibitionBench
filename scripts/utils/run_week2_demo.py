"""
run_week2_demo.py
=================
Week 2 可行性验证一键脚本。

前提：
  1. 已运行 collect_exhibitions.py（data/exhibitions.jsonl 存在）
  2. 已设置 OPENAI_API_KEY 环境变量

执行流程：
  Step 1: 构建 TES 样本 (data/tes_samples.jsonl)
  Step 2: 构建 MEIP 样本 (data/meip_samples.jsonl)
  Step 3: 运行 GPT-4o 零样本 MEIP baseline（前 100 个样本）
  Step 4: 计算并打印 MEIP 指标
  Step 5: 里程碑判断

使用方法：
  export OPENAI_API_KEY=sk-...
  python run_week2_demo.py
  python run_week2_demo.py --max-samples 50  # 快速测试
"""

from __future__ import annotations
import argparse
import logging
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from benchmark.tes_eval import build_tes_samples, evaluate_tes
from benchmark.meip_eval import build_meip_samples, evaluate_meip
from baselines.gpt4o_zeroshot import (
    run_meip_zero_shot, run_tes_zero_shot,
    INTERNAL_API_BASE, INTERNAL_API_KEY, DEFAULT_MODEL, FAST_MODEL,
)

import json
import time
from openai import OpenAI
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=200, help="MEIP 可行性验证样本数")
    parser.add_argument("--skip-build", action="store_true", help="跳过样本构建（已有样本时）")
    parser.add_argument("--skip-inference", action="store_true", help="跳过模型调用（已有预测时）")
    parser.add_argument("--model", default=FAST_MODEL, help=f"验证用模型（默认: {FAST_MODEL}）")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", INTERNAL_API_KEY)
    api_base = os.environ.get("OPENAI_API_BASE", INTERNAL_API_BASE)

    if not args.skip_inference:
        pass  # api_key/base 已内置，无需检查

    # ── Step 1 & 2: 构建样本 ──────────────────────────────────────────────
    exh_path = BASE / "data" / "exhibitions.jsonl"
    obj_path = BASE / "data" / "objects.jsonl"

    if not exh_path.exists():
        log.error(f"找不到 {exh_path}，请先运行: python collect_exhibitions.py --source met")
        sys.exit(1)

    tes_path = BASE / "data" / "tes_samples.jsonl"
    meip_path = BASE / "data" / "meip_samples.jsonl"

    if not args.skip_build:
        log.info("Step 1: 构建 TES 样本...")
        n_tes = build_tes_samples(exh_path, obj_path, tes_path)
        log.info(f"  → {n_tes} 个 TES 样本")

        log.info("Step 2: 构建 MEIP 样本...")
        n_meip = build_meip_samples(exh_path, obj_path, meip_path)
        log.info(f"  → {n_meip} 个 MEIP 样本")
    else:
        log.info("跳过样本构建")

    # ── Step 3: GPT-4o 零样本 MEIP ────────────────────────────────────────
    meip_pred_path = BASE / "results" / "gpt4o_zeroshot_meip_pred.jsonl"
    meip_pred_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_inference:
        log.info(f"Step 3: GPT-4o 零样本 MEIP（前 {args.max_samples} 个样本，模型: {args.model}）...")
        client = OpenAI(api_key=api_key, base_url=api_base)

        samples = []
        with open(meip_path, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        samples = samples[:args.max_samples]

        done_ids = set()
        if meip_pred_path.exists():
            with open(meip_pred_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass

        with open(meip_pred_path, "a", encoding="utf-8") as out:
            for sample in tqdm(samples, desc="MEIP zero-shot"):
                if sample["id"] in done_ids:
                    continue
                result = run_meip_zero_shot(client, sample, model=args.model)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                time.sleep(0.5)
    else:
        log.info("跳过 GPT-4o 推理")

    # ── Step 4: 计算指标 ──────────────────────────────────────────────────
    if meip_pred_path.exists():
        log.info("Step 4: 计算 MEIP 指标...")
        results = evaluate_meip(meip_path, meip_pred_path)

        print("\n" + "="*50)
        print("  Week 2 可行性验证结果")
        print("="*50)
        for metric, val in sorted(results.items()):
            print(f"  {metric:10s}: {val:.4f}")
        print("="*50)

        # ── Step 5: 里程碑判断 ───────────────────────────────────────────
        acc1 = results.get("Acc@1", 0.0)
        print(f"\n里程碑判断（随机基线 Acc@1 = 0.100）:")
        if acc1 > 0.30:
            print(f"  ✅ Acc@1 = {acc1:.3f} > 0.30 → 任务可行！继续推进全套 baseline。")
        elif acc1 > 0.15:
            print(f"  ⚠️  Acc@1 = {acc1:.3f}，高于随机但未达目标。考虑：")
            print(f"     - 减少候选数量（10选1 → 5选1）")
            print(f"     - 增加上下文展品数量")
            print(f"     - 检查数据质量（展品描述是否太短）")
        else:
            print(f"  ❌ Acc@1 = {acc1:.3f}，接近随机。请检查：")
            print(f"     - 数据质量（展品描述是否有意义）")
            print(f"     - 任务设计（上下文是否足够）")
            print(f"     - API 调用是否正常（error 比例）")
    else:
        log.warning(f"预测文件不存在: {meip_pred_path}")


if __name__ == "__main__":
    main()
