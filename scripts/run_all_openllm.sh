#!/usr/bin/env bash
# scripts/run_all_openllm.sh — Run all P0 open-weight models sequentially
# =========================================================================
# Usage:
#   bash scripts/run_all_openllm.sh          # full eval
#   bash scripts/run_all_openllm.sh smoke    # smoke test (50 samples each)
#
# Models (in order):
#   1. Qwen2.5-7B-Instruct  (1 GPU, ~15GB)   — smoke test / P1 baseline
#   2. Llama-3.1-8B-Instruct (1 GPU, ~16GB)  — P0 (requires HF_TOKEN)
#   3. Qwen3-8B              (1 GPU, ~16GB)   — P0 (requires transformers>=4.51)
#   4. Llama-3.3-70B-Instruct (4 GPU, ~80GB) — P0 (requires HF_TOKEN)
#   5. Qwen2.5-72B-Instruct  (4 GPU, ~80GB)  — P0

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-full}"

if [ "$MODE" = "smoke" ]; then
    MAX_SAMPLES=50
    echo "[INFO] SMOKE TEST mode: 50 samples per task"
else
    MAX_SAMPLES=""
    echo "[INFO] FULL EVAL mode"
fi

run_model() {
    local model_id="$1"
    local slug="$2"
    local tp="$3"
    echo ""
    echo "============================================================"
    echo " Starting: $model_id (slug=$slug, tp=$tp)"
    echo "============================================================"
    bash "$PROJ_DIR/scripts/run_openllm_eval.sh" "$model_id" "$slug" "$tp" "$MAX_SAMPLES"
}

# ── P1: Qwen2.5-7B (no auth needed, good smoke test model) ───────────────────
run_model "Qwen/Qwen2.5-7B-Instruct" "qwen2.5-7b" "1"

# ── P0: Llama-3.1-8B (needs HF_TOKEN) ────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    run_model "meta-llama/Llama-3.1-8B-Instruct" "llama-3.1-8b" "1"
else
    echo "[WARN] HF_TOKEN not set, skipping Llama-3.1-8B. Set HF_TOKEN=<your_token> to enable."
fi

# ── P0: Qwen2.5-72B (no auth needed) ─────────────────────────────────────────
run_model "Qwen/Qwen2.5-72B-Instruct" "qwen2.5-72b" "4"

# ── P0: Llama-3.3-70B (needs HF_TOKEN) ───────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    run_model "meta-llama/Llama-3.3-70B-Instruct" "llama-3.3-70b" "4"
else
    echo "[WARN] HF_TOKEN not set, skipping Llama-3.3-70B. Set HF_TOKEN=<your_token> to enable."
fi

# ── Final: Compile results ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Compiling results..."
echo "============================================================"
cd "$PROJ_DIR"
python3 results/compile_sota_results.py --latex

echo ""
echo "All done! Results in $PROJ_DIR/results/"
