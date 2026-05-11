#!/usr/bin/env bash
# scripts/run_openllm_eval.sh — Full evaluation pipeline for one open-weight model
# =================================================================================
# Usage:
#   bash scripts/run_openllm_eval.sh <model_id> <model_slug> [tensor_parallel_size] [max_samples]
#
# Examples (smoke test):
#   bash scripts/run_openllm_eval.sh Qwen/Qwen2.5-7B-Instruct qwen2.5-7b 1 50
#
# Examples (full eval):
#   bash scripts/run_openllm_eval.sh Qwen/Qwen2.5-72B-Instruct qwen2.5-72b 4
#   bash scripts/run_openllm_eval.sh meta-llama/Llama-3.3-70B-Instruct llama-3.3-70b 4
#   bash scripts/run_openllm_eval.sh meta-llama/Llama-3.1-8B-Instruct llama-3.1-8b 1
#   bash scripts/run_openllm_eval.sh Qwen/Qwen3-8B qwen3-8b 1

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJ_DIR/logs"
RESULTS_DIR="$PROJ_DIR/results"
VLLM_PORT=8000
VLLM_PID_FILE="/tmp/vllm_server.pid"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

MODEL_ID="${1:?Usage: $0 <model_id> <model_slug> [tp_size] [max_samples]}"
MODEL_SLUG="${2:?Usage: $0 <model_id> <model_slug> [tp_size] [max_samples]}"
TP_SIZE="${3:-1}"
MAX_SAMPLES="${4:-}"   # empty = full eval

DATE=$(date +%Y%m%d_%H%M%S)
EVAL_LOG="$LOG_DIR/eval_${MODEL_SLUG}_${DATE}.log"
VLLM_LOG="$LOG_DIR/vllm_${MODEL_SLUG}_${DATE}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$EVAL_LOG"; }

# ── Step 1: Stop any existing vLLM ───────────────────────────────────────────
log "=== Step 1: Stopping any existing vLLM service ==="
if [ -f "$VLLM_PID_FILE" ]; then
    OLD_PID=$(cat "$VLLM_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log "Killing old vLLM PID=$OLD_PID"
        pkill -P "$OLD_PID" 2>/dev/null || true
        # Give it a moment to clean up GPU memory
        python3 -c "import time; time.sleep(5)"
    fi
    rm -f "$VLLM_PID_FILE"
fi

# ── Step 2: Start vLLM ───────────────────────────────────────────────────────
log "=== Step 2: Starting vLLM for $MODEL_ID (TP=$TP_SIZE) ==="
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$VLLM_PORT" \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enforce-eager \
    --guided-decoding-backend lm-format-enforcer \
    >> "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "$VLLM_PID_FILE"
log "vLLM started with PID=$VLLM_PID, log: $VLLM_LOG"

# ── Step 3: Wait for service ready ───────────────────────────────────────────
log "=== Step 3: Waiting for vLLM service (max 180s) ==="
READY=0
for i in $(seq 1 180); do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        log "✓ vLLM service ready after ${i}s"
        READY=1
        break
    fi
    python3 -c "import time; time.sleep(1)"
done

if [ "$READY" -eq 0 ]; then
    log "✗ vLLM service failed to start. Check $VLLM_LOG"
    exit 1
fi

# ── Step 4: Run evaluation ────────────────────────────────────────────────────
log "=== Step 4: Running evaluation (model=$MODEL_SLUG) ==="

EXTRA_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_ARGS="--max-samples $MAX_SAMPLES --workers 8"
    log "SMOKE TEST mode: max_samples=$MAX_SAMPLES"
else
    EXTRA_ARGS="--workers 32"
    log "FULL EVAL mode"
fi

cd "$PROJ_DIR"
python3 baselines/openllm_baseline.py \
    --api-base "http://localhost:${VLLM_PORT}/v1" \
    --api-key vllm \
    --model "$MODEL_ID" \
    --tasks meip tes ecd \
    $EXTRA_ARGS \
    2>&1 | tee -a "$EVAL_LOG"

EVAL_EXIT=${PIPESTATUS[0]}

# ── Step 5: Verify results ────────────────────────────────────────────────────
log "=== Step 5: Verifying results ==="
for task in meip tes ecd; do
    RESULT_FILE="$RESULTS_DIR/${task}_${MODEL_SLUG}_shot0.json"
    if [ -f "$RESULT_FILE" ]; then
        log "✓ $RESULT_FILE exists"
    else
        log "✗ MISSING: $RESULT_FILE"
    fi
done

log "=== Evaluation complete (exit=$EVAL_EXIT) ==="
exit $EVAL_EXIT
