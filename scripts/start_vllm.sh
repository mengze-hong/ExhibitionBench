#!/usr/bin/env bash
# scripts/start_vllm.sh — One-click vLLM service manager for ExhibitionBench
# ===========================================================================
# Usage:
#   bash scripts/start_vllm.sh start <model_id> [tensor_parallel_size]
#   bash scripts/start_vllm.sh stop
#   bash scripts/start_vllm.sh status
#
# Examples:
#   bash scripts/start_vllm.sh start meta-llama/Llama-3.1-8B-Instruct
#   bash scripts/start_vllm.sh start meta-llama/Llama-3.3-70B-Instruct 4
#   bash scripts/start_vllm.sh start Qwen/Qwen2.5-72B-Instruct 4
#   bash scripts/start_vllm.sh start Qwen/Qwen3-8B
#   bash scripts/start_vllm.sh stop

set -euo pipefail

VLLM_PORT=8000
VLLM_MAX_MODEL_LEN=4096
VLLM_DTYPE=bfloat16
VLLM_PID_FILE="/tmp/vllm_server.pid"
LOG_DIR="$(dirname "$0")/../logs"
mkdir -p "$LOG_DIR"

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

stop_vllm() {
    if [ -f "$VLLM_PID_FILE" ]; then
        OLD_PID=$(cat "$VLLM_PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            log "Stopping existing vLLM process (PID=$OLD_PID)..."
            kill "$OLD_PID" 2>/dev/null || true
            # Wait up to 30s for process to exit
            for i in $(seq 1 30); do
                if ! kill -0 "$OLD_PID" 2>/dev/null; then
                    log "vLLM process stopped."
                    break
                fi
                sleep 1
            done
            # Force kill if still running
            if kill -0 "$OLD_PID" 2>/dev/null; then
                log "Force killing vLLM process..."
                kill -9 "$OLD_PID" 2>/dev/null || true
            fi
        else
            log "No running vLLM process found (stale PID file)."
        fi
        rm -f "$VLLM_PID_FILE"
    else
        # Also try to kill any vllm api_server processes
        pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && log "Killed vLLM api_server processes." || log "No vLLM processes found."
    fi
}

wait_for_service() {
    local port=$1
    local max_wait=120  # seconds
    log "Waiting for vLLM service on port $port (max ${max_wait}s)..."
    for i in $(seq 1 $max_wait); do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            log "✓ vLLM service is ready on port $port"
            return 0
        fi
        sleep 1
    done
    log "✗ vLLM service did not start within ${max_wait}s. Check logs."
    return 1
}

# ── auto tensor-parallel selection ───────────────────────────────────────────

auto_tp_size() {
    local model_id=$1
    # 70B / 72B models need 4 GPUs
    if echo "$model_id" | grep -qiE "70[Bb]|72[Bb]"; then
        echo 4
    else
        echo 1
    fi
}

# ── main ─────────────────────────────────────────────────────────────────────

CMD="${1:-help}"

case "$CMD" in
    start)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 start <model_id> [tensor_parallel_size]"
            exit 1
        fi
        MODEL_ID="$2"
        TP_SIZE="${3:-$(auto_tp_size "$MODEL_ID")}"
        LOG_FILE="$LOG_DIR/vllm_server_$(echo "$MODEL_ID" | tr '/' '_').log"

        log "=== Starting vLLM server ==="
        log "Model:           $MODEL_ID"
        log "Tensor parallel: $TP_SIZE"
        log "Port:            $VLLM_PORT"
        log "Max model len:   $VLLM_MAX_MODEL_LEN"
        log "Log file:        $LOG_FILE"

        # Stop any existing vLLM service first
        stop_vllm

        # Start new vLLM service in background
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_ID" \
            --tensor-parallel-size "$TP_SIZE" \
            --port "$VLLM_PORT" \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            --dtype "$VLLM_DTYPE" \
            --trust-remote-code \
            --enforce-eager \
            --guided-decoding-backend lm-format-enforcer \
            > "$LOG_FILE" 2>&1 &

        VLLM_PID=$!
        echo "$VLLM_PID" > "$VLLM_PID_FILE"
        log "vLLM server started with PID=$VLLM_PID"
        log "Tail logs: tail -f $LOG_FILE"

        # Wait for service to be ready
        wait_for_service "$VLLM_PORT"
        ;;

    stop)
        stop_vllm
        ;;

    status)
        if [ -f "$VLLM_PID_FILE" ]; then
            PID=$(cat "$VLLM_PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                log "vLLM is running (PID=$PID)"
                curl -s "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(service not responding yet)"
            else
                log "vLLM PID file exists but process is not running."
            fi
        else
            log "vLLM is not running (no PID file)."
        fi
        ;;

    help|*)
        echo "Usage:"
        echo "  $0 start <model_id> [tensor_parallel_size]"
        echo "  $0 stop"
        echo "  $0 status"
        echo ""
        echo "Examples:"
        echo "  $0 start meta-llama/Llama-3.1-8B-Instruct"
        echo "  $0 start meta-llama/Llama-3.3-70B-Instruct 4"
        echo "  $0 start Qwen/Qwen2.5-72B-Instruct 4"
        echo "  $0 start Qwen/Qwen3-8B"
        ;;
esac
