#!/usr/bin/env bash
# gpu_server_needed/run.sh — One-click ExhibitionBench open-weight evaluation
# ===========================================================================
# Usage:
#   bash gpu_server_needed/run.sh          # full eval (all P0 models)
#   bash gpu_server_needed/run.sh smoke    # smoke test (50 samples per task)
#
# Prerequisites:
#   1. pip install -r gpu_server_needed/requirements_gpu.txt
#   2. Upload data/ directory (see README.md for required files)
#   3. export HF_TOKEN=<your_huggingface_token>  (needed for Llama models)

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-full}"
LOG_DIR="$PROJ_DIR/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/run_wrapper.log"; }

log "================================================================"
log " ExhibitionBench GPU Eval — mode=$MODE"
log " Project: $PROJ_DIR"
log "================================================================"

# ── Pre-flight checks ─────────────────────────────────────────────────────────

ERRORS=0

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    log "ERROR: python not found. Please install Python 3.10+."
    ERRORS=$((ERRORS + 1))
fi
PYTHON=$(command -v python3 2>/dev/null || command -v python)

# Check vLLM
if ! $PYTHON -c "import vllm" 2>/dev/null; then
    log "ERROR: vllm not installed. Run: pip install -r gpu_server_needed/requirements_gpu.txt"
    ERRORS=$((ERRORS + 1))
fi

# Check HF_TOKEN (only warn, not hard fail — Qwen models don't need it)
if [ -z "${HF_TOKEN:-}" ]; then
    log "WARN: HF_TOKEN not set. Llama models will be skipped."
    log "      To enable Llama models: export HF_TOKEN=<your_hf_token>"
fi

# Check required data files
REQUIRED_DATA=(
    "$PROJ_DIR/data/meip_samples_v3_fixed.jsonl"
    "$PROJ_DIR/data/tes_samples_v3.jsonl"
    "$PROJ_DIR/data/ecd_samples_v3.jsonl"
)
for f in "${REQUIRED_DATA[@]}"; do
    if [ ! -f "$f" ]; then
        log "ERROR: Missing required data file: $f"
        log "       Please upload the data/ directory (see gpu_server_needed/README.md)"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ "$ERRORS" -gt 0 ]; then
    log "Pre-flight checks failed with $ERRORS error(s). Aborting."
    exit 1
fi

log "Pre-flight checks passed. Starting evaluation..."

# ── Run all open-weight models ────────────────────────────────────────────────

bash "$PROJ_DIR/scripts/run_all_openllm.sh" "$MODE"

# ── Package results ───────────────────────────────────────────────────────────

log ""
log "================================================================"
log " Packaging results..."
log "================================================================"

TARBALL="$PROJ_DIR/exhib_openllm_results_$(hostname -s)_$(date +%Y%m%d_%H%M).tar.gz"

# Only package result JSONs and relevant logs (not giant vLLM model logs)
tar -czf "$TARBALL" \
    -C "$PROJ_DIR" \
    $(ls results/*_shot0.json 2>/dev/null | sed "s|$PROJ_DIR/||" | tr '\n' ' ') \
    $(ls logs/openllm_*.log 2>/dev/null | sed "s|$PROJ_DIR/||" | tr '\n' ' ' || true) \
    logs/run_wrapper.log 2>/dev/null || true

if [ -f "$TARBALL" ]; then
    SIZE=$(du -sh "$TARBALL" | cut -f1)
    log "Results packaged: $TARBALL ($SIZE)"
    log ""
    log "To transfer results back to your local machine:"
    log "  scp $(hostname -s):$TARBALL /local/path/to/results/"
else
    log "WARN: Could not create tarball (no results found yet?)"
fi

log ""
log "================================================================"
log " All done!"
log " Results directory: $PROJ_DIR/results/"
log "================================================================"
