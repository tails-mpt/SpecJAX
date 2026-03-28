#!/usr/bin/env bash
# ============================================================================
# run_all_pending_sequential.sh — overnight training pipeline
#
# Trains the 4 pending models in order, one at a time on this v4-32:
#   1. Llama-3.1-8B-Instruct   (~6h)
#   2. Qwen2.5-14B-Instruct    (~3h)
#   3. Qwen3-8B                (~6h)
#   4. Qwen3-14B               (~3h)
#
# If SKIP_LLAMA=1 is set, assumes Llama is already running and starts from
# Qwen2.5-14B (used when Llama was launched separately).
#
# Usage:
#   nohup bash scripts/run/run_all_pending_sequential.sh SKIP_LLAMA=1 \
#     > /path/to/workspace/logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ============================================================================
set -euo pipefail

REPO_ROOT="/path/to/specjax"
LOG_DIR="/path/to/workspace/logs"
SKIP_LLAMA="${SKIP_LLAMA:-0}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_model() {
    local name="$1"
    local launch_script="$2"
    local dry_run_script="$3"
    local checkpoint_dir="$4"

    log "========================================================"
    log "Starting: $name"
    log "========================================================"

    # Dry-run first
    log "[$name] Running dry-run (5 steps)..."
    bash "$launch_script" --dry-run
    log "[$name] Dry-run PASSED"

    # Clean any dry-run artifacts
    rm -rf "$checkpoint_dir/final" "$checkpoint_dir/epoch_"* 2>/dev/null || true

    # Full training
    log "[$name] Launching full training..."
    bash "$launch_script"
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        log "[$name] FAILED with exit code $exit_code — attempting resume from latest checkpoint"
        LATEST=$(ls -dt "$checkpoint_dir"/epoch_* 2>/dev/null | head -1 || true)
        if [[ -n "$LATEST" ]]; then
            log "[$name] Resuming from: $LATEST"
            bash "$launch_script" --resume="$LATEST"
        else
            log "[$name] No checkpoint to resume from — BLOCKED"
            return 1
        fi
    fi

    log "[$name] Training complete. Checkpoint: $checkpoint_dir/final"
}

# Wait for a running training job to finish (used when Llama was pre-launched)
wait_for_llama() {
    log "Waiting for Llama-3.1-8B training to finish..."
    while ps aux | grep "specjax.train" | grep -v grep | grep -q "llama"; do
        sleep 60
        W0_LOG=$(ls "$LOG_DIR"/llama31_8b_eagle3_w0_*.log 2>/dev/null | sort | tail -1)
        if [[ -n "$W0_LOG" ]]; then
            LAST=$(tail -1 "$W0_LOG" 2>/dev/null || echo "")
            log "[llama] $LAST"
        fi
    done
    # Also wait for any lingering specjax.train process
    while ps aux | grep "specjax.train" | grep -v grep | grep -q "."; do
        sleep 30
    done
    log "Llama-3.1-8B training finished."
}

mkdir -p "$LOG_DIR"

# ── Model 1: Llama-3.1-8B ────────────────────────────────────────────────────
if [[ "$SKIP_LLAMA" == "1" ]]; then
    log "SKIP_LLAMA=1 — waiting for already-running Llama training to finish"
    wait_for_llama
    log "Llama-3.1-8B done. Proceeding to next model."
else
    run_model \
        "Llama-3.1-8B-Instruct" \
        "$REPO_ROOT/scripts/run/launch_all_workers_llama31_8b.sh" \
        "$REPO_ROOT/scripts/run/launch_all_workers_llama31_8b.sh" \
        "/path/to/workspace/checkpoints/llama31-8b-eagle3"
fi

# ── Model 2: Qwen2.5-14B ─────────────────────────────────────────────────────
run_model \
    "Qwen2.5-14B-Instruct" \
    "$REPO_ROOT/scripts/run/launch_all_workers_qwen25_14b.sh" \
    "$REPO_ROOT/scripts/run/launch_all_workers_qwen25_14b.sh" \
    "/path/to/workspace/checkpoints/qwen25-14b-eagle3"

# ── Model 3: Qwen3-8B ────────────────────────────────────────────────────────
run_model \
    "Qwen3-8B" \
    "$REPO_ROOT/scripts/run/launch_all_workers_qwen3_8b.sh" \
    "$REPO_ROOT/scripts/run/launch_all_workers_qwen3_8b.sh" \
    "/path/to/workspace/checkpoints/qwen3-8b-eagle3"

# ── Model 4: Qwen3-14B ───────────────────────────────────────────────────────
run_model \
    "Qwen3-14B" \
    "$REPO_ROOT/scripts/run/launch_all_workers_qwen3_14b.sh" \
    "$REPO_ROOT/scripts/run/launch_all_workers_qwen3_14b.sh" \
    "/path/to/workspace/checkpoints/qwen3-14b-eagle3"

log "========================================================"
log "ALL MODELS COMPLETE"
log "  llama31-8b:  /path/to/workspace/checkpoints/llama31-8b-eagle3/final"
log "  qwen25-14b:  /path/to/workspace/checkpoints/qwen25-14b-eagle3/final"
log "  qwen3-8b:    /path/to/workspace/checkpoints/qwen3-8b-eagle3/final"
log "  qwen3-14b:   /path/to/workspace/checkpoints/qwen3-14b-eagle3/final"
log "========================================================"
