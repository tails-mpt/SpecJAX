#!/usr/bin/env bash
# ============================================================================
# EAGLE3 draft head training for MiniMax-M2.5 (pure JAX on TPU)
#
# Hardware: TPU v6e-64 (64 chips, 2048 GB HBM total, TP=16, DP=4)
# Model:   229B MoE (256 experts, top-8, 10B active), FP8 block-wise
# Dataset: 54K mixed (ShareGPT + UltraChat + PerfectBlend)
#
# Memory budget (TP=16): ~14.3 GB/chip weights + ~3 GB activations = ~17 GB
#   → 15 GB headroom on 32 GB v6e chips
#
# Usage:
#   bash scripts/run/run_exp_minimax_m2.sh [--dry-run]
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/path/to/venv

# ── Paths ────────────────────────────────────────────────────────────────────
TARGET_MODEL="/path/to/models/MiniMax-M2.5"
DATA_PATH="$REPO_ROOT/training/data/sharegpt.jsonl"
OUTPUT_DIR="/path/to/checkpoints/minimax-m2-eagle3"
LOG_DIR="$REPO_ROOT/training/logs"

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE=1
GRAD_ACCUM=8
NUM_EPOCHS=3
LEARNING_RATE="3e-4"
MAX_LENGTH=512
WARMUP_RATIO="0.03"
LOG_EVERY=10
SAVE_EVERY=100
EXP_NAME="minimax-m2-eagle3"
TTT_LENGTH=1
TP=16  # TP=16 required for 229B MoE model

# ── Parse args ───────────────────────────────────────────────────────────────
DRY_RUN=false
MAX_STEPS_ARG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            MAX_STEPS_ARG="--max-steps 5"
            ;;
    esac
done

# ── Activate venv ────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV"
    echo "Create it with:"
    echo "  python3 -m venv $VENV"
    echo "  source $VENV/bin/activate"
    echo "  pip install -e $REPO_ROOT"
    exit 1
fi
source "$VENV/bin/activate"

# ── Pre-flight checks ───────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="

python3 -c "import jax; d=jax.devices(); print(f'JAX devices: {len(d)} ({d[0].platform})')" || {
    echo "ERROR: JAX cannot see TPU devices"
    exit 1
}

DEVICE_COUNT=$(python3 -c "import jax; print(jax.device_count())")
if [[ "$DEVICE_COUNT" -lt "$TP" ]]; then
    echo "ERROR: Need at least $TP devices for TP=$TP, but only $DEVICE_COUNT available"
    exit 1
fi

[[ -f "$DATA_PATH" ]] || { echo "ERROR: Dataset not found: $DATA_PATH"; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: Target model not found: $TARGET_MODEL"; exit 1; }

echo "Data:         $DATA_PATH  ($(wc -l < "$DATA_PATH") lines)"
echo "Target model: $TARGET_MODEL"
echo "Output:       $OUTPUT_DIR"
echo "TP:           $TP  (DP=$((DEVICE_COUNT / TP)))"
echo "Dry run:      $DRY_RUN"
echo ""

# ── Create dirs ──────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ── Timestamp for log file ───────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/minimax_m2_${TIMESTAMP}.log"

echo "=== Starting MiniMax-M2.5 EAGLE3 training ==="
echo "Log: $LOG_FILE"
echo ""
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * (DEVICE_COUNT / TP)))"
echo "Epochs: $NUM_EPOCHS  LR: $LEARNING_RATE  Max length: $MAX_LENGTH  TTT: $TTT_LENGTH"
echo ""

# ── Run training ─────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

python3 -m specjax.train \
    --target-model-path "$TARGET_MODEL"    \
    --target-model-type minimax_m2         \
    --data-path         "$DATA_PATH"       \
    --output-dir        "$OUTPUT_DIR"      \
    --exp-name          "$EXP_NAME"        \
    --num-epochs        "$NUM_EPOCHS"      \
    --learning-rate     "$LEARNING_RATE"   \
    --max-length        "$MAX_LENGTH"      \
    --warmup-ratio      "$WARMUP_RATIO"    \
    --batch-size        "$BATCH_SIZE"      \
    --grad-accum-steps  "$GRAD_ACCUM"      \
    --log-every         "$LOG_EVERY"       \
    --save-every        "$SAVE_EVERY"      \
    --ttt-length        "$TTT_LENGTH"      \
    --tp                "$TP"              \
    ${MAX_STEPS_ARG}                       \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Training complete ==="
echo "Checkpoints at: $OUTPUT_DIR"
echo "Log at:         $LOG_FILE"
