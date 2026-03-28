#!/usr/bin/env bash
# ============================================================================
# EAGLE3 draft head training for Llama-3.2-3B-Instruct (pure JAX on TPU)
#
# Hardware: TPU v6e-4 (4 chips, 128 GB HBM total)
# Dataset:  54K mixed (ShareGPT/UltraChat/PerfectBlend)
# TTT:      7 (full test-time training rollout)
#
# Artifacts (on GCS — survives preemption):
#   Checkpoints : /path/to/checkpoints/llama32-3b-eagle3/
#   XLA cache   : /tmp/xla_cache_jax/
#   Training log: training/logs/llama32_3b_<timestamp>.log
#
# Usage:
#   bash scripts/run/run_exp_llama32_3b.sh [--dry-run]
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/path/to/venv

# ── Secrets ──────────────────────────────────────────────────────────────────
SECRETS="~/.specjax.env"
if [[ -f "$SECRETS" ]]; then
    set -a; source "$SECRETS"; set +a
    echo "[ok] Loaded secrets"
fi

# ── Paths ────────────────────────────────────────────────────────────────────
TARGET_MODEL="/path/to/models/Llama-3.2-3B-Instruct"
DATA_PATH="$REPO_ROOT/training/data/mixed_54k.jsonl"
OUTPUT_DIR="/path/to/checkpoints/llama32-3b-eagle3"
LOG_DIR="$REPO_ROOT/training/logs"

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE=8
GRAD_ACCUM=4
NUM_EPOCHS=3
LEARNING_RATE="8e-4"
MAX_LENGTH=1024
WARMUP_RATIO="0.03"
LOG_EVERY=10
SAVE_EVERY=100
TTT_LENGTH=7
EXP_NAME="llama32-3b-eagle3"
WANDB_PROJECT="llama32-3b-eagle3-experiments"

# ── Parse args ───────────────────────────────────────────────────────────────
DRY_RUN=false
MAX_STEPS_ARG=""
RESUME_ARG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            MAX_STEPS_ARG="--max-steps 5"
            ;;
        --resume)
            # Find latest checkpoint in output dir
            LATEST_CKPT=$(ls -dt "$OUTPUT_DIR"/step_* "$OUTPUT_DIR"/epoch_* 2>/dev/null | head -1)
            if [[ -n "$LATEST_CKPT" ]] && [[ -d "$LATEST_CKPT" ]]; then
                RESUME_ARG="--draft-init-path $LATEST_CKPT"
                echo "Resuming from checkpoint: $LATEST_CKPT"
            else
                echo "No checkpoint found to resume from — starting fresh"
            fi
            ;;
    esac
done

# ── Activate venv ────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV"
    echo "Run: bash scripts/setup/setup_jax_env.sh"
    exit 1
fi
source "$VENV/bin/activate"

# ── Pre-flight checks ───────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="

python3 -c "import jax; print(f'JAX devices: {jax.devices()}')" || {
    echo "ERROR: JAX cannot see TPU devices"
    exit 1
}

[[ -f "$DATA_PATH" ]] || { echo "ERROR: Dataset not found: $DATA_PATH"; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: Target model not found: $TARGET_MODEL"; exit 1; }

echo "Data:         $DATA_PATH  ($(wc -l < "$DATA_PATH") lines)"
echo "Target model: $TARGET_MODEL"
echo "Output:       $OUTPUT_DIR"
echo "TTT length:   $TTT_LENGTH"
echo "W&B project:  $WANDB_PROJECT"
echo "Dry run:      $DRY_RUN"
echo ""

# ── Create dirs ──────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ── Timestamp for log file ───────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/llama32_3b_${TIMESTAMP}.log"

echo "=== Starting Llama-3.2-3B Eagle3 Training ==="
echo "Log: $LOG_FILE"
echo ""
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Epochs: $NUM_EPOCHS  LR: $LEARNING_RATE  Max length: $MAX_LENGTH  TTT: $TTT_LENGTH"
echo ""

# ── Run training ─────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

python3 -m specjax.train \
    --target-model-path "$TARGET_MODEL"  \
    --target-model-type llama            \
    ${RESUME_ARG}                        \
    --data-path         "$DATA_PATH"     \
    --output-dir        "$OUTPUT_DIR"    \
    --exp-name          "$EXP_NAME"      \
    --num-epochs        "$NUM_EPOCHS"    \
    --learning-rate     "$LEARNING_RATE" \
    --max-length        "$MAX_LENGTH"    \
    --warmup-ratio      "$WARMUP_RATIO"  \
    --batch-size        "$BATCH_SIZE"    \
    --grad-accum-steps  "$GRAD_ACCUM"    \
    --log-every         "$LOG_EVERY"     \
    --save-every        "$SAVE_EVERY"    \
    --ttt-length        "$TTT_LENGTH"    \
    --wandb-project     "$WANDB_PROJECT" \
    ${MAX_STEPS_ARG}                     \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Training complete ==="
echo "Checkpoints at: $OUTPUT_DIR"
echo "Log at:         $LOG_FILE"
