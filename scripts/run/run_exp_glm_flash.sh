#!/usr/bin/env bash
# ============================================================================
# Exp JAX-A: EAGLE3 draft head training for GLM-4.7-Flash (pure JAX on TPU)
#
# Mirrors the Exp TPU-A hyperparameter configuration (run_exp_tpu_a.sh) but
# uses the native JAX implementation for apples-to-apples comparison.
#
# Hardware: TPU v4-32 (4 chips, 128 GB HBM total)
# Dataset:  ShareGPT 10 K samples (same as GPU Exp I, 81.56% acceptance)
# Expected: ≥79% acceptance rate (matching PyTorch Exp K baseline)
#
# Artifacts (on GCS — survives preemption):
#   Checkpoints : /path/to/checkpoints/exp-jax-a/
#   XLA cache   : /tmp/xla_cache_jax/
#   Training log: training/logs/exp_jax_a_<timestamp>.log
#
# Usage:
#   bash scripts/run/run_exp_glm_flash.sh [--dry-run]
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/path/to/venv

# ── Paths ────────────────────────────────────────────────────────────────────
TARGET_MODEL="/path/to/models/GLM-4.7-Flash"
DRAFT_INIT="/path/to/models/GLM-4.7-Flash-Eagle3"
DATA_PATH="$REPO_ROOT/training/data/sharegpt.jsonl"
OUTPUT_DIR="/path/to/checkpoints/exp-jax-a"
LOG_DIR="$REPO_ROOT/training/logs"

# ── Hyperparameters (match Exp TPU-A for comparison) ─────────────────────────
BATCH_SIZE=8
GRAD_ACCUM=4
NUM_EPOCHS=1          # ~37h on TPU v4-32; 3 epochs too risky on SPOT
LEARNING_RATE="1e-4"
MAX_LENGTH=512        # OOM at 1024 with B=8
WARMUP_RATIO="0.03"
LOG_EVERY=10
SAVE_EVERY=100
EXP_NAME="exp-jax-a"

# ── Parse args ────────────────────────────────────────────────────────────────
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

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV"
    echo "Create it with:"
    echo "  python3 -m venv $VENV"
    echo "  source $VENV/bin/activate"
    echo "  pip install -e $REPO_ROOT"
    exit 1
fi
source "$VENV/bin/activate"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="

python3 -c "import jax; print(f'JAX devices: {jax.devices()}')" || {
    echo "ERROR: JAX cannot see TPU devices"
    exit 1
}

[[ -f "$DATA_PATH" ]] || { echo "ERROR: Dataset not found: $DATA_PATH"; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: Target model not found: $TARGET_MODEL"; exit 1; }

echo "Data:         $DATA_PATH  ($(wc -l < "$DATA_PATH") lines)"
echo "Target model: $TARGET_MODEL"
echo "Draft init:   ${DRAFT_INIT}"
echo "Output:       $OUTPUT_DIR"
echo "Dry run:      $DRY_RUN"
echo ""

# ── Create dirs ───────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── Timestamp for log file ────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/exp_jax_a_${TIMESTAMP}.log"

echo "=== Starting Exp JAX-A training ==="
echo "Log: $LOG_FILE"
echo ""
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Epochs: $NUM_EPOCHS  LR: $LEARNING_RATE  Max length: $MAX_LENGTH"
echo ""

# ── Run training ──────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

# Only pass --draft-init-path if the directory actually contains weights
DRAFT_INIT_ARG=""
if [[ -d "$DRAFT_INIT" ]] && ls "$DRAFT_INIT"/*.safetensors &>/dev/null; then
    DRAFT_INIT_ARG="--draft-init-path $DRAFT_INIT"
else
    echo "Note: $DRAFT_INIT not found — initialising Eagle3 from scratch"
fi

python3 -m specjax.train \
    --target-model-path "$TARGET_MODEL"  \
    ${DRAFT_INIT_ARG}                    \
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
    ${MAX_STEPS_ARG}                     \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Training complete ==="
echo "Checkpoints at: $OUTPUT_DIR"
echo "Log at:         $LOG_FILE"
