#!/usr/bin/env bash
# ============================================================================
# EAGLE3 draft head training for Qwen2.5-7B-Instruct (pure JAX on TPU)
#
# Target: Qwen2.5-7B-Instruct (dense GQA, bfloat16, 152K vocab)
# Hardware: TPU v4-32 (32 chips, 1024 GB HBM, mesh dp=8 tp=4)
# Training: LR 8e-4, TTT=7, 3 epochs, mixed 54K dataset
#
# Qwen2.5-7B is the most downloaded model on HuggingFace (~19.6M/month)
# with zero usable EAGLE3 draft heads available.
#
# Artifacts (on GCS — survives preemption):
#   Checkpoints : /path/to/checkpoints/qwen25-7b-eagle3/
#   XLA cache   : /tmp/xla_cache_jax/
#
# Usage:
#   bash scripts/run/run_exp_qwen25_7b.sh [--dry-run]
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/path/to/venv

# ── Load secrets ─────────────────────────────────────────────────────────────
SECRETS="~/.specjax.env"
if [[ -f "$SECRETS" ]]; then
    set -a; source "$SECRETS"; set +a
fi

# ── Paths ────────────────────────────────────────────────────────────────────
TARGET_MODEL="/path/to/models/Qwen2.5-7B-Instruct"
TARGET_MODEL_TYPE="qwen25"
DRAFT_INIT=""            # Random init from scratch
DATA_PATH="$REPO_ROOT/training/data/sharegpt.jsonl"
OUTPUT_DIR="/path/to/checkpoints/qwen25-7b-eagle3"
LOG_DIR="/path/to/workspace/logs"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE=4
GRAD_ACCUM=4       # effective batch size per DP rank = 4 x 4 = 16
                    # with dp=8: global effective batch = 128
NUM_EPOCHS=3
LEARNING_RATE="8e-4"
MAX_LENGTH=1024
WARMUP_RATIO="0.03"
TTT_LENGTH=7
LOG_EVERY=10
SAVE_EVERY=100
EXP_NAME="qwen25-7b-eagle3"
WANDB_PROJECT="qwen25-7b-eagle3-experiments"

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
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV — run setup script first"
    exit 1
fi
source "$VENV/bin/activate"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')" || {
    echo "ERROR: JAX cannot see TPU devices"; exit 1
}

[[ -f "$DATA_PATH" ]] || {
    echo "ERROR: Dataset not found: $DATA_PATH"
    echo "       Run: bash scripts/setup/prep_dataset_jax.sh"
    exit 1
}
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: Target model not found: $TARGET_MODEL"; exit 1; }

echo "Data:           $DATA_PATH  ($(wc -l < "$DATA_PATH") samples)"
echo "Target model:   $TARGET_MODEL (type=$TARGET_MODEL_TYPE)"
echo "Draft init:     ${DRAFT_INIT:-'(random init from scratch)'}"
echo "Output:         $OUTPUT_DIR"
echo "Max length:     $MAX_LENGTH"
echo "LR / Warmup:    $LEARNING_RATE / $WARMUP_RATIO"
echo "Epochs:         $NUM_EPOCHS"
echo "TTT length:     $TTT_LENGTH"
echo "Eff batch:      $((BATCH_SIZE * GRAD_ACCUM)) per DP rank  ($BATCH_SIZE x $GRAD_ACCUM accum)"
echo "W&B project:    $WANDB_PROJECT"
echo "Dry run:        $DRY_RUN"
echo ""

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/qwen25_7b_eagle3_${TIMESTAMP}.log"

echo "=== Starting Qwen2.5-7B-Instruct EAGLE3 Training ==="
echo "Log: $LOG_FILE"
echo ""

cd "$REPO_ROOT"

DRAFT_INIT_ARG=""
if [[ -n "$DRAFT_INIT" ]]; then
    DRAFT_INIT_ARG="--draft-init-path $DRAFT_INIT"
fi

python3 -m specjax.train \
    --target-model-path "$TARGET_MODEL"        \
    --target-model-type "$TARGET_MODEL_TYPE"   \
    ${DRAFT_INIT_ARG}                          \
    --data-path         "$DATA_PATH"           \
    --output-dir        "$OUTPUT_DIR"          \
    --exp-name          "$EXP_NAME"            \
    --num-epochs        "$NUM_EPOCHS"          \
    --learning-rate     "$LEARNING_RATE"       \
    --max-length        "$MAX_LENGTH"          \
    --warmup-ratio      "$WARMUP_RATIO"        \
    --batch-size        "$BATCH_SIZE"          \
    --grad-accum-steps  "$GRAD_ACCUM"          \
    --ttt-length        "$TTT_LENGTH"          \
    --log-every         "$LOG_EVERY"           \
    --save-every        "$SAVE_EVERY"          \
    --wandb-project     "$WANDB_PROJECT"       \
    --wandb-run-name    "$EXP_NAME"            \
    ${MAX_STEPS_ARG}                           \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Training complete ==="
echo "Checkpoints at: $OUTPUT_DIR"
echo "Log at:         $LOG_FILE"
