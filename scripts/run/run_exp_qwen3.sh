#!/usr/bin/env bash
# ============================================================================
# run_exp_qwen3_jax_a.sh — Qwen3-Coder-Next Eagle3 training: Hypothesis 3
#                           Config A (attention layers)
#
# Tests EAGLE3 with features from ATTENTION layers only: {3, 23, 47}
# (first, middle, last attention layers in the 3:1 GDN:ATT architecture).
#
# Hypothesis: attention layer hidden states encode immediate retrieval signals
# better suited for EAGLE3 feature fusion than GDN's compressed recurrent states.
#
# Parameters:
#   - Target     : Qwen3-Coder-Next (FP8)
#   - Aux layers : 3,23,47 (all attention layers)
#   - Dataset    : 54K mixed (Qwen3 tokenizer)
#   - LR         : 1e-4
#   - Warmup     : 3%
#   - MaxLen     : 1024
#   - Epochs     : 3
#   - Mode       : TTT-7
#   - Init       : Random (from scratch)
#
# Hardware: TPU v6e-16 (512 GB HBM) with FP8 weights
#
# Usage:
#   bash scripts/run/run_exp_qwen3.sh [--dry-run]
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
TARGET_MODEL="/path/to/models/Qwen3-Coder-Next-FP8"
TARGET_MODEL_TYPE="qwen3"
AUX_LAYERS="3,23,47"    # Hypothesis 3 Config A: all attention layers
DRAFT_INIT=""            # Random init from scratch
DATA_PATH="$REPO_ROOT/training/data/mixed_54k_qwen3.jsonl"
OUTPUT_DIR="/path/to/checkpoints/exp-qwen3-jax-a"
LOG_DIR="/path/to/workspace/logs"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE=4
GRAD_ACCUM=8       # effective batch size = 4 x 8 = 32
NUM_EPOCHS=3
LEARNING_RATE="1e-4"
MAX_LENGTH=1024
WARMUP_RATIO="0.03"
TTT_LENGTH=7
LOG_EVERY=10
SAVE_EVERY=500
EXP_NAME="exp-qwen3-jax-a-attn"
WANDB_PROJECT="qwen3-eagle3-experiments"

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
    echo "ERROR: venv not found at $VENV — run setup_qwen3_jax_env.sh first"
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
echo "Aux layers:     $AUX_LAYERS  (Hypothesis 3 Config A: attention layers)"
echo "Draft init:     ${DRAFT_INIT:-'(random init from scratch)'}"
echo "Output:         $OUTPUT_DIR"
echo "Max length:     $MAX_LENGTH"
echo "LR / Warmup:    $LEARNING_RATE / $WARMUP_RATIO"
echo "Epochs:         $NUM_EPOCHS"
echo "TTT length:     $TTT_LENGTH"
echo "Eff batch:      $((BATCH_SIZE * GRAD_ACCUM))  ($BATCH_SIZE per-chip x $GRAD_ACCUM accum)"
echo "W&B project:    $WANDB_PROJECT"
echo "Dry run:        $DRY_RUN"
echo ""

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/exp_qwen3_jax_a_${TIMESTAMP}.log"

echo "=== Starting Exp Qwen3-JAX-A (attention layers {3,23,47}) ==="
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
    --aux-layers        "$AUX_LAYERS"          \
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
