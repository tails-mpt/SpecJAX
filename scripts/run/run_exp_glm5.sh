#!/usr/bin/env bash
# ============================================================================
# run_exp_glm5_a.sh — First Eagle3 training experiment for GLM-5-FP8
#
# Conservative hyperparameters for initial validation:
#   - Single-step mode (TTT=1) to verify correctness first
#   - Small batch size (1) due to memory constraints
#   - Short max_length (512) to keep DSA-skip valid
#   - Baseline LR (1e-4)
#   - From-scratch random init (no pre-trained Eagle3 for GLM-5)
#
# Hardware: TPU v4e-32 (32 chips, ~1TB HBM total)
#   - TP=32 (all chips for tensor parallelism, no data parallelism)
#   - Target model in FP8 on-device (~744GB), dequantized per-layer
#
# Usage:
#   bash scripts/run/run_exp_glm5.sh [--dry-run]
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
TARGET_MODEL="/path/to/models/GLM-5-FP8"
DRAFT_INIT=""  # Empty = random init (from scratch, no existing Eagle3 for GLM-5)
DATA_PATH="$REPO_ROOT/training/data/mixed_54k.jsonl"
OUTPUT_DIR="/path/to/checkpoints/exp-glm5-a"
LOG_DIR="/path/to/workspace/logs"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"

# ── Hyperparameters (conservative first experiment) ──────────────────────────
BATCH_SIZE=1
GRAD_ACCUM=16          # effective batch size = 1 x 16 = 16
NUM_EPOCHS=3
LEARNING_RATE="1e-4"   # Baseline LR
MAX_LENGTH=512         # Keep short (DSA skip valid for <= 2048)
WARMUP_RATIO="0.03"
TTT_LENGTH=1           # Single-step first (verify correctness before TTT)
LOG_EVERY=10
SAVE_EVERY=500
EXP_NAME="exp-glm5-a"
WANDB_PROJECT="glm4-large-eagle3-experiments"
TP_SIZE=32             # All 32 chips for tensor parallelism

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
    echo "ERROR: venv not found at $VENV — run setup_glm5_env.sh first"
    exit 1
fi
source "$VENV/bin/activate"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')" || {
    echo "ERROR: JAX cannot see TPU devices"; exit 1
}

# Use fallback data path if mixed_54k doesn't exist
if [[ ! -f "$DATA_PATH" ]]; then
    ALT_DATA="$REPO_ROOT/training/data/sharegpt.jsonl"
    if [[ -f "$ALT_DATA" ]]; then
        echo "NOTE: mixed_54k.jsonl not found, using sharegpt.jsonl"
        DATA_PATH="$ALT_DATA"
    else
        echo "ERROR: Dataset not found: $DATA_PATH"
        echo "       Run: bash scripts/setup/prep_dataset_jax.sh"
        exit 1
    fi
fi
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: Target model not found: $TARGET_MODEL"; exit 1; }

echo "Data:           $DATA_PATH  ($(wc -l < "$DATA_PATH") samples)"
echo "Target model:   $TARGET_MODEL (GLM-5-FP8)"
echo "Draft init:     ${DRAFT_INIT:-'(random init from scratch)'}"
echo "Output:         $OUTPUT_DIR"
echo "Max length:     $MAX_LENGTH"
echo "LR / Warmup:    $LEARNING_RATE / $WARMUP_RATIO"
echo "Epochs:         $NUM_EPOCHS"
echo "TTT length:     $TTT_LENGTH"
echo "TP size:        $TP_SIZE"
echo "Eff batch:      $((BATCH_SIZE * GRAD_ACCUM))  ($BATCH_SIZE per-chip x $GRAD_ACCUM accum)"
echo "W&B project:    $WANDB_PROJECT"
echo "Dry run:        $DRY_RUN"
echo ""
echo "GLM-5-FP8 specific:"
echo "  - FP8 weights dequantized to bf16 per-layer during forward pass"
echo "  - DSA indexer skipped (max_length=$MAX_LENGTH <= topk=2048)"
echo "  - Eagle3 hidden_size=6144 (matching GLM-5 target)"
echo "  - Aux layers: {1, 38, 74} (SpecForge convention for 78 layers)"
echo ""

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/exp_glm5_a_${TIMESTAMP}.log"

echo "=== Starting Exp GLM5-A (single-step, from scratch) ==="
echo "Log: $LOG_FILE"
echo ""

cd "$REPO_ROOT"

DRAFT_INIT_ARG=""
if [[ -n "$DRAFT_INIT" ]]; then
    DRAFT_INIT_ARG="--draft-init-path $DRAFT_INIT"
fi

python3 -m specjax.train \
    --target-model-path "$TARGET_MODEL"   \
    ${DRAFT_INIT_ARG}                     \
    --data-path         "$DATA_PATH"      \
    --output-dir        "$OUTPUT_DIR"     \
    --exp-name          "$EXP_NAME"       \
    --num-epochs        "$NUM_EPOCHS"     \
    --learning-rate     "$LEARNING_RATE"  \
    --max-length        "$MAX_LENGTH"     \
    --warmup-ratio      "$WARMUP_RATIO"   \
    --batch-size        "$BATCH_SIZE"     \
    --grad-accum-steps  "$GRAD_ACCUM"     \
    --ttt-length        "$TTT_LENGTH"     \
    --log-every         "$LOG_EVERY"      \
    --save-every        "$SAVE_EVERY"     \
    --wandb-project     "$WANDB_PROJECT"  \
    --wandb-run-name    "$EXP_NAME"       \
    --target-model-type glm5              \
    --tp-size           "$TP_SIZE"        \
    ${MAX_STEPS_ARG}                      \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Training complete ==="
echo "Checkpoints at: $OUTPUT_DIR"
echo "Log at:         $LOG_FILE"
