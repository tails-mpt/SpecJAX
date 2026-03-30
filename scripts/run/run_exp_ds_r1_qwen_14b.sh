#!/usr/bin/env bash
# ============================================================================
# DeepSeek-R1-Distill-Qwen-14B Eagle3 training (pure JAX on TPU)
#
# Target: DeepSeek-R1-Distill-Qwen-14B (Qwen2 architecture, 14B dense)
# Hardware: TPU v4-32 multi-host (32 chips, 1024 GB HBM, TP=4 DP=8)
# Dataset: 54K mixed (ShareGPT + UltraChat + PerfectBlend)
# Mode: TTT-7, LR 8e-4, 3 epochs
#
# Usage:
#   bash scripts/run/run_exp_ds_r1_qwen_14b.sh [--dry-run]
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/path/to/venv

# ── Multi-host TPU env (v4-32: 8 processes × 4 chips, set by launcher) ──────
export PJRT_DEVICE=TPU
export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR
export TOKENIZERS_PARALLELISM=false
export XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache_jax

# ── Load secrets ─────────────────────────────────────────────────────────────
SECRETS="~/.specjax.env"
if [[ -f "$SECRETS" ]]; then
    set -a; source "$SECRETS"; set +a
fi

# ── Paths ────────────────────────────────────────────────────────────────────
TARGET_MODEL="/path/to/shared-storage/deepseek/models/DeepSeek-R1-Distill-Qwen-14B"
TARGET_MODEL_TYPE="qwen2"
DRAFT_INIT=""            # Random init from scratch
DATA_PATH="$REPO_ROOT/training/data/mixed_54k.jsonl"
OUTPUT_DIR="/path/to/checkpoints/ds-r1-qwen-14b-eagle3"
LOG_DIR="/path/to/workspace/logs"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"

# ── Hyperparameters ──────────────────────────────────────────────────────────
# v4-32: 32 GB/chip, 21 GB headroom, DP=8 — B=2 fits comfortably
BATCH_SIZE=2
GRAD_ACCUM=8       # effective batch = 2 x 8 = 16 per DP rank
NUM_EPOCHS=3
LEARNING_RATE="8e-4"
MAX_LENGTH=512
WARMUP_RATIO="0.03"
TTT_LENGTH=7
LOG_EVERY=10
SAVE_EVERY=200
EXP_NAME="ds-r1-qwen-14b-eagle3"
WANDB_PROJECT="ds-r1-qwen-14b-eagle3-experiments"

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
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV — run setup_jax_env.sh first"
    exit 1
fi
source "$VENV/bin/activate"

# ── Pre-flight checks ───────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="
python3 -c "import jax; print(f'JAX devices: {jax.device_count()} x {jax.devices()[0].platform}')" || {
    echo "ERROR: JAX cannot see TPU devices"; exit 1
}

# Fall back to sharegpt.jsonl if mixed_54k doesn't exist yet
if [[ ! -f "$DATA_PATH" ]]; then
    ALT_DATA="$REPO_ROOT/training/data/sharegpt.jsonl"
    if [[ -f "$ALT_DATA" ]]; then
        echo "WARN: $DATA_PATH not found, using $ALT_DATA"
        DATA_PATH="$ALT_DATA"
    else
        echo "ERROR: Dataset not found: $DATA_PATH"
        echo "       Run: bash scripts/setup/prep_dataset_jax.sh"
        exit 1
    fi
fi

[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: Target model not found: $TARGET_MODEL"; exit 1; }

echo "Data:           $DATA_PATH  ($(wc -l < "$DATA_PATH") samples)"
echo "Target model:   $TARGET_MODEL (type=$TARGET_MODEL_TYPE)"
echo "Draft init:     ${DRAFT_INIT:-'(random init from scratch)'}"
echo "Output:         $OUTPUT_DIR"
echo "Max length:     $MAX_LENGTH"
echo "LR / Warmup:    $LEARNING_RATE / $WARMUP_RATIO"
echo "Epochs:         $NUM_EPOCHS"
echo "TTT length:     $TTT_LENGTH"
echo "Eff batch:      $((BATCH_SIZE * GRAD_ACCUM))  ($BATCH_SIZE micro x $GRAD_ACCUM accum)"
echo "W&B project:    $WANDB_PROJECT"
echo "Dry run:        $DRY_RUN"
echo ""

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/exp_ds_r1_qwen_14b_${TIMESTAMP}.log"

echo "=== Starting DS-R1-Qwen-14B Eagle3 training ==="
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
