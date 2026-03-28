#!/usr/bin/env bash
# ============================================================================
# worker_start_qwen25_7b.sh — run on EVERY worker simultaneously.
#
# Auto-detects JAX_PROCESS_INDEX from the hostname suffix (-w-N).
# Run on all 4 workers at the same time:
#
#   for w in 0 1 2 3; do
#     ssh <tpu-name>-w-$w \
#       "bash /path/to/specjax/scripts/run/worker_start_qwen25_7b.sh" &
#   done; wait
#
# w0 (process 0) is the JAX coordinator. Workers 1-3 connect to it.
# All 4 must start within the JAX coordinator timeout (~5 minutes).
# ============================================================================

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
MAX_STEPS_ARG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            MAX_STEPS_ARG="--max-steps 5"
            ;;
    esac
done

# ── Detect worker index from hostname (e.g. t1v-n-...-w-2 → 2) ──────────────
HOSTNAME_VAL="$(hostname)"
WORKER_IDX="$(echo "$HOSTNAME_VAL" | grep -oP '(?<=-w-)\d+$' || echo '0')"
echo "[worker_start] hostname=$HOSTNAME_VAL  process_index=$WORKER_IDX"

# w0 internal IP — detected from this v4-32 slice
COORDINATOR_IP="${COORDINATOR_IP:-<coordinator-ip>}"
COORDINATOR_PORT="2222"
REPO_ROOT="/path/to/specjax"
VENV="/path/to/venv
SECRETS="~/.specjax.env"
LOG_DIR="/path/to/workspace/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── Load secrets ──────────────────────────────────────────────────────────────
[[ -f "$SECRETS" ]] && { set -a; source "$SECRETS"; set +a; }

# Shared NFS — venv and model are already available
source "$VENV/bin/activate"

TARGET_MODEL="/path/to/models/Qwen2.5-7B-Instruct"
TARGET_MODEL_TYPE="qwen25"
DATA_PATH="$REPO_ROOT/training/data/mixed_54k.jsonl"
OUTPUT_DIR="/path/to/workspace/checkpoints/qwen25-7b-eagle3"

# ── Verify prerequisites ──────────────────────────────────────────────────────
[[ -f "$DATA_PATH" ]]    || { echo "ERROR: dataset missing: $DATA_PATH";    exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: target model missing: $TARGET_MODEL"; exit 1; }

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/qwen25_7b_eagle3_w${WORKER_IDX}_${TIMESTAMP}.log"

echo "[worker_start] Starting training. Log → $LOG_FILE"
echo "[worker_start] coordinator=${COORDINATOR_IP}:${COORDINATOR_PORT}  process=${WORKER_IDX}/4"

# ── Launch training ───────────────────────────────────────────────────────────
export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:${COORDINATOR_PORT}"
export JAX_NUM_PROCESSES=4
export JAX_PROCESS_INDEX="${WORKER_IDX}"
export PJRT_DEVICE=TPU
export XLA_PERSISTENT_CACHE_PATH=/path/to/workspace/xla_cache_jax
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"
export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"

python3 -m specjax.train \
    --target-model-path "$TARGET_MODEL"                              \
    --target-model-type "$TARGET_MODEL_TYPE"                         \
    --data-path         "$DATA_PATH"                                 \
    --output-dir        "$OUTPUT_DIR"                                \
    --exp-name          qwen25-7b-eagle3                             \
    --num-epochs        3                                            \
    --learning-rate     8e-4                                         \
    --max-length        1024                                         \
    --warmup-ratio      0.03                                         \
    --batch-size        4                                            \
    --grad-accum-steps  4                                            \
    --ttt-length        7                                            \
    --log-every         10                                           \
    --save-every        100                                          \
    --wandb-project     qwen25-7b-eagle3-experiments                 \
    --wandb-run-name    qwen25-7b-eagle3                             \
    ${MAX_STEPS_ARG}                                                 \
    2>&1 | stdbuf -oL tee "$LOG_FILE"

echo "[worker_start] Worker $WORKER_IDX done."
