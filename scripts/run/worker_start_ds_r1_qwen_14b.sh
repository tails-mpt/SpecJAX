#!/usr/bin/env bash
# ============================================================================
# worker_start_ds_r1_qwen_14b.sh — run on EVERY worker simultaneously.
#
# Auto-detects JAX_PROCESS_INDEX from the hostname suffix (-w-N).
# Launch via: bash scripts/run/launch_all_workers_ds_r1_qwen_14b.sh
#
# Workers: w-0 (coordinator), w-1, w-2, w-3
# All 4 must start within the JAX coordinator timeout (~5 minutes).
# Hardware: TPU v4-32, 4 processes x 4 chips = 16 chips, TP=4 DP=4
# ============================================================================

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
MAX_STEPS_ARG=""
RESUME_ARG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            MAX_STEPS_ARG="--max-steps 5"
            ;;
        --resume=*)
            RESUME_ARG="--draft-init-path ${arg#--resume=}"
            ;;
    esac
done

# ── Detect worker index from hostname (e.g. t1v-n-...-w-2 → 2) ──────────────
HOSTNAME_VAL="$(hostname)"
WORKER_IDX="$(echo "$HOSTNAME_VAL" | grep -oP '(?<=-w-)\d+$' || echo '0')"
echo "[worker_start] hostname=$HOSTNAME_VAL  process_index=$WORKER_IDX"

COORDINATOR_IP="${COORDINATOR_IP:-<coordinator-ip>}"
COORDINATOR_PORT="2222"
REPO_ROOT="/path/to/specjax"
VENV="/path/to/venv
LOG_DIR="/path/to/workspace/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── Load secrets (try both locations) ─────────────────────────────────────────
[[ -r ~/.specjax.env ]]           && { set -a; source ~/.specjax.env;           set +a; }
[[ -r /path/to/workspace/secrets.env ]] && { set -a; source /path/to/workspace/secrets.env; set +a; }

# W&B key from netrc if not already set
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY=$(grep -A1 "api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}' || true)
    export WANDB_API_KEY
fi

source "$VENV/bin/activate"

TARGET_MODEL="/path/to/models/DeepSeek-R1-Distill-Qwen-14B"
TARGET_MODEL_TYPE="qwen2"
DATA_PATH="$REPO_ROOT/training/data/mixed_54k.jsonl"
OUTPUT_DIR="/path/to/workspace/checkpoints/ds-r1-qwen-14b-eagle3"

# ── Verify prerequisites ──────────────────────────────────────────────────────
[[ -f "$DATA_PATH" ]]    || { echo "ERROR: dataset missing: $DATA_PATH";       exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: target model missing: $TARGET_MODEL"; exit 1; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/ds_r1_qwen_14b_eagle3_w${WORKER_IDX}_${TIMESTAMP}.log"

echo "[worker_start] Starting training. Log → $LOG_FILE"
echo "[worker_start] coordinator=${COORDINATOR_IP}:${COORDINATOR_PORT}  process=${WORKER_IDX}/4"

# ── Launch training ───────────────────────────────────────────────────────────
export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:${COORDINATOR_PORT}"
export JAX_NUM_PROCESSES=4
export JAX_PROCESS_INDEX="${WORKER_IDX}"
export PJRT_DEVICE=TPU
export XLA_PERSISTENT_CACHE_PATH=/tmp/xla_cache_ds_r1_14b
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"
export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"

python3 -m specjax.train \
    --target-model-path "$TARGET_MODEL"                          \
    --target-model-type "$TARGET_MODEL_TYPE"                     \
    --data-path         "$DATA_PATH"                             \
    --output-dir        "$OUTPUT_DIR"                            \
    --exp-name          ds-r1-qwen-14b-eagle3                    \
    --num-epochs        3                                        \
    --learning-rate     8e-4                                     \
    --max-length        512                                      \
    --warmup-ratio      0.03                                     \
    --batch-size        2                                        \
    --grad-accum-steps  8                                        \
    --ttt-length        7                                        \
    --log-every         10                                       \
    --save-every        200                                      \
    --wandb-project     ds-r1-qwen-14b-eagle3-experiments        \
    --wandb-run-name    ds-r1-qwen-14b-eagle3                    \
    ${RESUME_ARG}                                                \
    ${MAX_STEPS_ARG}                                             \
    2>&1 | stdbuf -oL tee "$LOG_FILE"

echo "[worker_start] Worker $WORKER_IDX done."
