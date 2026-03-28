#!/usr/bin/env bash
# worker_start_qwen25_14b.sh — run on EVERY worker simultaneously.
# Launch via: bash scripts/run/launch_all_workers_qwen25_14b.sh
#
# Hardware: TPU v4-32, 4 processes x 8 chips = 32 chips, TP=4 DP=8
# Config: B=4, T=2048, grad_accum=2 → effective 64 seqs/step/DP-rank
set -euo pipefail

MAX_STEPS_ARG=""
RESUME_ARG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)    MAX_STEPS_ARG="--max-steps 5" ;;
        --resume=*)   RESUME_ARG="--draft-init-path ${arg#--resume=}" ;;
    esac
done

HOSTNAME_VAL="$(hostname)"
WORKER_IDX="$(echo "$HOSTNAME_VAL" | grep -oP '(?<=-w-)\d+$' || echo '0')"
echo "[worker_start] hostname=$HOSTNAME_VAL  process_index=$WORKER_IDX"

COORDINATOR_IP="${COORDINATOR_IP:-<coordinator-ip>}"
COORDINATOR_PORT="2222"
REPO_ROOT="/path/to/specjax"
VENV="/path/to/venv
LOG_DIR="/path/to/workspace/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

[[ -r ~/.specjax.env ]]              && { set -a; source ~/.specjax.env;              set +a; }
[[ -r /path/to/workspace/secrets.env ]] && { set -a; source /path/to/workspace/secrets.env; set +a; }

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY=$(grep -A1 "api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}' || true)
    export WANDB_API_KEY
fi

source "$VENV/bin/activate"

TARGET_MODEL="/path/to/models/Qwen2.5-14B-Instruct"
TARGET_MODEL_TYPE="qwen25"
DATA_PATH="$REPO_ROOT/training/data/mixed_54k.jsonl"
OUTPUT_DIR="/path/to/workspace/checkpoints/qwen25-14b-eagle3"

[[ -f "$DATA_PATH" ]]    || { echo "ERROR: dataset missing: $DATA_PATH";       exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: target model missing: $TARGET_MODEL"; exit 1; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/qwen25_14b_eagle3_w${WORKER_IDX}_${TIMESTAMP}.log"

echo "[worker_start] Starting training. Log → $LOG_FILE"
echo "[worker_start] coordinator=${COORDINATOR_IP}:${COORDINATOR_PORT}  process=${WORKER_IDX}/4"

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
    --target-model-path "$TARGET_MODEL"    \
    --target-model-type "$TARGET_MODEL_TYPE" \
    ${RESUME_ARG}                           \
    --data-path         "$DATA_PATH"        \
    --output-dir        "$OUTPUT_DIR"       \
    --exp-name          qwen25-14b-eagle3   \
    --num-epochs        3                   \
    --learning-rate     3e-4                \
    --max-length        2048                \
    --warmup-ratio      0.03                \
    --batch-size        2                   \
    --grad-accum-steps  4                   \
    --ttt-length        7                   \
    --log-every         10                  \
    --save-every        0                   \
    --wandb-project     qwen25-14b-eagle3-experiments \
    --wandb-run-name    qwen25-14b-eagle3   \
    ${MAX_STEPS_ARG}                        \
    2>&1 | stdbuf -oL tee "$LOG_FILE"

echo "[worker_start] Worker $WORKER_IDX done."
