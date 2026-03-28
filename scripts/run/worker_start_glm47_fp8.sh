#!/usr/bin/env bash
# ============================================================================
# worker_start_glm47_fp8.sh — run on EVERY worker simultaneously.
#
# Auto-detects JAX_PROCESS_INDEX from the hostname suffix (-w-N).
# Run on all 4 workers at the same time:
#
#   gcloud compute tpus tpu-vm ssh $TPU_NAME \
#       --worker=all \
#       --command="bash $REPO_ROOT/scripts/run/worker_start_glm47_fp8.sh"
#
# w0 (process 0) is the JAX coordinator. Workers 1-3 connect to it.
# All 4 must start within the JAX coordinator timeout (~5 minutes).
# ============================================================================

set -euo pipefail

# ── Detect worker index from hostname (e.g. t1v-n-...-w-2 → 2) ──────────────
HOSTNAME_VAL="$(hostname)"
WORKER_IDX="$(echo "$HOSTNAME_VAL" | grep -oP '(?<=-w-)\d+$' || echo '0')"
echo "[worker_start] hostname=$HOSTNAME_VAL  process_index=$WORKER_IDX"

# w0 internal IP — detected from this v4-32 slice
COORDINATOR_IP="${COORDINATOR_IP:-<coordinator-ip>}"
COORDINATOR_PORT="2222"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/path/to/venv
SECRETS="/path/to/workspace/secrets.env"
LOG_DIR="/path/to/workspace/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ── Load secrets ──────────────────────────────────────────────────────────────
[[ -f "$SECRETS" ]] && { set -a; source "$SECRETS"; set +a; }
echo "[worker_start] WANDB_API_KEY set: $( [[ -n "${WANDB_API_KEY:-}" ]] && echo yes || echo NO )"

# ── Ensure python3-venv is available ─────────────────────────────────────────
python3 -c "import ensurepip" 2>/dev/null || {
    echo "[worker_start] installing python3-venv ..."
    sudo apt-get install -y python3-venv python3-pip -q
}

# Shared NFS — venv and model are already available
source "$VENV/bin/activate"
TARGET_MODEL="/path/to/models/GLM-4.7-FP8"

# ── Verify prerequisites ──────────────────────────────────────────────────────
DATA_PATH="$REPO_ROOT/training/data/mixed_54k.jsonl"
if [[ ! -f "$DATA_PATH" ]]; then
    DATA_PATH="$REPO_ROOT/training/data/sharegpt.jsonl"
fi

[[ -f "$DATA_PATH" ]]    || { echo "ERROR: dataset missing: $DATA_PATH";    exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: target model missing: $TARGET_MODEL"; exit 1; }

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp_glm47_fp8_a_w${WORKER_IDX}_${TIMESTAMP}.log"

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
    --target-model-path "$TARGET_MODEL"                          \
    --data-path         "$DATA_PATH"                             \
    --output-dir        /path/to/workspace/checkpoints/exp-glm47-fp8-a    \
    --exp-name          exp-glm47-fp8-a                          \
    --num-epochs        3                                        \
    --learning-rate     1e-4                                     \
    --max-length        512                                      \
    --warmup-ratio      0.03                                     \
    --batch-size        2                                        \
    --grad-accum-steps  16                                       \
    --ttt-length        1                                        \
    --log-every         10                                       \
    --save-every        200                                      \
    --wandb-project     glm4-large-eagle3-experiments            \
    --wandb-run-name    exp-glm47-fp8-a                          \
    --target-model-type glm47-fp8                                \
    --tp-size           16                                       \
    2>&1 | stdbuf -oL tee "$LOG_FILE"

echo "[worker_start] Worker $WORKER_IDX done."
