#!/usr/bin/env bash
# Run offline throughput benchmark WITHOUT speculative decoding (baseline).
#
# Usage:
#   bash inference/run_benchmark_baseline.sh inference/configs/llama32_3b.env
#
# Produces a JSON result file in inference/results/<MODEL_NAME>_baseline.json
# Requires: setup.sh to have been run first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=tpu_env.sh
source "$SCRIPT_DIR/tpu_env.sh"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.env>"
    exit 1
fi

CONFIG="$1"
if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG"

MODEL_NAME="$(basename "$CONFIG" .env)"
RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"
RESULT_FILE="$RESULT_DIR/${MODEL_NAME}_baseline.json"

echo "=== Baseline Benchmark (no EAGLE3): $MODEL_NAME ==="
echo "Target: $TARGET_MODEL"
echo "TP=$TP_SIZE"
echo "Output: $RESULT_FILE"
echo ""

$PYTHON -m sgl_jax.bench_offline_throughput \
    --model-path "$TARGET_MODEL" \
    --tp-size "$TP_SIZE" \
    --dtype "${DTYPE:-bfloat16}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}" \
    --dataset-name sharegpt \
    --num-prompts "${NUM_PROMPTS:-200}" \
    --result-filename "$RESULT_FILE"

echo ""
echo "Results saved to $RESULT_FILE"
