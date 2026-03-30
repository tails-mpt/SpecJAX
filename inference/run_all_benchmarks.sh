#!/usr/bin/env bash
# Run baseline + EAGLE3 benchmarks for all model configs.
#
# Usage:
#   bash inference/run_all_benchmarks.sh [configs...]
#
# If no configs specified, runs all .env files in inference/configs/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=("$SCRIPT_DIR"/configs/*.env)
fi

echo "=== SpecJAX Inference Benchmark Suite ==="
echo "Models: ${#CONFIGS[@]}"
echo ""

for config in "${CONFIGS[@]}"; do
    MODEL_NAME="$(basename "$config" .env)"
    echo "=============================="
    echo "  Model: $MODEL_NAME"
    echo "=============================="

    echo "--- Baseline ---"
    bash "$SCRIPT_DIR/run_benchmark_baseline.sh" "$config" || {
        echo "WARNING: Baseline failed for $MODEL_NAME, skipping..."
        continue
    }

    echo ""
    echo "--- EAGLE3 ---"
    bash "$SCRIPT_DIR/run_benchmark.sh" "$config" || {
        echo "WARNING: EAGLE3 benchmark failed for $MODEL_NAME, skipping..."
        continue
    }

    echo ""
done

echo "=== All benchmarks complete ==="
echo "Run: python inference/compare_results.py inference/results/"
