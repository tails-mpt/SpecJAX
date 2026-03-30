#!/usr/bin/env bash
# ============================================================================
# Launch Qwen2.5-7B EAGLE3 training on all 4 workers of this v4-32.
#
# Usage:
#   bash scripts/run/launch_all_workers_qwen25.sh [--dry-run]
# ============================================================================
set -euo pipefail

SCRIPT="/path/to/specjax/scripts/run/worker_start_qwen25_7b.sh"
WORKERS=(worker-0 worker-1 worker-2 worker-3)

DRY_RUN_ARG=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN_ARG="--dry-run" ;;
    esac
done

echo "=== Launching on ${#WORKERS[@]} workers ==="
echo "Script: $SCRIPT $DRY_RUN_ARG"
echo ""

PIDS=()
for w in "${WORKERS[@]}"; do
    echo "Starting worker: $w"
    if [[ "$w" == "$(hostname)" || "$w" == "worker-0" ]]; then
        bash "$SCRIPT" $DRY_RUN_ARG &
    else
        ssh -o StrictHostKeyChecking=no "$w" "bash $SCRIPT $DRY_RUN_ARG" &
    fi
    PIDS+=($!)
done

echo ""
echo "All workers launched. PIDs: ${PIDS[*]}"
echo "Waiting for all workers to complete..."
echo ""

FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || {
        echo "Worker ${WORKERS[$i]} (PID ${PIDS[$i]}) FAILED"
        FAIL=1
    }
done

if [[ $FAIL -eq 0 ]]; then
    echo "=== All workers completed successfully ==="
else
    echo "=== Some workers FAILED ==="
    exit 1
fi
