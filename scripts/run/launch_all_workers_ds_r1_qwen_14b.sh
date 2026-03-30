#!/usr/bin/env bash
# ============================================================================
# Launch DS-R1-Distill-Qwen-14B Eagle3 training across all 4 workers.
#
# Usage:
#   bash scripts/run/launch_all_workers_ds_r1_qwen_14b.sh [--dry-run] [--resume=<ckpt_path>]
# ============================================================================
set -euo pipefail

SCRIPT="/path/to/specjax/scripts/run/worker_start_ds_r1_qwen_14b.sh"
SSH_KEY="/path/to/workspace/.ssh-shared/id_ed25519"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# Workers: w-0 is this machine (run locally); w-1,w-2,w-3 via SSH
WORKERS=(worker-0 worker-1 worker-2 worker-3)

PASS_ARGS=""
for arg in "$@"; do
    PASS_ARGS="$PASS_ARGS $arg"
done

echo "=== DS-R1-Distill-Qwen-14B Eagle3 — launching ${#WORKERS[@]} workers ==="
echo "Script: $SCRIPT $PASS_ARGS"
echo "Workers: ${WORKERS[*]}"
echo ""

PIDS=()
for i in "${!WORKERS[@]}"; do
    w="${WORKERS[$i]}"
    if [[ $i -eq 0 ]]; then
        echo "[$w] Starting locally..."
        bash "$SCRIPT" $PASS_ARGS &
    else
        echo "[$w] Starting via SSH..."
        ssh $SSH_OPTS "$w" "bash $SCRIPT $PASS_ARGS" &
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
