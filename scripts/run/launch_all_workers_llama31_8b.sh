#!/usr/bin/env bash
# Launch Llama-3.1-8B Eagle3 training across all 4 workers of this v4-32.
# Usage: bash scripts/run/launch_all_workers_llama31_8b.sh [--dry-run] [--resume=<ckpt_path>]
set -euo pipefail

SCRIPT="/path/to/specjax/scripts/run/worker_start_llama31_8b.sh"
SSH_KEY="/path/to/workspace/.ssh-shared/id_ed25519"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
WORKERS=(worker-0 worker-1 worker-2 worker-3)

PASS_ARGS=""
for arg in "$@"; do PASS_ARGS="$PASS_ARGS $arg"; done

echo "=== Llama-3.1-8B Eagle3 — launching ${#WORKERS[@]} workers ==="
echo "Script: $SCRIPT $PASS_ARGS"
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

echo "All workers launched. PIDs: ${PIDS[*]}"
echo "Waiting..."

FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "Worker ${WORKERS[$i]} FAILED"; FAIL=1; }
done

[[ $FAIL -eq 0 ]] && echo "=== All workers completed ===" || { echo "=== Some workers FAILED ==="; exit 1; }
