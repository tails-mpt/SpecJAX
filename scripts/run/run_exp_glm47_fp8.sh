#!/usr/bin/env bash
# ============================================================================
# run_exp_glm47_fp8_a.sh — Eagle3 training for GLM-4.7-FP8 on v4-32
#
# Multi-host launch: 4 workers × 4 chips = 16 chips, TP=16 (no DP).
# First experiment: single-step (TTT=1), conservative hyperparameters.
#
# Usage (from any machine with gcloud access):
#   bash scripts/run/run_exp_glm47_fp8.sh
#
# Or directly on w0 (launches worker_start on all workers via gcloud ssh):
#   bash scripts/run/run_exp_glm47_fp8.sh --local
# ============================================================================

set -euo pipefail

TPU_NAME="<tpu-name>"
PROJECT="845031671874"
ZONE="us-central2-b"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER_SCRIPT="scripts/run/worker_start_glm47_fp8.sh"

echo "========================================================"
echo "  Exp GLM47-FP8-A — v4-32 (16 chips, tp=16)"
echo "  Target: GLM-4.7-FP8 (~367B MoE, GQA attention)"
echo "  Mode: single-step (TTT=1), from scratch"
echo "========================================================"
echo ""

# Check if running locally on a worker or remotely
if [[ "${1:-}" == "--local" ]]; then
    echo "Running in local mode — launching on all workers via gcloud ssh ..."

    # Step 1: Setup on all workers
    echo "=== Step 1: Setup (all workers) ==="
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --worker=all \
        --command="cd $REPO_ROOT && bash scripts/setup/setup_glm5_env.sh && bash scripts/download/download_models_glm47_fp8.sh"

    # Step 2: Launch training on all workers
    echo ""
    echo "=== Step 2: Launch training (all workers) ==="
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --worker=all \
        --command="cd $REPO_ROOT && bash $WORKER_SCRIPT"
else
    # Direct execution on the current worker — used when gcloud ssh --worker=all
    # invokes this script on each worker.
    echo "Running worker_start_glm47_fp8.sh on this host ..."
    cd "$REPO_ROOT"
    bash "$WORKER_SCRIPT"
fi

echo ""
echo "=== Training complete ==="
echo "Checkpoints at: /path/to/checkpoints/exp-glm47-fp8-a"
