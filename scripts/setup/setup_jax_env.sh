#!/usr/bin/env bash
# ============================================================================
# setup_jax_env.sh — bootstrap the JAX training environment for SpecJAX
# on a fresh / preempted TPU VM.
#
# Usage:
#   bash scripts/setup/setup_jax_env.sh
# ============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "=== JAX Training Environment Setup ==="
setup_base

echo ""
echo "=== Setup complete ==="
echo "Download a target model, then train with:"
echo "  python -m specjax.train --target-model-path /path/to/model --target-model-type qwen3 --data-path data/sharegpt.jsonl --output-dir checkpoints/"
