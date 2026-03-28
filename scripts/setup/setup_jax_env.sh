#!/usr/bin/env bash
# ============================================================================
# setup_jax_env.sh — bootstrap the JAX training environment for GLM-4.7-Flash
# on a fresh / preempted TPU VM.
#
# Usage:
#   bash scripts/setup/setup_jax_env.sh
# ============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "=== JAX Training Environment Setup (GLM-Flash) ==="
setup_base

# ── Download GLM-Flash models ────────────────────────────────────────────────
bash "$SCRIPT_DIR/../download/download_models_jax.sh"

echo ""
echo "=== Setup complete. Run training with: ==="
echo "  bash scripts/run/run_exp_glm_flash.sh"
