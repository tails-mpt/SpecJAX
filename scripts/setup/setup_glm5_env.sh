#!/usr/bin/env bash
# ============================================================================
# setup_glm5_env.sh — bootstrap the JAX training environment for GLM-5-FP8
# on a fresh / preempted TPU VM.
#
# Usage:
#   bash scripts/setup/setup_glm5_env.sh
# ============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "=== GLM-5-FP8 JAX Training Environment Setup ==="
setup_base

# ── Ensure ml_dtypes for FP8 support ─────────────────────────────────────────
"$VENV/bin/python" -m pip install --quiet "ml_dtypes>=0.4.0"
"$VENV/bin/python" -c "import ml_dtypes; print(f'  ml_dtypes={ml_dtypes.__version__} (FP8 support ready)')"

# ── Download GLM-5-FP8 model ─────────────────────────────────────────────────
bash "$SCRIPT_DIR/../download/download_models_glm5.sh"

echo ""
echo "=== Setup complete. Run training with: ==="
echo "  bash scripts/run/run_exp_glm5.sh"
