#!/usr/bin/env bash
# ============================================================================
# setup_qwen3_jax_env.sh — bootstrap the JAX training environment for
# Qwen3-Coder-Next Eagle3 training on TPU.
#
# Usage:
#   bash scripts/setup/setup_qwen3_jax_env.sh [--bf16]
# ============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Default to FP8 model; pass --bf16 to use full-precision
MODEL_VARIANT="Qwen/Qwen3-Coder-Next-FP8"
MODEL_DIR="/path/to/models/Qwen3-Coder-Next-FP8"
for arg in "$@"; do
    case "$arg" in
        --bf16)
            MODEL_VARIANT="Qwen/Qwen3-Coder-Next"
            MODEL_DIR="/path/to/models/Qwen3-Coder-Next"
            ;;
    esac
done

echo "=== Qwen3-Coder-Next JAX Training Environment Setup ==="
echo "Model: $MODEL_VARIANT → $MODEL_DIR"
setup_base

# ── Download Qwen3-Coder-Next ────────────────────────────────────────────────
download_model "$MODEL_VARIANT" "$MODEL_DIR"

# ── Build dataset with Qwen3 tokenizer ───────────────────────────────────────
bash "$SCRIPT_DIR/prep_dataset_jax.sh"

echo ""
echo "=== Setup complete. Run training with: ==="
echo "  bash scripts/run/run_exp_qwen3.sh"
