#!/usr/bin/env bash
# ============================================================================
# Pre-flight checks for JAX EAGLE3 training on TPU
#
# Verifies:
#   1. JAX venv is active and can see TPU devices
#   2. Required packages installed (jax, optax, safetensors, transformers)
#   3. Target model weights exist
#   4. Training data exists
#   5. GCS mount is accessible (for checkpoints + XLA cache)
#   6. XLA smoke test (jit a small matmul)
#
# Usage:
#   bash scripts/preflight/preflight_jax.sh \
#       --target-model-path /path/to/models/Qwen3-8B \
#       --data-path         training/data/sharegpt.jsonl \
#       --output-dir        /path/to/checkpoints/qwen3-8b-eagle3
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TARGET_MODEL=""
DATA_PATH="$REPO_ROOT/training/data/sharegpt.jsonl"
OUTPUT_DIR=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-model-path) TARGET_MODEL="$2"; shift 2 ;;
        --data-path)         DATA_PATH="$2";    shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0

check() {
    local name="$1"
    local result="$2"   # starts with "ok" on success, or an error message
    if [[ "$result" == ok* ]]; then
        local detail="${result#ok}"
        echo "  ✓ $name${detail:+  ($detail)}"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name — $result"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== JAX Pre-flight Checks ==="
echo ""

# 1. Python / venv
PY_VERSION=$(python3 --version 2>&1) && check "Python" "ok" || check "Python" "$PY_VERSION"

# 2. JAX importable
python3 -c "import jax" 2>/dev/null \
    && check "jax importable" "ok" \
    || check "jax importable" "pip install jax[tpu]"

# 3. JAX sees TPU devices
N_DEVICES=$(python3 -c "import jax; print(len(jax.devices()))" 2>/dev/null || echo "0")
if [[ "$N_DEVICES" -ge 1 ]]; then
    check "JAX TPU devices ($N_DEVICES chips)" "ok"
else
    check "JAX TPU devices" "no TPU devices found — is PJRT_DEVICE=TPU?"
fi

# 4. Required packages
for pkg in optax safetensors transformers orbax; do
    python3 -c "import $pkg" 2>/dev/null \
        && check "package: $pkg" "ok" \
        || check "package: $pkg" "not installed"
done

# 5. Target model
if [[ -d "$TARGET_MODEL" ]] && [[ -f "$TARGET_MODEL/config.json" ]]; then
    N_SHARDS=$(ls "$TARGET_MODEL"/model*.safetensors 2>/dev/null | wc -l)
    check "Target model ($N_SHARDS shards): $TARGET_MODEL" "ok"
else
    check "Target model: $TARGET_MODEL" "not found — run scripts/download/download_models_jax.sh"
fi

# 6. Training data
if [[ -f "$DATA_PATH" ]]; then
    N_LINES=$(wc -l < "$DATA_PATH")
    check "Training data ($N_LINES lines): $DATA_PATH" "ok"
else
    check "Training data: $DATA_PATH" "not found — download training data first"
fi

# 7. GCS mount
if mountpoint -q /mnt/gcs 2>/dev/null || [[ -d /mnt/gcs ]]; then
    check "GCS mount /mnt/gcs" "ok"
else
    check "GCS mount /mnt/gcs" "not mounted — gcsfuse needed for checkpoints"
fi

# 8. Output dir writable
mkdir -p "$OUTPUT_DIR" 2>/dev/null \
    && check "Output dir writable: $OUTPUT_DIR" "ok" \
    || check "Output dir writable: $OUTPUT_DIR" "cannot create — check GCS permissions"

# 9. XLA cache dir writable
XLA_CACHE="${XLA_PERSISTENT_CACHE_PATH:-/tmp/xla_cache_jax}"
mkdir -p "$XLA_CACHE" 2>/dev/null \
    && check "XLA cache dir: $XLA_CACHE" "ok" \
    || check "XLA cache dir: $XLA_CACHE" "cannot create"

# 10. JAX XLA smoke test
python3 - <<'PYEOF' 2>/dev/null && check "XLA smoke test (jit matmul)" "ok" \
    || check "XLA smoke test" "failed — check JAX/TPU setup"
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
import jax
import jax.numpy as jnp

@jax.jit
def matmul(a, b):
    return a @ b

a = jnp.ones((8, 8), dtype=jnp.bfloat16)
b = jnp.ones((8, 8), dtype=jnp.bfloat16)
c = matmul(a, b)
assert c.shape == (8, 8)
PYEOF

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [[ "$FAIL" -gt 0 ]]; then
    echo "Fix the above issues before running training."
    exit 1
fi

echo "All checks passed — ready to train."
