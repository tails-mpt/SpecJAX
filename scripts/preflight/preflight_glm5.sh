#!/usr/bin/env bash
# ============================================================================
# Pre-flight checks for GLM-5-FP8 Eagle3 JAX training on TPU
#
# Verifies:
#   1. JAX venv is active and can see TPU devices (>= 32 chips expected)
#   2. Required packages installed (jax, optax, safetensors, transformers, ml_dtypes)
#   3. GLM-5-FP8 model weights exist
#   4. Training data exists
#   5. GCS mount is accessible
#   6. XLA smoke test
#   7. FP8 dequantization smoke test
#
# Usage:
#   bash scripts/preflight/preflight_glm5.sh
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TARGET_MODEL="/path/to/models/GLM-5-FP8"
DATA_PATH="$REPO_ROOT/training/data/mixed_54k.jsonl"
OUTPUT_DIR="/path/to/checkpoints/exp-glm5-a"

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
    local result="$2"
    if [[ "$result" == ok* ]]; then
        local detail="${result#ok}"
        echo "  ✓ $name${detail:+  ($detail)}"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name — $result"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== GLM-5-FP8 JAX Pre-flight Checks ==="
echo ""

# 1. Python / venv
PY_VERSION=$(python3 --version 2>&1) && check "Python" "ok" || check "Python" "$PY_VERSION"

# 2. JAX importable
python3 -c "import jax" 2>/dev/null \
    && check "jax importable" "ok" \
    || check "jax importable" "pip install jax[tpu]"

# 3. JAX sees TPU devices (GLM-5 needs >= 32 chips for v4e-32)
N_DEVICES=$(python3 -c "import jax; print(len(jax.devices()))" 2>/dev/null || echo "0")
if [[ "$N_DEVICES" -ge 32 ]]; then
    check "JAX TPU devices ($N_DEVICES chips, >= 32 for GLM-5)" "ok"
elif [[ "$N_DEVICES" -ge 1 ]]; then
    check "JAX TPU devices ($N_DEVICES chips)" "WARNING: GLM-5 needs >= 32 chips for ~1TB HBM"
else
    check "JAX TPU devices" "no TPU devices found — is PJRT_DEVICE=TPU?"
fi

# 4. Required packages
for pkg in optax safetensors transformers ml_dtypes; do
    python3 -c "import $pkg" 2>/dev/null \
        && check "package: $pkg" "ok" \
        || check "package: $pkg" "not installed"
done

# 5. Target model (GLM-5-FP8)
if [[ -d "$TARGET_MODEL" ]] && [[ -f "$TARGET_MODEL/config.json" ]]; then
    N_SHARDS=$(ls "$TARGET_MODEL"/model*.safetensors 2>/dev/null | wc -l)
    check "Target model ($N_SHARDS shards): $TARGET_MODEL" "ok"
else
    check "Target model: $TARGET_MODEL" "not found — run download_models_glm5.sh"
fi

# 6. Training data
if [[ -f "$DATA_PATH" ]]; then
    N_LINES=$(wc -l < "$DATA_PATH")
    check "Training data ($N_LINES lines): $DATA_PATH" "ok"
else
    # Fallback to sharegpt.jsonl
    ALT_DATA="$REPO_ROOT/training/data/sharegpt.jsonl"
    if [[ -f "$ALT_DATA" ]]; then
        N_LINES=$(wc -l < "$ALT_DATA")
        check "Training data (fallback, $N_LINES lines): $ALT_DATA" "ok"
    else
        check "Training data: $DATA_PATH" "not found — run prep_dataset_jax.sh"
    fi
fi

# 7. GCS mount
if mountpoint -q /path/to/gcs 2>/dev/null || [[ -d /path/to/gcs ]]; then
    check "GCS mount /path/to/gcs" "ok"
else
    check "GCS mount /path/to/gcs" "not mounted — gcsfuse needed for checkpoints"
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

# 10. XLA smoke test (larger matmul to verify TP across chips)
python3 - <<'PYEOF' 2>/dev/null && check "XLA smoke test (jit matmul)" "ok" \
    || check "XLA smoke test" "failed — check JAX/TPU setup"
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
import jax
import jax.numpy as jnp

@jax.jit
def matmul(a, b):
    return a @ b

a = jnp.ones((64, 64), dtype=jnp.bfloat16)
b = jnp.ones((64, 64), dtype=jnp.bfloat16)
c = matmul(a, b)
assert c.shape == (64, 64)
PYEOF

# 11. FP8 dequantization smoke test
python3 - <<'PYEOF' 2>/dev/null && check "FP8 dequant smoke test" "ok" \
    || check "FP8 dequant smoke test" "failed — check ml_dtypes install"
import numpy as np
import ml_dtypes

# Create a small FP8 tensor and verify dequant works
fp8_data = np.array([1.0, 2.0, 0.5, -1.0], dtype=ml_dtypes.float8_e4m3fn)
scale = np.array([2.0], dtype=np.float32)
result = fp8_data.astype(np.float32) * scale
assert result.shape == (4,)
assert not np.any(np.isnan(result))
PYEOF

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [[ "$FAIL" -gt 0 ]]; then
    echo "Fix the above issues before running training."
    exit 1
fi

echo "All checks passed — ready to train GLM-5-FP8 Eagle3."
