#!/usr/bin/env bash
# ============================================================================
# Pre-flight checks for GLM-4.7-FP8 Eagle3 JAX training on TPU v4-32
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TARGET_MODEL="/path/to/models/GLM-4.7-FP8"
DATA_PATH="$REPO_ROOT/training/data/mixed_54k.jsonl"
OUTPUT_DIR="/path/to/checkpoints/exp-glm47-fp8-a"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-model-path) TARGET_MODEL="$2"; shift 2 ;;
        --data-path)         DATA_PATH="$2";    shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PASS=0; FAIL=0

check() {
    local name="$1" result="$2"
    if [[ "$result" == ok* ]]; then
        echo "  ✓ $name${result#ok:+  (${result#ok})}"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $name — $result"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== GLM-4.7-FP8 JAX Pre-flight Checks ==="
echo ""

PY_VERSION=$(python3 --version 2>&1) && check "Python" "ok" || check "Python" "$PY_VERSION"

python3 -c "import jax" 2>/dev/null && check "jax importable" "ok" || check "jax importable" "not found"

for pkg in optax safetensors transformers ml_dtypes; do
    python3 -c "import $pkg" 2>/dev/null && check "package: $pkg" "ok" || check "package: $pkg" "not installed"
done

if [[ -d "$TARGET_MODEL" ]] && [[ -f "$TARGET_MODEL/config.json" ]]; then
    N_SHARDS=$(ls "$TARGET_MODEL"/model*.safetensors 2>/dev/null | wc -l)
    check "Target model ($N_SHARDS shards): $TARGET_MODEL" "ok"
else
    check "Target model: $TARGET_MODEL" "not found — run download_models_glm47_fp8.sh"
fi

if [[ -f "$DATA_PATH" ]]; then
    N_LINES=$(wc -l < "$DATA_PATH")
    check "Training data ($N_LINES lines): $DATA_PATH" "ok"
else
    ALT_DATA="$REPO_ROOT/training/data/sharegpt.jsonl"
    if [[ -f "$ALT_DATA" ]]; then
        check "Training data (fallback): $ALT_DATA" "ok"
    else
        check "Training data: $DATA_PATH" "not found"
    fi
fi

# FP8 channel-wise dequant smoke test
python3 - <<'PYEOF' 2>/dev/null && check "FP8 channel-wise dequant smoke test" "ok" \
    || check "FP8 channel-wise dequant smoke test" "failed"
import numpy as np
import ml_dtypes

w_fp8 = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=ml_dtypes.float8_e4m3fn)
scale = np.array([2.0, 3.0], dtype=np.float32)
result = w_fp8.astype(np.float32) * scale[:, np.newaxis]
assert result.shape == (2, 2)
assert not np.any(np.isnan(result))
assert abs(result[0, 0] - 2.0) < 0.1
PYEOF

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[[ "$FAIL" -gt 0 ]] && { echo "Fix the above issues before running training."; exit 1; }
echo "All checks passed — ready to train GLM-4.7-FP8 Eagle3."
