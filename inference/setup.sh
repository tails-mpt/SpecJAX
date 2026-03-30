#!/usr/bin/env bash
# Set up a Python 3.12 venv with sglang-jax for SpecJAX inference benchmarks.
#
# Usage:
#   bash inference/setup.sh
#
# Prerequisites:
#   - Python 3.12 installed (apt install python3.12 python3.12-venv)
#   - HF_TOKEN set in environment or stored at ~/.huggingface/token
#   - sglang-jax source cloned locally (or set SGLANG_JAX_ROOT)
#
# Creates a venv at VENV_DIR (default: /tmp/sglang-jax-venv) with all dependencies.

set -euo pipefail

SGLANG_JAX_ROOT="${SGLANG_JAX_ROOT:-./sglang-jax}"
VENV_DIR="${VENV_DIR:-/tmp/sglang-jax-venv}"

echo "=== SpecJAX Inference Setup ==="

# 1. Verify sglang-jax source exists
if [ ! -d "$SGLANG_JAX_ROOT/python" ]; then
    echo "Error: sglang-jax not found at $SGLANG_JAX_ROOT"
    echo "Clone it with: git clone https://github.com/tails-mpt/sglang-jax $SGLANG_JAX_ROOT"
    exit 1
fi

# 2. Verify Python 3.12
if ! command -v python3.12 &>/dev/null; then
    echo "Error: Python 3.12 required but not found."
    echo "Install with: sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi

# 3. Create venv if it doesn't exist
if [ ! -f "$VENV_DIR/bin/python3.12" ]; then
    echo "Creating Python 3.12 venv at $VENV_DIR ..."
    python3.12 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python3.12" -m ensurepip --upgrade
fi

# 4. Install dependencies
echo "Installing sglang-jax dependencies ..."
"$VENV_DIR/bin/pip" install -q \
    "jax==0.8.1" "jaxlib==0.8.1" "flax==0.12.4" \
    "fastapi~=0.116.1" "huggingface-hub~=0.34.3" "jinja2~=3.1.6" \
    "numpy~=2.2.6" "orjson~=3.11.1" "pillow~=11.3.0" "psutil~=7.0.0" \
    "pydantic~=2.11.7" "python-multipart>=0.0.20" "pyzmq~=27.0.1" \
    "safetensors~=0.6.1" "setproctitle~=1.3.6" "tiktoken>=0.10.0" \
    "typing-extensions~=4.14.1" "uvicorn~=0.35.0" "uvloop~=0.21.0" \
    "httpx" "openai" "pandas" "aiohttp" "pybase64" "partial_json_parser" \
    "omegaconf" "pathwaysutils" "llguidance~=1.3.0" "modelscope~=1.28.2" \
    "msgpack-python~=0.5.6" "transformers~=4.57.1" "requests~=2.32.4"

# 5. Verify HF token
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN found in environment."
elif [ -f "$HOME/.huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.huggingface/token")"
    echo "HF_TOKEN loaded from ~/.huggingface/token."
else
    echo "Warning: No HF_TOKEN found. Private models will not be accessible."
    echo "Set HF_TOKEN or run: huggingface-cli login"
fi

# 6. Quick import check
PYTHONPATH="$SGLANG_JAX_ROOT/python" "$VENV_DIR/bin/python3.12" -c \
    "from sgl_jax.bench_offline_throughput import BenchArgs; print('sglang-jax benchmark tools ready')" \
    && echo "Setup complete. Venv at: $VENV_DIR" \
    || { echo "Error: sgl_jax import failed"; exit 1; }
