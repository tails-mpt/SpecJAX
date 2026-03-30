#!/usr/bin/env bash
# ============================================================================
# _common.sh — shared setup functions for all SpecJAX environment scripts.
#
# Source this file from model-specific setup scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="/path/to/venv
SECRETS="~/.specjax.env"

# --- Shared setup functions ---------------------------------------------------

setup_system_deps() {
    if ! python3 -m venv --help &>/dev/null; then
        echo "Installing python3-venv ..."
        sudo apt-get install -y python3-venv
    fi
}

load_secrets() {
    if [[ -f "$SECRETS" ]]; then
        set -a; source "$SECRETS"; set +a
        echo "[ok] Loaded secrets from $SECRETS"
    else
        echo "[warn] No secrets file at $SECRETS — HuggingFace / W&B auth may fail"
    fi
}

setup_venv() {
    if [[ -f "$VENV/bin/activate" ]]; then
        echo "[skip] venv already exists at $VENV"
    else
        [[ -d "$VENV" ]] && rm -rf "$VENV"
        echo "Creating venv ..."
        python3 -m venv "$VENV"
        echo "[ok] venv created"
    fi
    source "$VENV/bin/activate"

    echo "Installing / updating requirements ..."
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/python" -m pip install --quiet -e "$REPO_ROOT"
    echo "[ok] requirements installed"
}

verify_imports() {
    "$VENV/bin/python" -c "import jax, optax; print(f'  jax={jax.__version__}  optax={optax.__version__}')"
}

setup_training_dirs() {
    if [[ ! -d /mnt/training ]]; then
        sudo mkdir -p /mnt/training
        sudo chown "$(whoami)" /mnt/training
    fi
    mkdir -p /path/to/models
    echo "[ok] /path/to/models ready"
}

download_model() {
    local model_id="$1"
    local model_dir="$2"
    if [[ -d "$model_dir" ]] && [[ -f "$model_dir/config.json" ]]; then
        echo "[skip] Model already downloaded at $model_dir"
    else
        echo "Downloading $model_id → $model_dir ..."
        "$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('$model_id', local_dir='$model_dir')
print('[ok] Model downloaded')
"
    fi
}

# --- Run common base setup ---------------------------------------------------

setup_base() {
    echo "Repo root: $REPO_ROOT"
    echo "Venv:      $VENV"
    echo ""
    setup_system_deps
    load_secrets
    setup_venv
    verify_imports
    setup_training_dirs
}
