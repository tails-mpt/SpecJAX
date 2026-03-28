#!/usr/bin/env bash
# TPU single-host environment setup for sglang-jax inference.
# Source this before running any benchmark on a multi-host TPU pod.
#
# This restricts JAX to the 4 local TPU chips (TP=4) without
# requiring coordination with other hosts in the pod.

export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_PROCESS_BOUNDS=1,1,1
export CLOUD_TPU_TASK_ID=0
export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR

# sglang-jax paths
export SGLANG_JAX_ROOT="${SGLANG_JAX_ROOT:-$SGLANG_JAX_ROOT}"
export VENV_DIR="${VENV_DIR:-/tmp/sglang-jax-venv}"
export PYTHONPATH="$SGLANG_JAX_ROOT/python:${PYTHONPATH:-}"
export PYTHON="$VENV_DIR/bin/python3.12"

# HF token
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.huggingface/token")"
fi
