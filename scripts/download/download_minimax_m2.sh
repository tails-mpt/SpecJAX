#!/usr/bin/env bash
# Download MiniMax-M2.5 model weights (FP8 quantized, ~230 GB).
# HuggingFace repo: MiniMaxAI/MiniMax-M2.5
set -euo pipefail

VENV="/path/to/venv
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

# Load HF token (MiniMax-M2.5 is public but token avoids rate limits)
SECRETS="~/.specjax.env"
[[ -f "$SECRETS" ]] && { set -a; source "$SECRETS"; set +a; }

MODELS_DIR="/path/to/models"
LOCAL_DIR="$MODELS_DIR/MiniMax-M2.5"
mkdir -p "$MODELS_DIR"

python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
local_dir = os.environ.get("LOCAL_DIR", "/path/to/models/MiniMax-M2.5")

if os.path.isdir(local_dir) and any(f.endswith(".safetensors") for f in os.listdir(local_dir)):
    print(f"[skip] {local_dir} already exists")
else:
    print(f"Downloading MiniMaxAI/MiniMax-M2.5 → {local_dir} ...")
    print("Note: ~230 GB FP8 weights across 125 shard files. This will take a while.")
    snapshot_download(
        repo_id="MiniMaxAI/MiniMax-M2.5",
        local_dir=local_dir,
        token=hf_token,
    )
    print(f"[done] {local_dir}")
PYEOF

echo "Download complete. Model at: $LOCAL_DIR"
