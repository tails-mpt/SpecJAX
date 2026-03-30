#!/usr/bin/env bash
# Download Qwen2.5-7B-Instruct from HuggingFace.
# Output: /path/to/models/Qwen2.5-7B-Instruct
set -euo pipefail

VENV="/path/to/venv
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

MODELS_DIR="/path/to/models"
mkdir -p "$MODELS_DIR"

python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

models = [
    ("Qwen/Qwen2.5-7B-Instruct", "/path/to/models/Qwen2.5-7B-Instruct", False),
]

for repo_id, local_dir, requires_auth in models:
    if os.path.isdir(local_dir) and any(f.endswith(".safetensors") for f in os.listdir(local_dir)):
        print(f"[skip] {local_dir} already exists")
        continue
    if requires_auth and not hf_token:
        print(f"[skip] {repo_id} requires HF_TOKEN (not set) — skipping")
        continue
    print(f"Downloading {repo_id} → {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, token=hf_token)
    print(f"Done: {local_dir}")

print("All models downloaded.")
PYEOF
