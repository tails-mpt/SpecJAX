#!/usr/bin/env bash
# Download all models needed for the pending training runs.
# Models: Llama-3.1-8B-Instruct, Qwen2.5-14B-Instruct, Qwen3-8B, Qwen3-14B
set -euo pipefail

VENV="/path/to/venv
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

# Load HF token
SECRETS="~/.specjax.env"
[[ -f "$SECRETS" ]] && { set -a; source "$SECRETS"; set +a; }

MODELS_DIR="/path/to/models"
mkdir -p "$MODELS_DIR"

python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
if hf_token:
    print(f"[ok] HF_TOKEN present ({hf_token[:8]}...)")
else:
    print("[warn] HF_TOKEN not set — gated models will fail")

models = [
    ("meta-llama/Llama-3.1-8B-Instruct",  "/path/to/models/Llama-3.1-8B-Instruct",  True),
    ("Qwen/Qwen2.5-14B-Instruct",           "/path/to/models/Qwen2.5-14B-Instruct",    False),
    ("Qwen/Qwen3-8B",                        "/path/to/models/Qwen3-8B",                False),
    ("Qwen/Qwen3-14B",                       "/path/to/models/Qwen3-14B",               False),
]

for repo_id, local_dir, requires_auth in models:
    if os.path.isdir(local_dir) and any(f.endswith(".safetensors") for f in os.listdir(local_dir)):
        print(f"[skip] {local_dir} already exists")
        continue
    if requires_auth and not hf_token:
        print(f"[skip] {repo_id} requires HF_TOKEN (not set)")
        continue
    print(f"Downloading {repo_id} → {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, token=hf_token)
    print(f"[done] {local_dir}")

print("All downloads complete.")
PYEOF
