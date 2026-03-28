#!/usr/bin/env bash
# Download GLM-5-FP8 from HuggingFace.
# Output: /path/to/models/GLM-5-FP8
#
# NOTE: GLM-5-FP8 is ~744B params in FP8 format. The download is very large
# (~400GB+ in safetensors) and will take significant time.
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
    ("zai-org/GLM-5-FP8", "/path/to/models/GLM-5-FP8", False),
]

for repo_id, local_dir, requires_auth in models:
    if os.path.isdir(local_dir) and any(f.endswith(".safetensors") for f in os.listdir(local_dir)):
        print(f"[skip] {local_dir} already exists")
        continue
    if requires_auth and not hf_token:
        print(f"[skip] {repo_id} requires HF_TOKEN (not set) — skipping")
        continue
    print(f"Downloading {repo_id} → {local_dir} ...")
    print(f"  (This is a large model, ~400GB+ — download may take a while)")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, token=hf_token)
    print(f"Done: {local_dir}")

print("All models downloaded.")
PYEOF
