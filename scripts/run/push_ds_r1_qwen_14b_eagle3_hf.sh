#!/usr/bin/env bash
# ============================================================================
# Push DeepSeek-R1-Distill-Qwen-14B Eagle3 draft head to HuggingFace.
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/run/push_ds_r1_qwen_14b_eagle3_hf.sh
#   bash scripts/run/push_ds_r1_qwen_14b_eagle3_hf.sh   # uses cached token
#
# Uploads to: thoughtworks/DeepSeek-R1-Distill-Qwen-14B-Eagle3 (private)
# ============================================================================
set -euo pipefail

REPO_ID="thoughtworks/DeepSeek-R1-Distill-Qwen-14B-Eagle3"
CKPT_DIR="/path/to/workspace/checkpoints/ds-r1-qwen-14b-eagle3/epoch_3"
VENV="/path/to/venv

source "$VENV/bin/activate"

# Load token from secrets if not already set
[[ -r ~/.specjax.env ]]              && { set -a; source ~/.specjax.env;              set +a; }
[[ -r /path/to/workspace/secrets.env ]] && { set -a; source /path/to/workspace/secrets.env; set +a; }

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set. Export it or add to secrets.env."
    exit 1
fi

echo "=== Pushing $REPO_ID (private) ==="
echo "Checkpoint: $CKPT_DIR"
ls -lh "$CKPT_DIR"
echo ""

python3 - <<PYEOF
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = "${REPO_ID}"
ckpt_dir = "${CKPT_DIR}"

# Create repo if it doesn't exist (private)
try:
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"create_repo: {e}")

# Upload all files
files = ["config.json", "model.safetensors", "README.md"]
for fname in files:
    fpath = f"{ckpt_dir}/{fname}"
    if not os.path.exists(fpath):
        print(f"SKIP (not found): {fname}")
        continue
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"Uploading {fname} ({size_mb:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Done: {fname}")

print(f"\nAll files uploaded.")
print(f"View at: https://huggingface.co/{repo_id}")
PYEOF
