#!/usr/bin/env bash
# ============================================================================
# Download DeepSeek-R1-Distill-Qwen-7B to GCS shared storage
#
# Usage:
#   bash scripts/download/download_ds_r1_qwen_7b.sh
# ============================================================================

set -euo pipefail

MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEST="/path/to/shared-storage/deepseek/models/DeepSeek-R1-Distill-Qwen-7B"
VENV="/path/to/venv

# Load secrets for HF_TOKEN
SECRETS="~/.specjax.env"
if [[ -f "$SECRETS" ]]; then
    set -a; source "$SECRETS"; set +a
fi

echo "=== Downloading $MODEL_ID ==="
echo "Destination: $DEST"
echo ""

source "$VENV/bin/activate"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$DEST')
print('[ok] Download complete')
"

echo ""
echo "=== Download complete ==="
echo "Model at: $DEST"
ls -lh "$DEST"/*.safetensors "$DEST"/config.json 2>/dev/null || ls -lh "$DEST"
