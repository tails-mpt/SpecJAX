#!/usr/bin/env bash
# ============================================================================
# Download DeepSeek-R1-Distill-Qwen-14B to local training storage
#
# Usage:
#   bash scripts/download/download_ds_r1_qwen_14b.sh
# ============================================================================

set -euo pipefail

MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEST="/path/to/models/DeepSeek-R1-Distill-Qwen-14B"

echo "=== Downloading $MODEL_ID ==="
echo "Destination: $DEST"
echo ""

mkdir -p "$DEST"

huggingface-cli download "$MODEL_ID" \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False

echo ""
echo "=== Download complete ==="
echo "Model at: $DEST"
echo "Files:"
ls -lh "$DEST"/*.safetensors "$DEST"/config.json 2>/dev/null || ls -lh "$DEST"
