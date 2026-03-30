#!/usr/bin/env bash
# ============================================================================
# prep_dataset_jax_d.sh — Build the 54K mixed training dataset for exp-jax-d/e
#
# Mirrors the GPU Exp K/J dataset composition:
#   45% ShareGPT    (~24,300 samples)  anon8231489123/ShareGPT_Vicuna_unfiltered
#   35% UltraChat   (~18,900 samples)  HuggingFaceH4/ultrachat_200k
#   20% PerfectBlend (~10,800 samples) mlabonne/open-perfectblend
#   ──────────────────────────────────────────────────────────
#   Total:           54,000 samples
#
# Output: training/data/mixed_54k.jsonl  (OpenAI messages format)
# Format: {"messages": [{"role": "user"/"assistant", "content": "..."}]}
#
# Idempotent: skips if output already exists.
#
# Usage:
#   bash scripts/setup/prep_dataset_jax.sh
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="/path/to/venv
OUTPUT="$REPO_ROOT/training/data/mixed_54k.jsonl"

# ── Load secrets (HF_TOKEN for private datasets if needed) ───────────────────
SECRETS="~/.specjax.env"
if [[ -f "$SECRETS" ]]; then
    set -a; source "$SECRETS"; set +a
fi

# ── Idempotency check ─────────────────────────────────────────────────────────
if [[ -f "$OUTPUT" ]]; then
    N=$(wc -l < "$OUTPUT")
    echo "[skip] $OUTPUT already exists ($N lines)"
    exit 0
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV — run setup_jax_env.sh first"
    exit 1
fi
source "$VENV/bin/activate"

mkdir -p "$(dirname "$OUTPUT")"
echo "Building 54K mixed dataset → $OUTPUT"
echo "Mix: 45% ShareGPT + 35% UltraChat + 20% PerfectBlend"
echo ""

export OUTPUT_PATH="$OUTPUT"

python3 - <<'PYEOF'
import json, os, random, sys

OUTPUT    = os.environ["OUTPUT_PATH"]
SEED      = 42
N_TOTAL   = 54000
N_SG      = int(N_TOTAL * 0.45)   # 24,300  ShareGPT
N_UC      = int(N_TOTAL * 0.35)   # 18,900  UltraChat
N_PB      = N_TOTAL - N_SG - N_UC # 10,800  PerfectBlend

random.seed(SEED)

from datasets import load_dataset

# ── 1. ShareGPT ──────────────────────────────────────────────────────────────
print(f"Loading ShareGPT ({N_SG} samples)...", flush=True)
sg_ds = load_dataset(
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    data_files="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
    split="train",
)
sg_ds = sg_ds.shuffle(seed=SEED).select(range(min(N_SG, len(sg_ds))))

role_map = {"human": "user", "gpt": "assistant", "system": "system"}

def sg_to_messages(item):
    msgs = []
    for turn in item.get("conversations", []):
        role = role_map.get(turn.get("from", ""), "user")
        content = turn.get("value", "").strip()
        if content:
            msgs.append({"role": role, "content": content})
    return msgs

samples = []
skipped = 0
for item in sg_ds:
    msgs = sg_to_messages(item)
    if len(msgs) >= 2:
        samples.append({"messages": msgs})
    else:
        skipped += 1
print(f"  ShareGPT: {len(samples)} samples (skipped {skipped} empty)", flush=True)

# ── 2. UltraChat-200K ────────────────────────────────────────────────────────
print(f"Loading UltraChat ({N_UC} samples)...", flush=True)
uc_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
uc_ds = uc_ds.shuffle(seed=SEED).select(range(min(N_UC, len(uc_ds))))

uc_samples = []
skipped = 0
for item in uc_ds:
    msgs = item.get("messages", [])
    # Normalise to plain list of dicts (drop any extra keys)
    clean = [{"role": m["role"], "content": m["content"]} for m in msgs if m.get("content", "").strip()]
    if len(clean) >= 2:
        uc_samples.append({"messages": clean})
    else:
        skipped += 1
print(f"  UltraChat: {len(uc_samples)} samples (skipped {skipped} empty)", flush=True)
samples.extend(uc_samples)

# ── 3. Open-PerfectBlend ─────────────────────────────────────────────────────
print(f"Loading Open-PerfectBlend ({N_PB} samples)...", flush=True)
pb_ds = load_dataset("mlabonne/open-perfectblend", split="train")
pb_ds = pb_ds.shuffle(seed=SEED).select(range(min(N_PB, len(pb_ds))))

pb_samples = []
skipped = 0
for item in pb_ds:
    # open-perfectblend uses "conversations" with "from"/"value" or "messages"
    if "messages" in item:
        msgs = [{"role": m["role"], "content": m["content"]}
                for m in item["messages"] if m.get("content", "").strip()]
    elif "conversations" in item:
        msgs = []
        for turn in item["conversations"]:
            role = role_map.get(turn.get("from", ""), "user")
            content = turn.get("value", "").strip()
            if content:
                msgs.append({"role": role, "content": content})
    else:
        skipped += 1
        continue
    if len(msgs) >= 2:
        pb_samples.append({"messages": msgs})
    else:
        skipped += 1
print(f"  PerfectBlend: {len(pb_samples)} samples (skipped {skipped} empty)", flush=True)
samples.extend(pb_samples)

# ── 4. Shuffle & write ───────────────────────────────────────────────────────
random.shuffle(samples)
print(f"\nTotal: {len(samples)} samples — writing to {OUTPUT}", flush=True)

with open(OUTPUT, "w") as f:
    for item in samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done. {len(samples)} lines written.", flush=True)
PYEOF

N=$(wc -l < "$OUTPUT")
echo ""
echo "Dataset ready: $OUTPUT ($N samples)"
