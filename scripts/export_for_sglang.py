#!/usr/bin/env python3
"""
Export / patch a SpecJAX Eagle3 checkpoint for sglang-jax compatibility.

Usage:
    python scripts/export_for_sglang.py /path/to/checkpoint [--output /path/to/output]

If --output is not specified, patches the checkpoint in-place.

What this does:
  1. Validates the checkpoint (safetensors keys, shapes, config fields)
  2. Patches config.json to ensure sglang-jax required fields are present
     (tie_word_embeddings, architectures, etc.)
  3. Reports any remaining issues
"""

import argparse
import json
import os
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specjax.models.draft.eagle3 import validate_eagle3_checkpoint


def patch_config(config_path: str) -> list[str]:
    """Patch config.json for sglang-jax compatibility. Returns list of changes made."""
    with open(config_path) as f:
        cfg = json.load(f)

    changes = []

    # Ensure tie_word_embeddings is explicitly False
    if cfg.get("tie_word_embeddings") is not False:
        cfg["tie_word_embeddings"] = False
        changes.append("Set tie_word_embeddings: false")

    # Ensure architectures uses base name (sglang-jax appends Eagle3 automatically)
    archs = cfg.get("architectures", [])
    if archs and archs[0] == "LlamaForCausalLMEagle3":
        cfg["architectures"] = ["LlamaForCausalLM"]
        changes.append("Changed architectures to LlamaForCausalLM (sglang-jax auto-appends Eagle3)")

    # Ensure num_hidden_layers is present
    if "num_hidden_layers" not in cfg:
        cfg["num_hidden_layers"] = 1
        changes.append("Added num_hidden_layers: 1")

    if changes:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    return changes


def main():
    parser = argparse.ArgumentParser(description="Export SpecJAX checkpoint for sglang-jax")
    parser.add_argument("checkpoint_dir", help="Path to the SpecJAX checkpoint directory")
    parser.add_argument("--output", "-o", help="Output directory (default: patch in-place)")
    args = parser.parse_args()

    src = args.checkpoint_dir
    dst = args.output or src

    if not os.path.isdir(src):
        print(f"Error: {src} is not a directory")
        sys.exit(1)

    # Copy if output specified
    if dst != src:
        print(f"Copying {src} -> {dst}")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Patch config
    config_path = os.path.join(dst, "config.json")
    if os.path.exists(config_path):
        changes = patch_config(config_path)
        if changes:
            print(f"Patched config.json:")
            for c in changes:
                print(f"  - {c}")
        else:
            print("config.json already compatible")
    else:
        print("Warning: no config.json found")

    # Validate
    print("\nValidating checkpoint...")
    warnings = validate_eagle3_checkpoint(dst)
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
        sys.exit(1)
    else:
        print("Checkpoint is compatible with sglang-jax")


if __name__ == "__main__":
    main()
