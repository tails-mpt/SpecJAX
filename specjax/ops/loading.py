"""Shared weight loading helpers for SpecJAX target models."""

import json
import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)


def discover_shards(model_path: str) -> list[str]:
    """Discover safetensors shard filenames from model.safetensors.index.json.

    Returns a sorted list of shard filenames (not full paths).
    Falls back to ["model.safetensors"] if no index file exists.
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        return sorted(set(index["weight_map"].values()))
    return ["model.safetensors"]


def stack_moe_experts(
    params: dict,
    n_routed_experts: int,
    rename_shared: bool = True,
) -> dict:
    """
    Stack per-expert weight tensors into batched [E, ...] arrays.

    The HuggingFace checkpoint stores each expert separately:
      model.layers.{i}.mlp.experts.{e}.gate_proj.weight  [I, D]
      model.layers.{i}.mlp.experts.{e}.up_proj.weight    [I, D]
      model.layers.{i}.mlp.experts.{e}.down_proj.weight  [D, I]

    We replace these with:
      model.layers.{i}.mlp.experts.gate_up_proj  [E, 2I, D]  (gate and up concatenated)
      model.layers.{i}.mlp.experts.down_proj     [E, D, I]

    If rename_shared=True, renames shared_experts.* -> shared_expert.* to match moe_forward.
    """
    E = n_routed_experts
    result = {}
    stacked_layers = set()

    for key, val in params.items():
        # Check for per-expert keys: model.layers.{i}.mlp.experts.{e}.{proj}.weight
        m = re.match(
            r"(model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$",
            key,
        )
        if m:
            layer_pfx = m.group(1)
            layer_idx = int(m.group(2))
            if layer_idx not in stacked_layers:
                gates = np.stack(
                    [params[f"{layer_pfx}.{e}.gate_proj.weight"] for e in range(E)],
                    axis=0,
                )   # [E, I, D]
                ups = np.stack(
                    [params[f"{layer_pfx}.{e}.up_proj.weight"] for e in range(E)],
                    axis=0,
                )   # [E, I, D]
                downs = np.stack(
                    [params[f"{layer_pfx}.{e}.down_proj.weight"] for e in range(E)],
                    axis=0,
                )   # [E, D, I]
                result[f"{layer_pfx}.gate_up_proj"] = np.concatenate(
                    [gates, ups], axis=1
                )  # [E, 2I, D]
                result[f"{layer_pfx}.down_proj"] = downs  # [E, D, I]
                stacked_layers.add(layer_idx)
                logger.warning(f"  Stacked MoE experts for layer {layer_idx}")
            continue

        # Rename shared_experts.* -> shared_expert.*
        if rename_shared and ".mlp.shared_experts." in key:
            new_key = key.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
            result[new_key] = val
            continue

        result[key] = val

    return result
