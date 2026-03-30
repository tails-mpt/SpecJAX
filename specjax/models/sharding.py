"""
SPMD Mesh and PartitionSpec helpers for GLM-4.7-Flash on TPU.

Mirrors the xs.mark_sharding strategy from the PyTorch/XLA implementation:
  - 2D weights [out, in]: shard output dim → PartitionSpec("tp", None)
  - 3D MoE gate/up [E, 2I, D]: shard output dim → PartitionSpec(None, "tp", None)
  - 3D MoE down [E, D, I]: shard input dim → PartitionSpec(None, None, "tp")
  - 1D/scalar: replicated → PartitionSpec(None,)

Usage:
    from specjax.models.sharding import make_mesh, shard_params
    mesh = make_mesh()
    sharded_params = shard_params(params, mesh)
"""

import logging
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

logger = logging.getLogger(__name__)

PyTree = Any


def make_mesh(tp: int = 4) -> Mesh:
    """
    Create an SPMD mesh across all available TPU chips.

    Args:
        tp: Tensor-parallel size (default 4). Must evenly divide the total
            number of chips. Common values: 4, 8, 16.

    Single-host (n <= tp): 1D tensor-parallel mesh with axis "tp".
    Multi-host (n > tp, n divisible by tp): 2D data+tensor-parallel mesh with
        axes ("dp", "tp"), where dp=n//tp.  The device array is shaped
        [dp, tp] so that mesh[i, :] are the tp chips belonging to DP rank i.

    GLM-4.7-Flash has 20 attention heads — valid TP sizes are 1,2,4,5,10,20.
    TP=4 is used in most configs; larger models (70B+, 229B MoE) use TP=8 or 16.
    """
    devices = jax.devices()
    n = len(devices)
    if n % tp != 0:
        raise ValueError(
            f"Number of devices ({n}) must be divisible by tp ({tp})"
        )
    if n > tp:
        dp = n // tp
        device_arr = np.array(devices).reshape(dp, tp)
        mesh = Mesh(device_arr, axis_names=("dp", "tp"))
        logger.warning(f"SPMD mesh: {n} chips  |  2D mesh (dp={dp}, tp={tp})")
    else:
        device_arr = np.array(devices).reshape(n)
        mesh = Mesh(device_arr, axis_names=("tp",))
        logger.warning(f"SPMD mesh: {n} chips  |  1D mesh (tp={n})")
    return mesh


def batch_sharding(mesh: Mesh) -> NamedSharding:
    """
    Return a sharding for batch tensors [B, T]:
      - If the mesh has a "dp" axis: shard leading dim across dp, replicate T.
      - Otherwise: fully replicated (single-host mode).
    """
    if "dp" in mesh.axis_names:
        return NamedSharding(mesh, PartitionSpec("dp", None))
    return NamedSharding(mesh, PartitionSpec(None, None))


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec())


def tp_col(mesh: Mesh) -> NamedSharding:
    """Column-parallel: shard output dim of [out, in] weight."""
    return NamedSharding(mesh, PartitionSpec("tp", None))


def tp_row(mesh: Mesh) -> NamedSharding:
    """Row-parallel: shard input dim of [out, in] weight."""
    return NamedSharding(mesh, PartitionSpec(None, "tp"))


def tp_moe_gate_up(mesh: Mesh) -> NamedSharding:
    """MoE gate/up proj [E, 2I, D]: shard output intermediate dim."""
    return NamedSharding(mesh, PartitionSpec(None, "tp", None))


def tp_moe_down(mesh: Mesh) -> NamedSharding:
    """MoE down proj [E, D, I]: shard input intermediate dim (matches gate/up output shard)."""
    return NamedSharding(mesh, PartitionSpec(None, None, "tp"))


def _pspec_for(name: str, arr: jnp.ndarray) -> PartitionSpec:
    """
    Return the appropriate PartitionSpec for a parameter given its name and shape.
    Matches the PyTorch/XLA mark_sharding rules from the original implementation.
    """
    ndim = arr.ndim
    numel = arr.size

    if numel <= 1_000_000:
        # Small params: replicated (norms, biases, small embeddings)
        return PartitionSpec(*([None] * ndim))

    if ndim == 1:
        return PartitionSpec(None)

    if ndim == 2:
        # [out_features, in_features] → shard output dim
        return PartitionSpec("tp", None)

    if ndim == 3:
        # MoE experts: [E, ?, ?]
        if "down_proj" in name:
            # [E, out_dim, in_dim] — shard in_dim to match gate/up output shard
            return PartitionSpec(None, None, "tp")
        else:
            # gate_up_proj [E, 2I, D] — shard output dim
            return PartitionSpec(None, "tp", None)

    # Fallback: replicate
    return PartitionSpec(*([None] * ndim))


def shard_params(params: dict, mesh: Mesh) -> dict:
    """
    Shard a flat {name: jnp.ndarray} parameter dict onto the mesh.

    Large parameters (>1M elements) are sharded; smaller ones are replicated.
    Returns a new dict with each array placed on the appropriate devices.
    """
    sharded = {}
    total = len(params)
    for i, (name, arr) in enumerate(params.items()):
        pspec = _pspec_for(name, arr)
        sharding = NamedSharding(mesh, pspec)
        sharded[name] = jax.device_put(arr, sharding)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.warning(f"  Sharding params: {i + 1}/{total}")
    return sharded
