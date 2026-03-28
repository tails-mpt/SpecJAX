"""
GLM-4.7-FP8 forward pass in pure JAX — frozen target model.

Architecture: Glm4MoeForCausalLM
  vocab_size=151552, hidden_size=5120, num_hidden_layers=92,
  num_attention_heads=96, num_key_value_heads=8 (GQA 12:1),
  head_dim=128, attention_bias=True (Q/K/V only),
  use_qk_norm=True, partial_rotary_factor=0.5,
  rope_theta=1_000_000, rms_norm_eps=1e-5,
  n_routed_experts=160, n_shared_experts=1, num_experts_per_tok=8,
  moe_intermediate_size=1536, routed_scaling_factor=2.5,
  first_k_dense_replace=3 (layers 0-2 dense, 3-91 MoE)

Key differences from GLM-Flash / GLM-5:
  - Standard GQA attention (not MLA with LoRA-compressed KV)
  - QK normalization (RMSNorm per-head before RoPE)
  - Partial RoPE: only first 50% of head_dim (64 of 128) get rotation,
    standard layout (not interleaved)
  - Channel-wise FP8 quantization (compressed-tensors format):
    one scale per output row, not block-wise
  - MTP layer skipped (present in config but not implemented in HF model)
"""

import gc
import json
import logging
import os
import re
from typing import Optional

import ml_dtypes  # must import before safetensors to register float8 dtypes with numpy
import numpy as np
# Register FP8 dtypes with numpy so safetensors can deserialize them
np.float8_e4m3fn = ml_dtypes.float8_e4m3fn
np.float8_e5m2 = ml_dtypes.float8_e5m2

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding
from safetensors.numpy import load_file

from specjax.ops.norm import rms_norm
from specjax.ops.rope import build_rope_freqs, rotate_half
from specjax.ops.moe import moe_forward as _moe_forward, mlp_forward
from specjax.ops.fp8 import dequant_fp8_channel, dequant_expert_jit
from specjax.ops.loading import discover_shards

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclass (plain Python — no JAX dependency)
# ---------------------------------------------------------------------------

class GLM47Config:
    vocab_size: int = 151_552
    hidden_size: int = 5_120
    num_hidden_layers: int = 92
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = True
    use_qk_norm: bool = True
    partial_rotary_factor: float = 0.5
    rope_theta: float = 1_000_000.0
    n_routed_experts: int = 160
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1_536
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 3
    intermediate_size: int = 12_288       # dense MLP intermediate (layers 0-2)
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 202_752

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: dict) -> "GLM47Config":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# ---------------------------------------------------------------------------
# FP8 loading helpers (channel-wise, compressed-tensors format)
# ---------------------------------------------------------------------------

def _is_weight_scale(key: str) -> bool:
    """Check if a key is a scale tensor."""
    return key.endswith(".weight_scale") or key.endswith("_scale") or key.endswith(".weight_scale_inv")


def _is_fp8(v_np: np.ndarray) -> bool:
    """Check if numpy array is FP8 quantized."""
    return (
        v_np.dtype == np.uint8
        or str(v_np.dtype) == "float8_e4m3fn"
        or (hasattr(ml_dtypes, "float8_e4m3fn")
            and v_np.dtype == ml_dtypes.float8_e4m3fn)
    )


def _dequant_fp8_1d(w_fp8: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize a 1D FP8 tensor."""
    w_f32 = w_fp8.astype(np.float32)
    if scale.ndim == 0 or scale.size == 1:
        return (w_f32 * float(scale)).astype(ml_dtypes.bfloat16)
    return (w_f32 * scale).astype(ml_dtypes.bfloat16)


def _should_skip_key(key: str) -> bool:
    """Skip MTP weights — not used in training forward pass."""
    return key.startswith("model.nextn_predict_layers")


def _find_scale(k: str, scales: dict) -> Optional[np.ndarray]:
    """Find scale tensor for a given weight key."""
    for suffix in [".weight_scale", ".weight_scale_inv", "_scale"]:
        scale_key = k.replace(".weight", suffix) if ".weight" in k else k + suffix
        scale = scales.get(scale_key)
        if scale is not None:
            return scale
    return None


def _dequant_tensor(k: str, v_np: np.ndarray, scales: dict) -> np.ndarray:
    """Dequantize a single tensor, returning bf16 numpy array."""
    if _is_fp8(v_np):
        scale = _find_scale(k, scales)

        if scale is None:
            logger.warning(
                f"  WARNING: No scale found for FP8 weight {k}, "
                f"casting directly to bf16"
            )
            return v_np.astype(np.float32).astype(ml_dtypes.bfloat16)
        elif v_np.ndim == 2:
            return dequant_fp8_channel(v_np, scale)
        elif v_np.ndim == 1:
            return _dequant_fp8_1d(v_np, scale)
        else:
            logger.warning(
                f"  WARNING: FP8 weight {k} has {v_np.ndim}D shape "
                f"{v_np.shape}, casting directly"
            )
            return v_np.astype(np.float32).astype(ml_dtypes.bfloat16)
    else:
        return np.array(v_np, dtype=ml_dtypes.bfloat16)


# ---------------------------------------------------------------------------
# Weight loading (streaming, per-layer expert stacking)
# ---------------------------------------------------------------------------

def load_params(model_path: str, mesh=None) -> tuple[dict, "GLM47Config"]:
    """
    Load GLM-4.7-FP8 weights from safetensors shards.

    Memory strategy (23 GB/chip vs 32 GB budget on v4-32):
      - MoE expert weights are kept in FP8 on TPU with separate f32 scales,
        then dequantized just-in-time during the forward pass. This halves
        the 42 GB/chip MoE storage to ~21 GB/chip.
      - Non-expert weights (attention, norms, embed, shared experts) are
        small enough to store in bf16 (~2 GB/chip).
      - Each tensor is sharded onto TPU immediately via
        jax.make_array_from_callback (no multi-host broadcast).

    Returns:
        params: flat dict {name: jax.Array} sharded on mesh
        config: GLM47Config
    """
    from specjax.models.sharding import _pspec_for

    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = GLM47Config.from_dict(cfg_dict)

    shard_files = discover_shards(model_path)

    logger.warning(
        f"Loading {len(shard_files)} safetensors shard(s) from {model_path}"
    )

    params: dict = {}           # final sharded params
    expert_buf: dict = {}       # temporary: per-expert raw arrays (FP8 or bf16)
    expert_scale_buf: dict = {} # temporary: per-expert scale arrays
    all_scales: dict = {}       # FP8 scale tensors (small, kept in RAM)
    skipped_keys = 0

    def _place_on_device(name: str, arr_np: np.ndarray):
        """Shard a numpy array onto mesh and store in params."""
        if mesh is not None:
            pspec = _pspec_for(name, arr_np)
            sharding = NamedSharding(mesh, pspec)
            global_shape = arr_np.shape
            def _cb(index):
                return arr_np[index]
            params[name] = jax.make_array_from_callback(
                global_shape, sharding, _cb
            )
        else:
            params[name] = jnp.array(arr_np)

    expert_pattern = re.compile(
        r"(model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )

    for shard_idx, shard_name in enumerate(shard_files):
        shard_path = os.path.join(model_path, shard_name)
        shard = load_file(shard_path)

        # Collect scale tensors first (tiny — keep all in RAM)
        for k, v in shard.items():
            if _is_weight_scale(k):
                all_scales[k] = np.array(v, dtype=np.float32)

        for k, v in shard.items():
            if _is_weight_scale(k):
                continue
            if _should_skip_key(k):
                skipped_keys += 1
                continue

            v_np = np.array(v)
            del v

            # Check if this is a per-expert MoE weight
            m = expert_pattern.match(k)
            if m:
                # MoE expert weights: keep as uint8 on TPU (1 byte/element)
                # with separate f32 channel-wise scales. Dequantized JIT in
                # the forward pass.
                if _is_fp8(v_np):
                    expert_buf[k] = v_np.view(np.uint8)
                    scale = _find_scale(k, all_scales)
                    if scale is not None:
                        expert_scale_buf[k] = scale
                else:
                    expert_buf[k] = np.array(v_np, dtype=ml_dtypes.bfloat16)

                del v_np
                layer_pfx = m.group(1)
                layer_idx = int(m.group(2))

                # Check if all experts for this layer are collected
                layer_keys = [
                    ek for ek in expert_buf
                    if ek.startswith(f"{layer_pfx}.")
                ]
                if len(layer_keys) == config.n_routed_experts * 3:
                    E = config.n_routed_experts

                    # Stack gate, up, down projections
                    gates_raw = np.stack(
                        [expert_buf[f"{layer_pfx}.{e}.gate_proj.weight"] for e in range(E)],
                        axis=0,
                    )
                    ups_raw = np.stack(
                        [expert_buf[f"{layer_pfx}.{e}.up_proj.weight"] for e in range(E)],
                        axis=0,
                    )
                    gate_up_raw = np.concatenate([gates_raw, ups_raw], axis=1)
                    del gates_raw, ups_raw
                    _place_on_device(f"{layer_pfx}.gate_up_proj", gate_up_raw)
                    del gate_up_raw

                    downs_raw = np.stack(
                        [expert_buf[f"{layer_pfx}.{e}.down_proj.weight"] for e in range(E)],
                        axis=0,
                    )
                    _place_on_device(f"{layer_pfx}.down_proj", downs_raw)
                    del downs_raw

                    # Stack and place scales (channel-wise: [out_features, 1] per expert)
                    gate_scale_key = f"{layer_pfx}.0.gate_proj.weight"
                    if gate_scale_key in expert_scale_buf:
                        gate_scales = np.stack([
                            expert_scale_buf[f"{layer_pfx}.{e}.gate_proj.weight"]
                            for e in range(E)
                        ], axis=0)
                        up_scales = np.stack([
                            expert_scale_buf[f"{layer_pfx}.{e}.up_proj.weight"]
                            for e in range(E)
                        ], axis=0)
                        gate_up_scales = np.concatenate([gate_scales, up_scales], axis=1)
                        del gate_scales, up_scales
                        _place_on_device(f"{layer_pfx}.gate_up_proj.scale", gate_up_scales)
                        del gate_up_scales

                        down_scales = np.stack([
                            expert_scale_buf[f"{layer_pfx}.{e}.down_proj.weight"]
                            for e in range(E)
                        ], axis=0)
                        _place_on_device(f"{layer_pfx}.down_proj.scale", down_scales)
                        del down_scales

                        for ek in layer_keys:
                            expert_scale_buf.pop(ek, None)

                    for ek in layer_keys:
                        del expert_buf[ek]

                    logger.warning(f"  Stacked + sharded MoE experts for layer {layer_idx}")
                    gc.collect()
            elif ".mlp.shared_experts." in k:
                arr_bf16 = _dequant_tensor(k, v_np, all_scales)
                del v_np
                new_key = k.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
                _place_on_device(new_key, arr_bf16)
                del arr_bf16
            else:
                arr_bf16 = _dequant_tensor(k, v_np, all_scales)
                del v_np
                _place_on_device(k, arr_bf16)
                del arr_bf16

        logger.warning(f"  Loaded {shard_name}  ({len(shard)} tensors)  [{shard_idx+1}/{len(shard_files)}]")
        del shard
        gc.collect()

    if expert_buf:
        logger.warning(f"  WARNING: {len(expert_buf)} unbuffered expert tensors remaining")
        for k, v in expert_buf.items():
            _place_on_device(k, v)
        expert_buf.clear()

    del all_scales, expert_scale_buf
    gc.collect()

    logger.warning(
        f"Total parameters: {len(params)} "
        f"(skipped {skipped_keys} MTP tensors)"
    )

    return params, config


# ---------------------------------------------------------------------------
# GQA Attention (standard Grouped Query Attention, NOT MLA)
# ---------------------------------------------------------------------------

def gqa_forward(
    hidden_states: jnp.ndarray,        # [B, T, H]
    attention_mask: jnp.ndarray,       # [B, T]  (1=attend, 0=mask)
    cos: jnp.ndarray,                  # [T, rotary_dim]
    sin: jnp.ndarray,                  # [T, rotary_dim]
    p: dict,                           # weight dict for this attention layer
    cfg: GLM47Config,
) -> jnp.ndarray:                      # [B, T, H]
    """
    Standard GQA with QK normalization and partial RoPE.

    Attention projections have bias on Q/K/V (not O).
    QK norm: RMSNorm per-head before RoPE.
    Partial RoPE: only first rotary_dim dims of head_dim get rotation.
    """
    B, T, H = hidden_states.shape
    n_h = cfg.num_attention_heads       # 96
    n_kv = cfg.num_key_value_heads      # 8
    d = cfg.head_dim                    # 128
    rotary_dim = int(d * cfg.partial_rotary_factor)  # 64

    # -- Q/K/V projections (with bias) --
    q = hidden_states @ p["q_proj.weight"].T
    k = hidden_states @ p["k_proj.weight"].T
    v = hidden_states @ p["v_proj.weight"].T

    if cfg.attention_bias:
        q = q + p["q_proj.bias"]
        k = k + p["k_proj.bias"]
        v = v + p["v_proj.bias"]

    q = q.reshape(B, T, n_h, d)
    k = k.reshape(B, T, n_kv, d)
    v = v.reshape(B, T, n_kv, d)

    # -- QK normalization (RMSNorm per-head, before RoPE) --
    if cfg.use_qk_norm:
        q = rms_norm(q, p["q_norm.weight"], cfg.rms_norm_eps)
        k = rms_norm(k, p["k_norm.weight"], cfg.rms_norm_eps)

    # -- Partial RoPE (standard layout, first rotary_dim dims only) --
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_rot = q_rot.transpose(0, 2, 1, 3)
    k_rot = k_rot.transpose(0, 2, 1, 3)

    cos_bc = cos[None, None, :, :]
    sin_bc = sin[None, None, :, :]
    q_rot = q_rot * cos_bc + rotate_half(q_rot) * sin_bc
    k_rot = k_rot * cos_bc + rotate_half(k_rot) * sin_bc

    q_rot = q_rot.transpose(0, 2, 1, 3)
    k_rot = k_rot.transpose(0, 2, 1, 3)

    q = jnp.concatenate([q_rot, q_pass], axis=-1)
    k = jnp.concatenate([k_rot, k_pass], axis=-1)

    # -- Scaled dot-product attention with GQA --
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    n_groups = n_h // n_kv
    k_t = jnp.repeat(k_t, n_groups, axis=1)
    v_t = jnp.repeat(v_t, n_groups, axis=1)

    scale = d ** -0.5
    attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

    causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    attn_weights = jnp.where(causal_mask[None, None], attn_weights, jnp.finfo(jnp.float32).min)

    pad_mask = attention_mask[:, None, None, :].astype(jnp.bool_)
    attn_weights = jnp.where(pad_mask, attn_weights, jnp.finfo(jnp.float32).min)

    attn_probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(hidden_states.dtype)

    attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v_t)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, n_h * d)

    # o_proj: NO bias
    return attn_output @ p["o_proj.weight"].T


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

def decoder_layer_forward(
    hidden_states: jnp.ndarray,
    attention_mask: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    layer_idx: int,
    params: dict,
    cfg: GLM47Config,
) -> jnp.ndarray:
    pfx = f"model.layers.{layer_idx}"

    # Pre-norm + GQA attention + residual
    attn_in = rms_norm(hidden_states, params[f"{pfx}.input_layernorm.weight"], cfg.rms_norm_eps)

    attn_p = {k.removeprefix(f"{pfx}.self_attn."): v
              for k, v in params.items()
              if k.startswith(f"{pfx}.self_attn.")}
    attn_out = gqa_forward(attn_in, attention_mask, cos, sin, attn_p, cfg)
    hidden_states = hidden_states + attn_out

    # Pre-norm + MLP/MoE + residual
    mlp_in = rms_norm(hidden_states, params[f"{pfx}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

    if layer_idx < cfg.first_k_dense_replace:
        # Dense MLP (layers 0-2)
        mlp_p = {k.removeprefix(f"{pfx}.mlp."): v
                 for k, v in params.items()
                 if k.startswith(f"{pfx}.mlp.")}
        mlp_out = mlp_forward(mlp_in, mlp_p)
    else:
        # Sparse MoE (layers 3-91)
        moe_p = {k.removeprefix(f"{pfx}.mlp."): v
                 for k, v in params.items()
                 if k.startswith(f"{pfx}.mlp.")}
        mlp_out = _moe_forward(mlp_in, moe_p, cfg.num_experts_per_tok,
                               cfg.routed_scaling_factor, cfg.norm_topk_prob)

    return hidden_states + mlp_out


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------

def glm47_forward(
    input_ids: jnp.ndarray,        # [B, T]  int32
    attention_mask: jnp.ndarray,   # [B, T]  int32/bool
    params: dict,
    cfg: GLM47Config,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full GLM-4.7-FP8 forward pass (no KV cache — training mode).

    Returns:
        last_hidden_state    [B, T, H]
        embed_tokens_weight  [vocab_size, H]
        aux_hidden_states    [B, T, 3*H]  -- from layers {1, 45, 88}
        target_logits        [B, T, vocab_size]  -- for KL divergence loss
    """
    B, T = input_ids.shape
    num_layers = cfg.num_hidden_layers

    # Layer indices for multi-layer feature fusion (SpecForge convention)
    # For GLM-4.7-FP8 (92 layers): {1, 45, 88}
    aux_layer_indices = {1, num_layers // 2 - 1, num_layers - 4}

    embed_w = params["model.embed_tokens.weight"]
    hidden_states = embed_w[input_ids]

    # Build RoPE tables for partial rotation (rotary_dim = head_dim * 0.5 = 64)
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    cos, sin = build_rope_freqs(rotary_dim, T, cfg.rope_theta)
    cos = cos[:T]
    sin = sin[:T]

    aux_hidden = []
    for layer_idx in range(num_layers):
        hidden_states = decoder_layer_forward(
            hidden_states, attention_mask, cos, sin, layer_idx, params, cfg
        )
        if layer_idx in aux_layer_indices:
            aux_hidden.append(hidden_states)

    hidden_states = rms_norm(hidden_states, params["model.norm.weight"], cfg.rms_norm_eps)

    aux_hidden_states = jnp.concatenate(aux_hidden, axis=-1)

    target_logits = hidden_states @ params["lm_head.weight"].T

    return hidden_states, embed_w, aux_hidden_states, target_logits
