"""
Llama-70B FP8 target model forward pass in pure JAX — frozen target model.

Covers Llama-3.3-70B-Instruct-FP8 (nvidia/Llama-3.3-70B-Instruct-FP8):
  Standard GQA + RoPE + SiLU-gated MLP — same architecture as Llama 3.x
  but with per-tensor FP8 quantized linear weights.

Llama-3.3-70B-Instruct-FP8 reference config:
  vocab_size=128256, hidden_size=8192, num_hidden_layers=80,
  num_attention_heads=64, num_key_value_heads=8 (GQA 8:1),
  head_dim=128, intermediate_size=28672,
  rope_theta=500000.0, rms_norm_eps=1e-5, tie_word_embeddings=False

FP8 format (nvidia compressed-tensors, per-tensor):
  Each linear weight is float8_e4m3fn with a scalar float32 weight_scale.
  Dequant: bf16_weight = fp8_weight * weight_scale
  input_scale / k_scale / v_scale are for inference KV cache — ignored here.

Memory strategy for TP=4 on v4-32 (33 GB/chip):
  Linear weights stored as uint8 on TPU (1 byte/param) with separate f32
  scalar scales. ~17.1 GB/chip for 80 layers. Dequantized JIT during forward.
  Embeddings + norms in bf16 (~1 GB/chip). Total ~18 GB/chip, leaving ~15 GB
  for draft model, optimizer, and activations.

Design:
  - Weights stored as flat dict {str: jnp.ndarray} (uint8 for linears, bf16 for rest).
  - All functions are pure JAX — no Flax modules, no mutable state.
  - Model is frozen: no gradients flow through it.
  - Returns (last_hidden_state, embed_tokens_weight, aux_hidden_states,
    target_logits) for the Eagle3 draft model training pipeline.
"""

import gc
import json
import logging
import os

import ml_dtypes  # must import before safetensors to register float8 dtypes
import numpy as np
np.float8_e4m3fn = ml_dtypes.float8_e4m3fn
np.float8_e5m2 = ml_dtypes.float8_e5m2

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding
from safetensors.numpy import load_file

from specjax.ops.norm import rms_norm
from specjax.ops.rope import build_rope_freqs, rotate_half
from specjax.ops.moe import mlp_forward
from specjax.ops.loading import discover_shards

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config (plain Python — no JAX dependency)
# ---------------------------------------------------------------------------

class LlamaFP8Config:
    vocab_size: int = 128_256
    hidden_size: int = 8_192
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 28_672
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 131_072
    tie_word_embeddings: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: dict) -> "LlamaFP8Config":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# ---------------------------------------------------------------------------
# FP8 helpers
# ---------------------------------------------------------------------------

def _is_fp8(v_np: np.ndarray) -> bool:
    return (
        v_np.dtype == np.uint8
        or str(v_np.dtype) == "float8_e4m3fn"
        or (hasattr(ml_dtypes, "float8_e4m3fn")
            and v_np.dtype == ml_dtypes.float8_e4m3fn)
    )


def _is_scale_key(key: str) -> bool:
    return (key.endswith(".weight_scale")
            or key.endswith(".input_scale")
            or key.endswith(".k_scale")
            or key.endswith(".v_scale"))


def _dequant_jit(w_uint8: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """JIT-friendly FP8->bf16 dequantization (per-tensor or channel-wise).

    w_uint8: [out, in] stored as uint8 (raw FP8 e4m3fn bytes)
    scale: scalar (per-tensor, nvidia format),
           [out, 1] (channel-wise 2D, compressed-tensors),
           or [out] (channel-wise 1D)

    On XLA/TPU: bitcast uint8 -> float8_e4m3fn -> f32, multiply by scale, cast to bf16.
    """
    w_f32 = lax.bitcast_convert_type(w_uint8, jnp.float8_e4m3fn).astype(jnp.float32)
    scale_f32 = scale.astype(jnp.float32)
    if scale_f32.ndim == 2:
        # [out, 1] column vector — broadcasts directly with [out, in]
        return (w_f32 * scale_f32).astype(jnp.bfloat16)
    if scale_f32.ndim == 1 and scale_f32.size > 1:
        # [out] 1D — add trailing dim for broadcast
        return (w_f32 * scale_f32[:, None]).astype(jnp.bfloat16)
    # scalar
    return (w_f32 * scale_f32).astype(jnp.bfloat16)


# ---------------------------------------------------------------------------
# Weight loading (streaming, FP8 weights kept as uint8 on TPU)
# ---------------------------------------------------------------------------

def load_params(model_path: str, mesh=None) -> tuple[dict, "LlamaFP8Config"]:
    """
    Load Llama-70B-FP8 weights from safetensors shards.

    Linear weights are stored as uint8 on TPU (1 byte/element) with separate
    float32 scalar scales. Dequantized JIT in the forward pass.
    Non-linear weights (embeddings, norms) are stored in bf16.

    Returns:
        params: flat dict {name: jax.Array} sharded on mesh
        config: LlamaFP8Config
    """
    from specjax.models.sharding import _pspec_for

    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = LlamaFP8Config.from_dict(cfg_dict)

    shard_files = discover_shards(model_path)
    logger.warning(f"Loading {len(shard_files)} safetensors shard(s) from {model_path}")

    params: dict = {}
    all_scales: dict = {}

    def _place_on_device(name: str, arr_np: np.ndarray, pspec_override=None):
        if mesh is not None:
            pspec = pspec_override if pspec_override is not None else _pspec_for(name, arr_np)
            sharding = NamedSharding(mesh, pspec)
            def _cb(index):
                return arr_np[index]
            params[name] = jax.make_array_from_callback(
                arr_np.shape, sharding, _cb
            )
        else:
            params[name] = jnp.array(arr_np)

    for shard_idx, shard_name in enumerate(shard_files):
        shard_path = os.path.join(model_path, shard_name)
        shard = load_file(shard_path)

        # First pass: collect all scale tensors (tiny, kept in RAM)
        for k, v in shard.items():
            if _is_scale_key(k):
                all_scales[k] = np.array(v, dtype=np.float32)

        # Second pass: process weight tensors
        for k, v in shard.items():
            if _is_scale_key(k):
                continue

            v_np = np.array(v)
            del v

            if _is_fp8(v_np) and v_np.ndim >= 2:
                # FP8 linear weight: store as uint8 on TPU
                arr_uint8 = v_np.view(np.uint8)
                _place_on_device(k, arr_uint8)
                del arr_uint8

                # Find and place the corresponding weight_scale
                scale_key = k.replace(".weight", ".weight_scale")
                scale = all_scales.get(scale_key)
                if scale is not None:
                    # Channel-wise scales must be sharded on the same axis
                    # as the weight's output dim (TP axis).
                    if scale.size > 1:
                        from jax.sharding import PartitionSpec
                        if scale.ndim == 2:
                            # [out, 1] column vector
                            _place_on_device(scale_key, scale,
                                             pspec_override=PartitionSpec("tp", None))
                        elif scale.ndim == 1:
                            # [out] 1D
                            _place_on_device(scale_key, scale,
                                             pspec_override=PartitionSpec("tp"))
                    else:
                        _place_on_device(scale_key, scale)
                else:
                    logger.warning(f"  WARNING: No scale found for FP8 weight {k}")
                    _place_on_device(scale_key, np.array(1.0, dtype=np.float32))
            else:
                # Non-FP8: embeddings, norms, lm_head — store as bf16
                arr_bf16 = np.array(v_np, dtype=ml_dtypes.bfloat16)
                _place_on_device(k, arr_bf16)
                del arr_bf16

            del v_np

        logger.warning(
            f"  Loaded {shard_name}  ({len(shard)} tensors)  "
            f"[{shard_idx + 1}/{len(shard_files)}]"
        )
        del shard
        gc.collect()

    del all_scales
    gc.collect()

    logger.warning(f"Total parameters: {len(params)}")
    return params, config


# ---------------------------------------------------------------------------
# GQA Attention with JIT FP8 dequant (no bias, no QK norm, full RoPE)
# ---------------------------------------------------------------------------

def gqa_forward(
    hidden_states: jnp.ndarray,        # [B, T, H]
    attention_mask: jnp.ndarray,       # [B, T]
    cos: jnp.ndarray,                  # [T, head_dim]
    sin: jnp.ndarray,                  # [T, head_dim]
    p: dict,                           # weight dict for this attention layer
    cfg: LlamaFP8Config,
) -> jnp.ndarray:
    B, T, H = hidden_states.shape
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim

    # JIT dequant FP8 weights -> bf16
    q_w = _dequant_jit(p["q_proj.weight"], p["q_proj.weight_scale"])
    k_w = _dequant_jit(p["k_proj.weight"], p["k_proj.weight_scale"])
    v_w = _dequant_jit(p["v_proj.weight"], p["v_proj.weight_scale"])
    o_w = _dequant_jit(p["o_proj.weight"], p["o_proj.weight_scale"])

    q = hidden_states @ q_w.T
    k = hidden_states @ k_w.T
    v = hidden_states @ v_w.T

    q = q.reshape(B, T, n_h, d)
    k = k.reshape(B, T, n_kv, d)
    v = v.reshape(B, T, n_kv, d)

    # Full RoPE
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)

    cos_bc = cos[None, None, :, :]
    sin_bc = sin[None, None, :, :]
    q = q * cos_bc + rotate_half(q) * sin_bc
    k = k * cos_bc + rotate_half(k) * sin_bc

    # GQA: repeat K/V heads
    v = v.transpose(0, 2, 1, 3)
    n_groups = n_h // n_kv
    if n_groups > 1:
        k = jnp.repeat(k, n_groups, axis=1)
        v = jnp.repeat(v, n_groups, axis=1)

    # Scaled dot-product attention
    scale = d ** -0.5
    attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

    causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    attn_weights = jnp.where(causal_mask[None, None], attn_weights, jnp.finfo(jnp.float32).min)

    pad_mask = attention_mask[:, None, None, :].astype(jnp.bool_)
    attn_weights = jnp.where(pad_mask, attn_weights, jnp.finfo(jnp.float32).min)

    attn_probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(hidden_states.dtype)

    attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, n_h * d)

    return attn_output @ o_w.T


# ---------------------------------------------------------------------------
# MLP with JIT FP8 dequant
# ---------------------------------------------------------------------------

def mlp_fp8_forward(
    x: jnp.ndarray,    # [B, T, H]
    p: dict,            # weight dict for this MLP layer
) -> jnp.ndarray:
    """SiLU-gated MLP with JIT FP8 dequant."""
    gate_w = _dequant_jit(p["gate_proj.weight"], p["gate_proj.weight_scale"])
    up_w = _dequant_jit(p["up_proj.weight"], p["up_proj.weight_scale"])
    down_w = _dequant_jit(p["down_proj.weight"], p["down_proj.weight_scale"])

    gate = jax.nn.silu(x @ gate_w.T)
    up = x @ up_w.T
    return (gate * up) @ down_w.T


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
    cfg: LlamaFP8Config,
) -> jnp.ndarray:
    pfx = f"model.layers.{layer_idx}"

    # Pre-norm + self-attention + residual
    attn_in = rms_norm(hidden_states, params[f"{pfx}.input_layernorm.weight"], cfg.rms_norm_eps)

    attn_p = {k.removeprefix(f"{pfx}.self_attn."): v
              for k, v in params.items()
              if k.startswith(f"{pfx}.self_attn.")}
    attn_out = gqa_forward(attn_in, attention_mask, cos, sin, attn_p, cfg)
    hidden_states = hidden_states + attn_out

    # Pre-norm + MLP (FP8 dequant) + residual
    mlp_in = rms_norm(hidden_states, params[f"{pfx}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

    mlp_p = {k.removeprefix(f"{pfx}.mlp."): v
             for k, v in params.items()
             if k.startswith(f"{pfx}.mlp.")}
    mlp_out = mlp_fp8_forward(mlp_in, mlp_p)

    return hidden_states + mlp_out


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------

def llama_fp8_forward(
    input_ids: jnp.ndarray,        # [B, T]  int32
    attention_mask: jnp.ndarray,   # [B, T]  int32/bool
    params: dict,
    cfg: LlamaFP8Config,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full Llama-70B-FP8 forward pass (no KV cache — training mode).

    Returns:
        last_hidden_state    [B, T, H]
        embed_tokens_weight  [vocab_size, H]
        aux_hidden_states    [B, T, 3*H]  -- from layers {1, 39, 76}
        target_logits        [B, T, vocab_size]  -- for KL divergence loss
    """
    B, T = input_ids.shape
    num_layers = cfg.num_hidden_layers

    # Layer indices for multi-layer feature fusion (SpecForge convention)
    # For Llama-3.3-70B (80 layers): {1, 39, 76}
    aux_layer_indices = {1, num_layers // 2 - 1, num_layers - 4}

    # Token embeddings (bf16, not FP8)
    embed_w = params["model.embed_tokens.weight"]
    hidden_states = embed_w[input_ids]

    # Build RoPE tables
    cos, sin = build_rope_freqs(cfg.head_dim, T, cfg.rope_theta)
    cos = cos[:T]
    sin = sin[:T]

    aux_hidden = []
    for layer_idx in range(num_layers):
        hidden_states = decoder_layer_forward(
            hidden_states, attention_mask, cos, sin, layer_idx, params, cfg
        )
        if layer_idx in aux_layer_indices:
            aux_hidden.append(hidden_states)

    # Final norm
    hidden_states = rms_norm(hidden_states, params["model.norm.weight"], cfg.rms_norm_eps)

    # Concatenate auxiliary hidden states
    aux_hidden_states = jnp.concatenate(aux_hidden, axis=-1)

    # Target logits (lm_head is bf16, not FP8)
    target_logits = hidden_states @ params["lm_head.weight"].T

    return hidden_states, embed_w, aux_hidden_states, target_logits
