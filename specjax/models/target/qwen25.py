"""
Qwen2.5 forward pass in pure JAX — frozen target model.

Architecture: Qwen2ForCausalLM (dense transformer, no MoE)
  vocab_size=152064, hidden_size=3584, num_hidden_layers=28,
  num_attention_heads=28, num_key_value_heads=4 (GQA 7:1),
  head_dim=128, intermediate_size=18944,
  attention_bias=True (Q/K/V only, not O),
  rope_theta=1_000_000, rms_norm_eps=1e-6,
  MLP: SiLU-gated (gate_proj, up_proj, down_proj)

Supports all Qwen2.5 dense sizes: 0.5B, 1.5B, 3B, 7B, 14B.
Config is read from the model's config.json at load time.

Key differences from GLM-4.7-FP8 (glm47_fp8.py):
  - No MoE — all layers are dense SiLU-gated MLP
  - No FP8 — pure bfloat16
  - No QK norm — standard Q/K before RoPE
  - Full RoPE — all head_dim dimensions rotated (not partial)
"""

import json
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from specjax.ops.norm import rms_norm
from specjax.ops.rope import build_rope_freqs, rotate_half
from specjax.ops.moe import mlp_forward
from specjax.ops.loading import discover_shards

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (plain Python — no JAX dependency)
# ---------------------------------------------------------------------------

class Qwen25Config:
    vocab_size: int = 152_064
    hidden_size: int = 3_584
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    intermediate_size: int = 18_944
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32_768

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen25Config":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_params(model_path: str, mesh=None) -> tuple[dict, "Qwen25Config"]:
    """
    Load Qwen2.5 weights from safetensors shards.

    All weights are stored in bfloat16 (no FP8 dequantization needed).
    No MoE expert stacking required — all layers are dense.

    Returns:
        params: flat dict {name: jnp.ndarray} in bfloat16
        config: Qwen25Config
    """
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = Qwen25Config.from_dict(cfg_dict)

    shard_files = discover_shards(model_path)

    import ml_dtypes

    logger.warning(f"Loading {len(shard_files)} safetensors shard(s) from {model_path}")
    params: dict = {}
    for shard_name in shard_files:
        shard_path = os.path.join(model_path, shard_name)
        shard = load_file(shard_path)
        for k, v in shard.items():
            params[k] = np.array(v, dtype=ml_dtypes.bfloat16)
        logger.warning(f"  Loaded {shard_name}  ({len(shard)} tensors)")

    logger.warning(f"Total parameters: {len(params)}")

    if mesh is not None:
        from specjax.models.sharding import shard_params
        logger.warning("Sharding parameters onto SPMD mesh ...")
        params = shard_params(params, mesh)
        logger.warning("Sharding complete.")
    else:
        params = {k: jnp.array(np.array(v, dtype=ml_dtypes.bfloat16))
                  for k, v in params.items()}

    return params, config


# ---------------------------------------------------------------------------
# GQA Attention (standard Grouped Query Attention with full RoPE)
# ---------------------------------------------------------------------------

def gqa_forward(
    hidden_states: jnp.ndarray,        # [B, T, H]
    attention_mask: jnp.ndarray,       # [B, T]  (1=attend, 0=mask)
    cos: jnp.ndarray,                  # [T, head_dim]
    sin: jnp.ndarray,                  # [T, head_dim]
    p: dict,                           # weight dict for this attention layer
    cfg: Qwen25Config,
) -> jnp.ndarray:                      # [B, T, H]
    """
    Standard GQA with full RoPE and attention bias on Q/K/V (not O).
    """
    B, T, H = hidden_states.shape
    n_h = cfg.num_attention_heads       # 28
    n_kv = cfg.num_key_value_heads      # 4
    d = cfg.head_dim                    # 128

    # Q/K/V projections (with bias on Q/K/V)
    q = hidden_states @ p["q_proj.weight"].T + p["q_proj.bias"]
    k = hidden_states @ p["k_proj.weight"].T + p["k_proj.bias"]
    v = hidden_states @ p["v_proj.weight"].T + p["v_proj.bias"]

    q = q.reshape(B, T, n_h, d).transpose(0, 2, 1, 3)    # [B, n_h, T, d]
    k = k.reshape(B, T, n_kv, d).transpose(0, 2, 1, 3)   # [B, n_kv, T, d]
    v = v.reshape(B, T, n_kv, d).transpose(0, 2, 1, 3)

    # Full RoPE (all head_dim dimensions rotated)
    cos_bc = cos[None, None, :, :]   # [1, 1, T, d]
    sin_bc = sin[None, None, :, :]
    q = q * cos_bc + rotate_half(q) * sin_bc
    k = k * cos_bc + rotate_half(k) * sin_bc

    # GQA: expand K/V heads to match Q heads
    n_groups = n_h // n_kv
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

    # o_proj: NO bias
    return attn_output @ p["o_proj.weight"].T


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

def decoder_layer_forward(
    hidden_states: jnp.ndarray,   # [B, T, H]
    attention_mask: jnp.ndarray,  # [B, T]
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    layer_idx: int,
    params: dict,
    cfg: Qwen25Config,
) -> jnp.ndarray:
    pfx = f"model.layers.{layer_idx}"

    # Pre-norm + GQA attention + residual
    attn_in = rms_norm(hidden_states, params[f"{pfx}.input_layernorm.weight"], cfg.rms_norm_eps)

    attn_p = {k.removeprefix(f"{pfx}.self_attn."): v
              for k, v in params.items()
              if k.startswith(f"{pfx}.self_attn.")}
    attn_out = gqa_forward(attn_in, attention_mask, cos, sin, attn_p, cfg)
    hidden_states = hidden_states + attn_out

    # Pre-norm + dense MLP + residual
    mlp_in = rms_norm(hidden_states, params[f"{pfx}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

    mlp_p = {k.removeprefix(f"{pfx}.mlp."): v
             for k, v in params.items()
             if k.startswith(f"{pfx}.mlp.")}
    mlp_out = mlp_forward(mlp_in, mlp_p)

    return hidden_states + mlp_out


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------

def qwen25_forward(
    input_ids: jnp.ndarray,        # [B, T]  int32
    attention_mask: jnp.ndarray,   # [B, T]  int32/bool
    params: dict,
    cfg: Qwen25Config,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full Qwen2.5 forward pass (no KV cache — training mode).

    Returns:
        last_hidden_state    [B, T, H]
        embed_tokens_weight  [vocab_size, H]
        aux_hidden_states    [B, T, 3*H]  — from layers {1, L//2-1, L-4}
        target_logits        [B, T, vocab_size]  — for KL divergence loss
    """
    B, T = input_ids.shape
    num_layers = cfg.num_hidden_layers
    d = cfg.head_dim

    # Layer indices for multi-layer feature fusion (SpecForge convention)
    aux_layer_indices = {1, num_layers // 2 - 1, num_layers - 4}

    # Token embeddings
    embed_w = params["model.embed_tokens.weight"]              # [V, H]
    hidden_states = embed_w[input_ids]                         # [B, T, H]

    # Build RoPE tables (full head_dim rotation)
    cos, sin = build_rope_freqs(d, T, cfg.rope_theta)
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

    # Concatenate 3 auxiliary hidden states for Eagle3 multi-layer fusion
    aux_hidden_states = jnp.concatenate(aux_hidden, axis=-1)   # [B, T, 3*H]

    # Target logits for KL divergence loss
    target_logits = hidden_states @ params["lm_head.weight"].T  # [B, T, V]

    return hidden_states, embed_w, aux_hidden_states, target_logits
