"""
GLM-4.7-Flash forward pass in pure JAX — frozen target model.

Architecture: Glm4MoeLiteForCausalLM
  vocab_size=154880, hidden_size=2048, num_hidden_layers=47,
  num_attention_heads=20, num_key_value_heads=20,
  q_lora_rank=768, kv_lora_rank=512,
  qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
  rope_theta=1_000_000,
  n_routed_experts=64, n_shared_experts=1, num_experts_per_tok=4,
  moe_intermediate_size=1536, routed_scaling_factor=1.8,
  first_k_dense_replace=1 (layer 0 is dense MLP, layers 1–46 are MoE),
  rms_norm_eps=1e-5

Design:
  - Weights are stored as a flat dict {str: jnp.ndarray} (loaded from safetensors).
  - All functions are pure JAX — no Flax modules, no mutable state.
  - The model is frozen: no gradients flow through it.
  - The forward pass returns (last_hidden_state, embed_tokens_weight,
    aux_hidden_states, target_logits) for the training script to pass to the
    Eagle3 draft model. aux_hidden_states is [B,T,3H] from 3 intermediate
    layers (matching EAGLE-3 multi-layer fusion). target_logits is [B,T,V]
    for KL divergence loss.

Weight loading:
  - load_params(model_path): reads all safetensors shards, converts to jnp.bfloat16,
    then calls shard_params(params, mesh) for SPMD placement.

MoE dispatch:
  - Uses the static-shape einsum approach (same as the PyTorch/XLA patch) to avoid
    data-dependent ops that would cause XLA recompilation / CPU-TPU syncs.
"""

import json
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from specjax.ops.norm import rms_norm
from specjax.ops.rope import build_rope_freqs, apply_rope_interleaved
from specjax.ops.moe import moe_forward as _moe_forward, mlp_forward
from specjax.ops.loading import discover_shards, stack_moe_experts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclass (plain Python — no JAX dependency)
# ---------------------------------------------------------------------------

class GLMConfig:
    vocab_size: int = 154_880
    hidden_size: int = 2_048
    num_hidden_layers: int = 47
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    q_lora_rank: int = 768
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    rope_theta: float = 1_000_000.0
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 1_536
    routed_scaling_factor: float = 1.8
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 1
    intermediate_size: int = 10_240       # dense MLP intermediate (layer 0)
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 202_752

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: dict) -> "GLMConfig":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_params(model_path: str, mesh=None) -> tuple[dict, "GLMConfig"]:
    """
    Load GLM-4.7-Flash weights from safetensors shards.

    Returns:
        params: flat dict {name: jnp.ndarray} in bfloat16
        config: GLMConfig
    """
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = GLMConfig.from_dict(cfg_dict)

    shard_files = discover_shards(model_path)

    import ml_dtypes

    logger.warning(f"Loading {len(shard_files)} safetensors shard(s) from {model_path}")
    # Keep as CPU numpy (ml_dtypes.bfloat16) until sharding — avoids single-chip OOM
    params: dict = {}
    for shard_name in shard_files:
        shard_path = os.path.join(model_path, shard_name)
        shard = load_file(shard_path)
        for k, v in shard.items():
            params[k] = np.array(v, dtype=ml_dtypes.bfloat16)
        logger.warning(f"  Loaded {shard_name}  ({len(shard)} tensors)")

    logger.warning(f"Total parameters (raw): {len(params)}")

    # Stack per-expert weights into [E, 2I, D] / [E, D, I] tensors
    params = stack_moe_experts(params, config.n_routed_experts)
    logger.warning(f"Total parameters (stacked): {len(params)}")

    if mesh is not None:
        from specjax.models.sharding import shard_params
        logger.warning("Sharding parameters onto SPMD mesh ...")
        params = shard_params(params, mesh)
        logger.warning("Sharding complete.")
    else:
        # No mesh: convert numpy -> JAX bfloat16 (single device, test/debug only)
        params = {k: jnp.array(np.array(v, dtype=ml_dtypes.bfloat16))
                  for k, v in params.items()}

    return params, config


# ---------------------------------------------------------------------------
# Multi-head Latent Attention (MLA)
# ---------------------------------------------------------------------------

def mla_forward(
    hidden_states: jnp.ndarray,        # [B, T, H]
    attention_mask: jnp.ndarray,       # [B, T]  (1=attend, 0=mask)
    cos: jnp.ndarray,                  # [T, qk_rope_head_dim]
    sin: jnp.ndarray,                  # [T, qk_rope_head_dim]
    p: dict,                           # weight dict for this attention layer
    cfg: GLMConfig,
) -> jnp.ndarray:                      # [B, T, H]
    B, T, H = hidden_states.shape
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d_rope = cfg.qk_rope_head_dim
    d_nope = cfg.qk_nope_head_dim
    d_v = cfg.v_head_dim

    # -- Q path --
    c_q = hidden_states @ p["q_a_proj.weight"].T                    # [B, T, q_lora_rank]
    c_q = rms_norm(c_q, p["q_a_layernorm.weight"], cfg.rms_norm_eps)
    q = c_q @ p["q_b_proj.weight"].T                                # [B, T, n_h*(d_nope+d_rope)]
    q = q.reshape(B, T, n_h, d_nope + d_rope)
    q_nope = q[..., :d_nope]                                        # [B, T, n_h, d_nope]
    q_rope  = q[..., d_nope:]                                       # [B, T, n_h, d_rope]

    # -- KV path --
    kv_and_rope_k = hidden_states @ p["kv_a_proj_with_mqa.weight"].T   # [B, T, kv_lora_rank+d_rope]
    compressed_kv = kv_and_rope_k[:, :, :cfg.kv_lora_rank]
    k_rope_single = kv_and_rope_k[:, :, cfg.kv_lora_rank:]           # [B, T, d_rope]

    compressed_kv = rms_norm(compressed_kv, p["kv_a_layernorm.weight"], cfg.rms_norm_eps)
    kv = compressed_kv @ p["kv_b_proj.weight"].T                    # [B, T, n_kv*(d_nope+d_v)]
    kv = kv.reshape(B, T, n_kv, d_nope + d_v)
    k_nope = kv[..., :d_nope]                                       # [B, T, n_kv, d_nope]
    v       = kv[..., d_nope:]                                       # [B, T, n_kv, d_v]

    k_rope = jnp.broadcast_to(
        k_rope_single[:, :, None, :],
        (B, T, n_kv, d_rope),
    )

    # -- Apply RoPE --
    q_rope_t = q_rope.transpose(0, 2, 1, 3)
    k_rope_t = k_rope.transpose(0, 2, 1, 3)
    q_rope_t, k_rope_t = apply_rope_interleaved(q_rope_t, k_rope_t, cos, sin)
    q_rope_back = q_rope_t.transpose(0, 2, 1, 3)
    k_rope_back = k_rope_t.transpose(0, 2, 1, 3)

    q_full = jnp.concatenate([q_nope, q_rope_back], axis=-1)
    k_full = jnp.concatenate([k_nope, k_rope_back], axis=-1)

    q_t = q_full.transpose(0, 2, 1, 3)
    k_t = k_full.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    if n_kv != n_h:
        n_groups = n_h // n_kv
        k_t = jnp.repeat(k_t, n_groups, axis=1)
        v_t = jnp.repeat(v_t, n_groups, axis=1)

    # -- Scaled dot-product attention --
    head_dim = d_nope + d_rope
    scale = head_dim ** -0.5
    attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q_t, k_t) * scale

    causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    attn_weights = jnp.where(causal_mask[None, None], attn_weights, jnp.finfo(jnp.float32).min)

    pad_mask = attention_mask[:, None, None, :].astype(jnp.bool_)
    attn_weights = jnp.where(pad_mask, attn_weights, jnp.finfo(jnp.float32).min)

    attn_probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(hidden_states.dtype)

    attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v_t)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, n_h * d_v)

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
    cfg: GLMConfig,
) -> jnp.ndarray:
    pfx = f"model.layers.{layer_idx}"

    # Pre-norm + self-attention + residual
    attn_in = rms_norm(hidden_states, params[f"{pfx}.input_layernorm.weight"], cfg.rms_norm_eps)

    attn_p = {k.removeprefix(f"{pfx}.self_attn."): v
              for k, v in params.items()
              if k.startswith(f"{pfx}.self_attn.")}
    attn_out = mla_forward(attn_in, attention_mask, cos, sin, attn_p, cfg)
    hidden_states = hidden_states + attn_out

    # Pre-norm + MLP/MoE + residual
    mlp_in = rms_norm(hidden_states, params[f"{pfx}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

    if layer_idx < cfg.first_k_dense_replace:
        # Dense MLP (layer 0)
        mlp_p = {k.removeprefix(f"{pfx}.mlp."): v
                 for k, v in params.items()
                 if k.startswith(f"{pfx}.mlp.")}
        mlp_out = mlp_forward(mlp_in, mlp_p)
    else:
        # Sparse MoE (layers 1-46)
        moe_p = {k.removeprefix(f"{pfx}.mlp."): v
                 for k, v in params.items()
                 if k.startswith(f"{pfx}.mlp.")}
        mlp_out = _moe_forward(mlp_in, moe_p, cfg.num_experts_per_tok,
                               cfg.routed_scaling_factor, cfg.norm_topk_prob)

    return hidden_states + mlp_out


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------

def glm_forward(
    input_ids: jnp.ndarray,        # [B, T]  int32
    attention_mask: jnp.ndarray,   # [B, T]  int32/bool
    params: dict,
    cfg: GLMConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full GLM-4.7-Flash forward pass (no KV cache — training mode).

    Returns:
        last_hidden_state    [B, T, H]
        embed_tokens_weight  [vocab_size, H]
        aux_hidden_states    [B, T, 3*H]  — concatenated hidden states from
                             3 intermediate layers (low, mid, high) for
                             EAGLE-3 multi-layer feature fusion
        target_logits        [B, T, vocab_size]  — for KL divergence loss
    """
    B, T = input_ids.shape
    num_layers = cfg.num_hidden_layers

    # Layer indices for multi-layer feature fusion (matches SpecForge convention)
    # For GLM-4.7-Flash (47 layers): {1, 22, 43}
    aux_layer_indices = {1, num_layers // 2 - 1, num_layers - 4}

    # Token embeddings
    embed_w = params["model.embed_tokens.weight"]              # [V, H]
    hidden_states = embed_w[input_ids]                         # [B, T, H]

    # Build RoPE tables once for this sequence length
    cos, sin = build_rope_freqs(cfg.qk_rope_head_dim, T, cfg.rope_theta)
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
