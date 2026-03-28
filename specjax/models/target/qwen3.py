"""
Qwen3 forward pass in pure JAX — frozen target model.

Architecture: Qwen3ForCausalLM (dense transformer, no MoE)
  Covers Qwen3-8B and Qwen3-14B.

Key differences from Qwen2.5 (qwen25.py):
  - attention_bias=False: no bias on Q/K/V projections
  - Per-head QK RMSNorm (q_norm, k_norm weights per layer, shape [head_dim])
    applied after reshape, before RoPE — same convention as Llama 3
  - tie_word_embeddings=True: lm_head.weight aliased from embed_tokens.weight
  - Vocab size 151936 (vs 152064 in Qwen2.5)
  - rope_theta=10000 (default; overridden from config.json)

All other aspects (dense SiLU-gated MLP, full RoPE, GQA) are identical
to qwen25.py.

Qwen3-8B defaults:
  vocab_size=151936, hidden_size=4096, num_hidden_layers=36,
  num_attention_heads=32, num_key_value_heads=8, head_dim=128,
  intermediate_size=22528, rope_theta=10000, rms_norm_eps=1e-6

Qwen3-14B:
  hidden_size=5120, num_hidden_layers=40, num_attention_heads=40,
  num_key_value_heads=8 — all read from config.json at load time.
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
# Config
# ---------------------------------------------------------------------------

class Qwen3Config:
    vocab_size: int = 151_936
    hidden_size: int = 4_096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 0  # 0 = compute from hidden_size // num_attention_heads
    intermediate_size: int = 22_528
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32_768
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Fall back to computed head_dim if not explicitly set or zero
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen3Config":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_params(model_path: str, mesh=None) -> tuple[dict, "Qwen3Config"]:
    """
    Load Qwen3 weights from safetensors shards.

    Handles tied embeddings: if lm_head.weight is absent, it is aliased
    from model.embed_tokens.weight (tie_word_embeddings=True by default).

    Returns:
        params: flat dict {name: jnp.ndarray} in bfloat16
        config: Qwen3Config
    """
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = Qwen3Config.from_dict(cfg_dict)

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

    # Tied embeddings: alias lm_head from embed_tokens if not stored separately
    if "lm_head.weight" not in params and config.tie_word_embeddings:
        logger.warning("lm_head.weight not found — aliasing from model.embed_tokens.weight (tied embeddings)")
        params["lm_head.weight"] = params["model.embed_tokens.weight"]

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
# GQA Attention with QK per-head RMSNorm (no attention bias)
# ---------------------------------------------------------------------------

def gqa_forward(
    hidden_states: jnp.ndarray,        # [B, T, H]
    attention_mask: jnp.ndarray,       # [B, T]  (1=attend, 0=mask)
    cos: jnp.ndarray,                  # [T, head_dim]
    sin: jnp.ndarray,                  # [T, head_dim]
    p: dict,                           # weight dict for this attention layer
    cfg: Qwen3Config,
) -> jnp.ndarray:                      # [B, T, H]
    """
    GQA with full RoPE, per-head QK RMSNorm, and no attention bias.
    """
    B, T, H = hidden_states.shape
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim

    # Q/K/V projections — no bias in Qwen3
    q = hidden_states @ p["q_proj.weight"].T    # [B, T, n_h * d]
    k = hidden_states @ p["k_proj.weight"].T    # [B, T, n_kv * d]
    v = hidden_states @ p["v_proj.weight"].T    # [B, T, n_kv * d]

    q = q.reshape(B, T, n_h, d).transpose(0, 2, 1, 3)    # [B, n_h, T, d]
    k = k.reshape(B, T, n_kv, d).transpose(0, 2, 1, 3)   # [B, n_kv, T, d]
    v = v.reshape(B, T, n_kv, d).transpose(0, 2, 1, 3)

    # Per-head QK RMSNorm (Qwen3 addition — not present in Qwen2.5)
    # q_norm.weight and k_norm.weight have shape [head_dim]; broadcast over B, n_h, T.
    eps = cfg.rms_norm_eps
    q = rms_norm(q, p["q_norm.weight"], eps)   # [B, n_h, T, d]
    k = rms_norm(k, p["k_norm.weight"], eps)   # [B, n_kv, T, d]

    # Full RoPE
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
    cfg: Qwen3Config,
) -> jnp.ndarray:
    pfx = f"model.layers.{layer_idx}"

    attn_in = rms_norm(hidden_states, params[f"{pfx}.input_layernorm.weight"], cfg.rms_norm_eps)

    attn_p = {k.removeprefix(f"{pfx}.self_attn."): v
              for k, v in params.items()
              if k.startswith(f"{pfx}.self_attn.")}
    attn_out = gqa_forward(attn_in, attention_mask, cos, sin, attn_p, cfg)
    hidden_states = hidden_states + attn_out

    mlp_in = rms_norm(hidden_states, params[f"{pfx}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

    mlp_p = {k.removeprefix(f"{pfx}.mlp."): v
             for k, v in params.items()
             if k.startswith(f"{pfx}.mlp.")}
    mlp_out = mlp_forward(mlp_in, mlp_p)

    return hidden_states + mlp_out


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------

def qwen3_forward(
    input_ids: jnp.ndarray,        # [B, T]  int32
    attention_mask: jnp.ndarray,   # [B, T]  int32/bool
    params: dict,
    cfg: Qwen3Config,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full Qwen3 forward pass (no KV cache — training mode).

    Returns:
        last_hidden_state    [B, T, H]
        embed_tokens_weight  [vocab_size, H]
        aux_hidden_states    [B, T, 3*H]  — from layers {1, L//2-1, L-4}
        target_logits        [B, T, vocab_size]
    """
    B, T = input_ids.shape
    num_layers = cfg.num_hidden_layers
    d = cfg.head_dim

    aux_layer_indices = {1, num_layers // 2 - 1, num_layers - 4}

    embed_w = params["model.embed_tokens.weight"]
    hidden_states = embed_w[input_ids]

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

    hidden_states = rms_norm(hidden_states, params["model.norm.weight"], cfg.rms_norm_eps)

    aux_hidden_states = jnp.concatenate(aux_hidden, axis=-1)   # [B, T, 3*H]

    target_logits = hidden_states @ params["lm_head.weight"].T  # [B, T, V]

    return hidden_states, embed_w, aux_hidden_states, target_logits
