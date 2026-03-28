"""
GLM-5-FP8 forward pass in pure JAX — frozen target model.

Architecture: GlmMoeDsaForCausalLM
  vocab_size=154880, hidden_size=6144, num_hidden_layers=78,
  num_attention_heads=64, num_key_value_heads=64,
  q_lora_rank=2048, kv_lora_rank=512,
  qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
  rope_theta=1_000_000,
  n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8,
  moe_intermediate_size=2048, routed_scaling_factor=2.5,
  first_k_dense_replace=3 (layers 0-2 are dense MLP, layers 3-77 are MoE),
  rms_norm_eps=1e-5

Design:
  - Adapted from glm_flash.py for the larger GLM-5-FP8 model.
  - FP8 (e4m3) weights are dequantized to bfloat16 during loading using
    block-wise scales (block_size=[128,128]).
  - DSA (Dynamic Sparse Attention) indexer is SKIPPED: with max_length <= 2048,
    the top-k=2048 selection selects all positions, making DSA equivalent to
    full causal attention. Indexer weights are ignored during loading.
  - MTP (Multi-Token Prediction) layer is SKIPPED: not needed for Eagle3 training.
  - All pure-JAX forward-pass functions (MLA, MoE, router, decoder layer) are
    identical to glm_flash.py — they are already config-parameterized.
"""

import json
import logging
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from specjax.ops.norm import rms_norm
from specjax.ops.rope import build_rope_freqs, apply_rope_interleaved
from specjax.ops.moe import moe_forward as _moe_forward, mlp_forward
from specjax.ops.fp8 import dequant_fp8_block, dequant_fp8_1d
from specjax.ops.loading import discover_shards, stack_moe_experts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclass (plain Python — no JAX dependency)
# ---------------------------------------------------------------------------

class GLM5Config:
    vocab_size: int = 154_880
    hidden_size: int = 6_144
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    q_lora_rank: int = 2_048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    rope_theta: float = 1_000_000.0
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2_048
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
    def from_dict(cls, d: dict) -> "GLM5Config":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# ---------------------------------------------------------------------------
# FP8 loading helpers
# ---------------------------------------------------------------------------

# Weight keys that are NOT quantized (from GLM-5-FP8 quantization_config)
_SKIP_QUANTIZE_PATTERNS = [
    "lm_head",
    "model.embed_tokens",
    "input_layernorm",
    "post_attention_layernorm",
    "mlp.gate",            # router weights, not gate_proj
    "model.norm",
    "indexer",             # DSA indexer (skipped entirely)
]


def _is_weight_scale(key: str) -> bool:
    """Check if a key is a scale tensor (not a model weight)."""
    return key.endswith(".weight_scale") or key.endswith("_scale")


def _should_skip_key(key: str) -> bool:
    """Skip DSA indexer and MTP weights — not used in training forward pass."""
    if ".indexer." in key or key.startswith("model.nextn_predict_layers"):
        return True
    return False


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_params(model_path: str, mesh=None) -> tuple[dict, "GLM5Config"]:
    """
    Load GLM-5-FP8 weights from safetensors shards.

    FP8 (e4m3) weights are dequantized to bfloat16 using block-wise scales.
    DSA indexer and MTP weights are skipped.

    Returns:
        params: flat dict {name: jnp.ndarray} in bfloat16
        config: GLM5Config
    """
    import ml_dtypes

    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = GLM5Config.from_dict(cfg_dict)

    shard_files = discover_shards(model_path)

    logger.warning(
        f"Loading {len(shard_files)} safetensors shard(s) from {model_path}"
    )

    # First pass: collect all tensors as numpy, dequantizing FP8 -> bf16
    params: dict = {}
    scales: dict = {}  # weight_scale tensors collected for FP8 dequant
    skipped_keys = 0

    for shard_name in shard_files:
        shard_path = os.path.join(model_path, shard_name)
        shard = load_file(shard_path)

        # Collect scale tensors first (needed for dequant)
        for k, v in shard.items():
            if _is_weight_scale(k):
                scales[k] = np.array(v, dtype=np.float32)

        for k, v in shard.items():
            # Skip scale tensors (handled separately)
            if _is_weight_scale(k):
                continue

            # Skip DSA indexer and MTP weights
            if _should_skip_key(k):
                skipped_keys += 1
                continue

            v_np = np.array(v)

            # Check if this weight is FP8 quantized
            is_fp8 = (
                v_np.dtype == np.uint8
                or str(v_np.dtype) == "float8_e4m3fn"
                or (hasattr(ml_dtypes, "float8_e4m3fn")
                    and v_np.dtype == ml_dtypes.float8_e4m3fn)
            )

            if is_fp8:
                # Find matching scale tensor
                scale_key = k.replace(".weight", ".weight_scale")
                scale = scales.get(scale_key)
                if scale is None:
                    scale_key = k + "_scale"
                    scale = scales.get(scale_key)

                if scale is None:
                    logger.warning(
                        f"  WARNING: No scale found for FP8 weight {k}, "
                        f"casting directly to bf16"
                    )
                    params[k] = v_np.astype(np.float32).astype(ml_dtypes.bfloat16)
                elif v_np.ndim == 2:
                    params[k] = dequant_fp8_block(v_np, scale)
                elif v_np.ndim == 1:
                    params[k] = dequant_fp8_1d(v_np, scale)
                else:
                    logger.warning(
                        f"  WARNING: FP8 weight {k} has {v_np.ndim}D shape "
                        f"{v_np.shape}, casting directly"
                    )
                    params[k] = v_np.astype(np.float32).astype(ml_dtypes.bfloat16)
            else:
                # Non-FP8: convert to bfloat16 (embeddings, norms, router, lm_head)
                params[k] = np.array(v_np, dtype=ml_dtypes.bfloat16)

        logger.warning(f"  Loaded {shard_name}  ({len(shard)} tensors)")
        del shard  # free CPU memory

    logger.warning(
        f"Total parameters (raw): {len(params)} "
        f"(skipped {skipped_keys} indexer/MTP tensors)"
    )

    del scales

    # Stack per-expert weights into [E, 2I, D] / [E, D, I] tensors
    params = stack_moe_experts(params, config.n_routed_experts)
    logger.warning(f"Total parameters (stacked): {len(params)}")

    if mesh is not None:
        from specjax.models.sharding import shard_params
        logger.warning("Sharding parameters onto SPMD mesh ...")
        params = shard_params(params, mesh)
        logger.warning("Sharding complete.")
    else:
        import ml_dtypes
        params = {
            k: jnp.array(np.array(v, dtype=ml_dtypes.bfloat16))
            for k, v in params.items()
        }

    return params, config


# ---------------------------------------------------------------------------
# Multi-head Latent Attention (MLA) — identical logic to GLM-Flash
# ---------------------------------------------------------------------------

def mla_forward(
    hidden_states: jnp.ndarray,        # [B, T, H]
    attention_mask: jnp.ndarray,       # [B, T]  (1=attend, 0=mask)
    cos: jnp.ndarray,                  # [T, qk_rope_head_dim]
    sin: jnp.ndarray,                  # [T, qk_rope_head_dim]
    p: dict,                           # weight dict for this attention layer
    cfg: GLM5Config,
) -> jnp.ndarray:                      # [B, T, H]
    B, T, H = hidden_states.shape
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d_rope = cfg.qk_rope_head_dim
    d_nope = cfg.qk_nope_head_dim
    d_v = cfg.v_head_dim

    # -- Q path --
    c_q = hidden_states @ p["q_a_proj.weight"].T
    c_q = rms_norm(c_q, p["q_a_layernorm.weight"], cfg.rms_norm_eps)
    q = c_q @ p["q_b_proj.weight"].T
    q = q.reshape(B, T, n_h, d_nope + d_rope)
    q_nope = q[..., :d_nope]
    q_rope = q[..., d_nope:]

    # -- KV path --
    kv_and_rope_k = hidden_states @ p["kv_a_proj_with_mqa.weight"].T
    compressed_kv = kv_and_rope_k[:, :, :cfg.kv_lora_rank]
    k_rope_single = kv_and_rope_k[:, :, cfg.kv_lora_rank:]

    compressed_kv = rms_norm(compressed_kv, p["kv_a_layernorm.weight"], cfg.rms_norm_eps)
    kv = compressed_kv @ p["kv_b_proj.weight"].T
    kv = kv.reshape(B, T, n_kv, d_nope + d_v)
    k_nope = kv[..., :d_nope]
    v = kv[..., d_nope:]

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
    cfg: GLM5Config,
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
        # Dense MLP (layers 0-2)
        mlp_p = {k.removeprefix(f"{pfx}.mlp."): v
                 for k, v in params.items()
                 if k.startswith(f"{pfx}.mlp.")}
        mlp_out = mlp_forward(mlp_in, mlp_p)
    else:
        # Sparse MoE (layers 3-77)
        moe_p = {k.removeprefix(f"{pfx}.mlp."): v
                 for k, v in params.items()
                 if k.startswith(f"{pfx}.mlp.")}
        mlp_out = _moe_forward(mlp_in, moe_p, cfg.num_experts_per_tok,
                               cfg.routed_scaling_factor, cfg.norm_topk_prob)

    return hidden_states + mlp_out


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------

def glm5_forward(
    input_ids: jnp.ndarray,        # [B, T]  int32
    attention_mask: jnp.ndarray,   # [B, T]  int32/bool
    params: dict,
    cfg: GLM5Config,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full GLM-5-FP8 forward pass (no KV cache — training mode).

    DSA indexer is NOT used: with max_length <= 2048, top-k=2048 selects all
    positions, making DSA equivalent to full causal attention.

    Returns:
        last_hidden_state    [B, T, H]
        embed_tokens_weight  [vocab_size, H]
        aux_hidden_states    [B, T, 3*H]
        target_logits        [B, T, vocab_size]
    """
    B, T = input_ids.shape
    num_layers = cfg.num_hidden_layers

    # For GLM-5 (78 layers): {1, 38, 74}
    aux_layer_indices = {1, num_layers // 2 - 1, num_layers - 4}

    embed_w = params["model.embed_tokens.weight"]
    hidden_states = embed_w[input_ids]

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

    hidden_states = rms_norm(hidden_states, params["model.norm.weight"], cfg.rms_norm_eps)

    aux_hidden_states = jnp.concatenate(aux_hidden, axis=-1)

    target_logits = hidden_states @ params["lm_head.weight"].T

    return hidden_states, embed_w, aux_hidden_states, target_logits
