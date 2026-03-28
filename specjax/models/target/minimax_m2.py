"""
MiniMax-M2.5 forward pass in pure JAX — frozen target model.

Architecture: MiniMaxM2ForCausalLM (229B MoE, 10B active)
  vocab_size=200064, hidden_size=3072, num_hidden_layers=62,
  num_attention_heads=48, num_key_value_heads=8, head_dim=128,
  rotary_dim=64 (partial RoPE — first 64 of 128 dims),
  rope_theta=5_000_000,
  num_local_experts=256, num_experts_per_tok=8, intermediate_size=1536,
  shared_intermediate_size=0 (no shared expert),
  scoring_func=sigmoid, use_routing_bias=True (e_score_correction_bias),
  use_qk_norm=True, qk_norm_type=per_layer,
  rms_norm_eps=1e-6

Design:
  - Weights stored as flat dict {str: jnp.ndarray} loaded from safetensors.
  - All functions are pure JAX — no Flax modules, no mutable state.
  - Frozen: no gradients flow through it.
  - Returns (last_hidden_state, embed_w, aux_hidden_states, target_logits)
    for EAGLE3 draft training.

Weight loading:
  - FP8 block-wise (128x128) dequantization via dequant_fp8_block().
  - Expert weights renamed from w1/w3/w2 -> gate_proj/up_proj/down_proj
    and block_sparse_moe -> mlp during loading, then stacked via
    stack_moe_experts().

MoE routing:
  - Sigmoid-based top-k with post-sigmoid e_score_correction_bias.
    Bias affects expert SELECTION only, not final routing weights.
    This differs from the shared topk_router (which adds bias pre-sigmoid),
    so we implement a custom router here.
"""

import json
import logging
import os
import re

# ml_dtypes must be imported before safetensors to register float8/bfloat16
# dtypes with numpy (safetensors.numpy uses them during deserialization).
import ml_dtypes  # noqa: F401 -- side-effect import

import jax
import jax.numpy as jnp
import numpy as np

# Patch numpy with ml_dtypes fp8/bf16 types so safetensors can deserialize them.
for _dt_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "bfloat16"):
    if not hasattr(np, _dt_name) and hasattr(ml_dtypes, _dt_name):
        setattr(np, _dt_name, getattr(ml_dtypes, _dt_name))

from safetensors.numpy import load_file

from specjax.ops.norm import rms_norm
from specjax.ops.rope import build_rope_freqs, apply_partial_rope
from specjax.ops.moe import moe_experts_forward
from specjax.ops.fp8 import dequant_fp8_block
from specjax.ops.loading import discover_shards, stack_moe_experts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class MiniMaxM2Config:
    vocab_size: int = 200_064
    hidden_size: int = 3_072
    num_hidden_layers: int = 62
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    rotary_dim: int = 64
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 196_608
    # MoE
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    intermediate_size: int = 1_536
    # Attention
    use_qk_norm: bool = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # Map HuggingFace config keys to our names
        if "num_local_experts" in kwargs:
            self.n_routed_experts = kwargs["num_local_experts"]

    @classmethod
    def from_dict(cls, d: dict) -> "MiniMaxM2Config":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k) or k == "num_local_experts"})


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _rename_minimax_keys(params: dict) -> dict:
    """Rename MiniMax weight keys to match SpecJAX conventions.

    - block_sparse_moe -> mlp
    - experts.{e}.w1 -> experts.{e}.gate_proj
    - experts.{e}.w3 -> experts.{e}.up_proj
    - experts.{e}.w2 -> experts.{e}.down_proj
    - gate.weight stays as mlp.gate.weight (router)
    - e_score_correction_bias stays under mlp.
    """
    renamed = {}
    for k, v in params.items():
        new_k = k.replace(".block_sparse_moe.", ".mlp.")
        new_k = re.sub(r"\.w1\.", ".gate_proj.", new_k)
        new_k = re.sub(r"\.w3\.", ".up_proj.", new_k)
        new_k = re.sub(r"\.w2\.", ".down_proj.", new_k)
        renamed[new_k] = v
    return renamed


def load_params(model_path: str, mesh=None) -> tuple[dict, MiniMaxM2Config]:
    """Load MiniMax-M2.5 weights from safetensors, dequant FP8 -> bf16."""
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = MiniMaxM2Config.from_dict(cfg_dict)

    shard_files = discover_shards(model_path)

    fp8_dtypes = (ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2)

    logger.warning(f"Loading {len(shard_files)} safetensors shard(s) from {model_path}")
    raw_params: dict = {}
    for shard_name in shard_files:
        shard_path = os.path.join(model_path, shard_name)
        shard = load_file(shard_path)
        for k, v in shard.items():
            raw_params[k] = v
        logger.warning(f"  Loaded {shard_name}  ({len(shard)} tensors)")

    logger.warning(f"Total parameters (raw): {len(raw_params)}")

    # Skip MTP module weights (num_mtp_modules=3) — not needed for EAGLE3 training
    params: dict = {}
    for k, v in raw_params.items():
        if k.startswith("model.mtp_"):
            continue
        if k.endswith(".weight_scale_inv"):
            continue  # handled during dequant
        if v.dtype in fp8_dtypes:
            scale_key = k + "_scale_inv"
            if scale_key in raw_params:
                v = dequant_fp8_block(v, raw_params[scale_key], block_h=128, block_w=128)
            else:
                v = v.astype(np.float32).astype(ml_dtypes.bfloat16)
        elif v.dtype != ml_dtypes.bfloat16:
            v = np.array(v, dtype=ml_dtypes.bfloat16)
        params[k] = v

    logger.warning(f"Total parameters (dequantized, MTP stripped): {len(params)}")

    # Rename keys to SpecJAX convention
    params = _rename_minimax_keys(params)

    # Stack per-expert weights into batched tensors
    params = stack_moe_experts(params, config.n_routed_experts, rename_shared=False)
    logger.warning(f"Total parameters (stacked): {len(params)}")

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
# MoE routing (post-sigmoid bias — MiniMax-specific)
# ---------------------------------------------------------------------------

def _minimax_topk_router(
    hidden_states: jnp.ndarray,   # [B*T, H]
    gate_w: jnp.ndarray,          # [n_experts, H]
    bias: jnp.ndarray,            # [n_experts]
    num_experts_per_tok: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    MiniMax sigmoid router with post-sigmoid e_score_correction_bias.

    The bias shifts expert selection but does NOT affect final routing weights:
      1. scores = sigmoid(hidden @ gate.T)
      2. selection_scores = scores + bias   (for top-k only)
      3. top_k on selection_scores
      4. routing_weights = gather from ORIGINAL scores, then renormalize
    """
    logits = hidden_states.astype(jnp.float32) @ gate_w.T        # [T, E]
    scores = jax.nn.sigmoid(logits)                                # [T, E]
    selection_scores = scores + bias.astype(jnp.float32)           # [T, E]
    selected_experts = jnp.argsort(-selection_scores, axis=-1)[..., :num_experts_per_tok]
    routing_weights = jnp.take_along_axis(scores, selected_experts, axis=-1)
    routing_weights = routing_weights / (routing_weights.sum(axis=-1, keepdims=True) + 1e-9)
    return routing_weights.astype(jnp.bfloat16), selected_experts.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Attention (GQA + QK-norm + partial RoPE)
# ---------------------------------------------------------------------------

def attention_forward(hidden_states, attention_mask, cos, sin, p, cfg):
    """GQA attention with per-layer QK-norm and partial RoPE (rotary_dim=64)."""
    B, T, H = hidden_states.shape
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    rope_dim = cfg.rotary_dim
    n_groups = n_h // n_kv

    q = hidden_states @ p["q_proj.weight"].T     # [B, T, n_h*d]
    k = hidden_states @ p["k_proj.weight"].T     # [B, T, n_kv*d]
    v = hidden_states @ p["v_proj.weight"].T     # [B, T, n_kv*d]

    # QK-norm (per_layer): RMSNorm on the FLAT projected tensor before reshape
    if cfg.use_qk_norm:
        q = rms_norm(q, p["q_norm.weight"], cfg.rms_norm_eps)
        k = rms_norm(k, p["k_norm.weight"], cfg.rms_norm_eps)

    q = q.reshape(B, T, n_h, d).transpose(0, 2, 1, 3)    # [B, n_h, T, d]
    k = k.reshape(B, T, n_kv, d).transpose(0, 2, 1, 3)   # [B, n_kv, T, d]
    v = v.reshape(B, T, n_kv, d).transpose(0, 2, 1, 3)   # [B, n_kv, T, d]

    # Partial RoPE: rotate first rotary_dim dims, pass-through the rest
    q, k = apply_partial_rope(q, k, cos, sin, rope_dim)

    # GQA: repeat KV heads
    if n_groups > 1:
        k = jnp.repeat(k, n_groups, axis=1)
        v = jnp.repeat(v, n_groups, axis=1)

    scale = d ** -0.5
    attn_w = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    attn_w = jnp.where(causal_mask[None, None], attn_w, jnp.finfo(jnp.float32).min)
    pad_mask = attention_mask[:, None, None, :].astype(jnp.bool_)
    attn_w = jnp.where(pad_mask, attn_w, jnp.finfo(jnp.float32).min)
    attn_p = jax.nn.softmax(attn_w.astype(jnp.float32), axis=-1).astype(hidden_states.dtype)
    attn_out = jnp.einsum("bhqk,bhkd->bhqd", attn_p, v)

    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, n_h * d)
    return attn_out @ p["o_proj.weight"].T


# ---------------------------------------------------------------------------
# MoE block (no shared expert)
# ---------------------------------------------------------------------------

def moe_block_forward(hidden_states, p, cfg):
    """MoE block: MiniMax sigmoid router + batched expert dispatch. No shared expert."""
    B, T, H = hidden_states.shape
    flat = hidden_states.reshape(B * T, H)

    routing_weights, selected_experts = _minimax_topk_router(
        flat,
        p["gate.weight"],
        p["e_score_correction_bias"],
        cfg.num_experts_per_tok,
    )

    routed_out = moe_experts_forward(
        flat,
        routing_weights,
        selected_experts,
        p["experts.gate_up_proj"],   # [E, 2I, D]
        p["experts.down_proj"],      # [E, D, I]
    )

    return routed_out.reshape(B, T, H)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

def decoder_layer_forward(hidden_states, attention_mask, cos, sin, layer_idx, params, cfg):
    """Single MiniMax-M2.5 decoder layer: pre-norm + attn + pre-norm + MoE."""
    pfx = f"model.layers.{layer_idx}"

    # Self-attention
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states,
        params[f"{pfx}.input_layernorm.weight"],
        cfg.rms_norm_eps,
    )
    attn_p = {
        "q_proj.weight": params[f"{pfx}.self_attn.q_proj.weight"],
        "k_proj.weight": params[f"{pfx}.self_attn.k_proj.weight"],
        "v_proj.weight": params[f"{pfx}.self_attn.v_proj.weight"],
        "o_proj.weight": params[f"{pfx}.self_attn.o_proj.weight"],
    }
    if cfg.use_qk_norm:
        attn_p["q_norm.weight"] = params[f"{pfx}.self_attn.q_norm.weight"]
        attn_p["k_norm.weight"] = params[f"{pfx}.self_attn.k_norm.weight"]
    hidden_states = attention_forward(hidden_states, attention_mask, cos, sin, attn_p, cfg)
    hidden_states = residual + hidden_states

    # MoE
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states,
        params[f"{pfx}.post_attention_layernorm.weight"],
        cfg.rms_norm_eps,
    )
    moe_p = {
        "gate.weight": params[f"{pfx}.mlp.gate.weight"],
        "e_score_correction_bias": params[f"{pfx}.mlp.e_score_correction_bias"],
        "experts.gate_up_proj": params[f"{pfx}.mlp.experts.gate_up_proj"],
        "experts.down_proj": params[f"{pfx}.mlp.experts.down_proj"],
    }
    hidden_states = moe_block_forward(hidden_states, moe_p, cfg)
    hidden_states = residual + hidden_states

    return hidden_states


# ---------------------------------------------------------------------------
# Full forward pass
# ---------------------------------------------------------------------------

def minimax_m2_forward(input_ids, attention_mask, params, cfg):
    """
    MiniMax-M2.5 frozen target forward.

    Returns:
        last_hidden_state  [B, T, H]
        embed_w            [V, H]
        aux_hidden_states  [B, T, 3*H]  (from 3 intermediate layers)
        target_logits      [B, T, V]
    """
    B, T = input_ids.shape
    num_layers = cfg.num_hidden_layers

    embed_w = params["model.embed_tokens.weight"]
    hidden_states = embed_w[input_ids]   # [B, T, H]

    # Partial RoPE: cos/sin tables for rotary_dim dimensions
    cos, sin = build_rope_freqs(cfg.rotary_dim, T, cfg.rope_theta)
    cos, sin = cos[:T], sin[:T]

    # Auxiliary layer indices for EAGLE3 multi-layer fusion
    aux_layer_indices = {1, num_layers // 2 - 1, num_layers - 4}
    aux_hidden = []

    for layer_idx in range(num_layers):
        hidden_states = decoder_layer_forward(
            hidden_states, attention_mask, cos, sin, layer_idx, params, cfg,
        )
        if layer_idx in aux_layer_indices:
            aux_hidden.append(hidden_states)

    # Final norm
    hidden_states = rms_norm(
        hidden_states,
        params["model.norm.weight"],
        cfg.rms_norm_eps,
    )

    # Concatenate aux features for EAGLE3
    aux_hidden_states = jnp.concatenate(aux_hidden, axis=-1)  # [B, T, 3*H]

    # Compute target logits
    target_logits = hidden_states @ params["lm_head.weight"].T  # [B, T, V]

    return hidden_states, embed_w, aux_hidden_states, target_logits
