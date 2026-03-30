"""
Qwen3-Coder-Next forward pass in pure JAX — frozen target model.

Architecture: Qwen3NextForCausalLM (hybrid GDN/Attention, MoE)
  vocab_size=151936, hidden_size=2048, num_hidden_layers=48,
  full_attention_interval=4 (layers 3,7,11,...,47 are GQA; rest are GDN),
  num_attention_heads=16, num_key_value_heads=2, head_dim=256,
  partial_rotary_factor=0.25 (64 of 256 dims get RoPE),
  num_experts=512, num_experts_per_tok=10, moe_intermediate_size=512,
  shared_expert_intermediate_size=512,
  rope_theta=5_000_000, rms_norm_eps=1e-6

GDN weight layout:
  linear_attn.in_proj_qkvz  [12288, 2048]  fused q(2048)+k(2048)+v(4096)+z(4096)
  linear_attn.conv1d         [8192, 1, 4]   depthwise causal conv on q+k+v
  linear_attn.in_proj_ba     [64, 2048]     beta(32) + A/dt(32)
  linear_attn.A_log          [32]           log of decay A
  linear_attn.dt_bias        [32]           dt bias
  linear_attn.norm           [128]          group norm on v output
  linear_attn.out_proj       [2048, 4096]   project v_heads*v_dim -> hidden

Attention weight layout:
  self_attn.q_proj  [8192, 2048]  16 heads x (256 q + 256 gate) = 8192
  self_attn.k_proj  [512, 2048]   2 kv_heads x 256
  self_attn.v_proj  [512, 2048]   2 kv_heads x 256
  self_attn.o_proj  [2048, 4096]  16 heads x 256 = 4096 -> 2048
  self_attn.q_norm  [256]         per-head RMSNorm
  self_attn.k_norm  [256]         per-head RMSNorm

FP8 quantization:
  Block-wise FP8 (128x128 blocks). Dequantize: bf16 = fp8 * scale_inv.
  weight_scale_inv tensors have shape [ceil(out/128), ceil(in/128)].
"""

import json
import logging
import os
from typing import Optional

# ml_dtypes must be imported before safetensors to register float8/bfloat16
# dtypes with numpy (safetensors.numpy uses them during deserialization).
import ml_dtypes  # noqa: F401 -- side-effect import

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# Patch numpy with ml_dtypes fp8/bf16 types so safetensors can deserialize them.
for _dt_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "bfloat16"):
    if not hasattr(np, _dt_name) and hasattr(ml_dtypes, _dt_name):
        setattr(np, _dt_name, getattr(ml_dtypes, _dt_name))

from safetensors.numpy import load_file

from specjax.ops.norm import rms_norm
from specjax.ops.rope import build_rope_freqs, apply_partial_rope
from specjax.ops.moe import moe_forward as _moe_forward
from specjax.ops.fp8 import dequant_fp8_qwen
from specjax.ops.loading import discover_shards, stack_moe_experts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

class Qwen3NextConfig:
    vocab_size: int = 151_936
    hidden_size: int = 2_048
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    full_attention_interval: int = 4
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262_144
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = True
    # GDN
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    gdn_chunk_size: int = 64

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen3NextConfig":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    @property
    def rope_head_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def is_attention_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.full_attention_interval == 0


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_params(model_path: str, mesh=None) -> tuple[dict, Qwen3NextConfig]:
    """Load Qwen3-Coder-Next weights from safetensors, dequant FP8 -> bf16."""
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = Qwen3NextConfig.from_dict(cfg_dict)

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

    # Dequantize FP8 weights using their scale_inv tensors, skip scale_inv keys
    params: dict = {}
    for k, v in raw_params.items():
        if k.endswith(".weight_scale_inv"):
            continue  # handled during dequant
        if v.dtype in fp8_dtypes:
            scale_key = k + "_scale_inv"
            if scale_key in raw_params:
                v = dequant_fp8_qwen(v, raw_params[scale_key])
            else:
                v = v.astype(np.float32).astype(ml_dtypes.bfloat16)
        elif v.dtype != ml_dtypes.bfloat16:
            v = np.array(v, dtype=ml_dtypes.bfloat16)
        params[k] = v

    logger.warning(f"Total parameters (dequantized): {len(params)}")

    # Stack per-expert weights
    params = stack_moe_experts(params, config.num_experts, rename_shared=False)
    logger.warning(f"Total parameters (stacked): {len(params)}")

    if mesh is not None:
        from specjax.models.sharding import shard_params
        logger.warning("Sharding parameters onto SPMD mesh ...")
        sharded = shard_params(params, mesh)
        params = sharded
        logger.warning("Sharding complete.")
    else:
        params = {k: jnp.array(np.array(v, dtype=ml_dtypes.bfloat16))
                  for k, v in params.items()}

    return params, config


# ---------------------------------------------------------------------------
# Per-head RMSNorm (for QK norm in attention and GDN group norm)
# ---------------------------------------------------------------------------

def _per_head_rms_norm(x, weight, eps):
    """x: [B, T, n_h, d], weight: [d]"""
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
    return (weight * x_f32 * lax.rsqrt(var + eps)).astype(x.dtype)


# ---------------------------------------------------------------------------
# GQA Attention with Gating (full attention layers)
# ---------------------------------------------------------------------------

def attention_forward(hidden_states, attention_mask, cos, sin, p, cfg):
    """
    GQA with gating and partial RoPE.
    q_proj [8192, 2048] = 16 heads x (256_q + 256_gate).
    Output = attn_output * sigmoid(gate).
    """
    B, T, H = hidden_states.shape
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    rope_dim = cfg.rope_head_dim
    n_groups = n_h // n_kv

    # Q: [B, T, 2*n_h*d] -> split q and gate
    q_gate = hidden_states @ p["q_proj.weight"].T
    q_gate = q_gate.reshape(B, T, n_h, 2 * d)
    q = q_gate[..., :d]
    gate = q_gate[..., d:]

    k = (hidden_states @ p["k_proj.weight"].T).reshape(B, T, n_kv, d)
    v = (hidden_states @ p["v_proj.weight"].T).reshape(B, T, n_kv, d)

    q = _per_head_rms_norm(q, p["q_norm.weight"], cfg.rms_norm_eps)
    k = _per_head_rms_norm(k, p["k_norm.weight"], cfg.rms_norm_eps)

    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    gate = gate.transpose(0, 2, 1, 3)

    q, k = apply_partial_rope(q, k, cos, sin, rope_dim)

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

    # Output gating
    attn_out = attn_out * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(attn_out.dtype)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, n_h * d)
    return attn_out @ p["o_proj.weight"].T


# ---------------------------------------------------------------------------
# GDN (Gated DeltaNet) — chunk-mode
# ---------------------------------------------------------------------------

def gdn_forward(hidden_states, p, cfg):
    """
    Gated DeltaNet layer.

    in_proj_qkvz [12288, H] = q(nk*dk) + k(nk*dk) + v(nv*dv) + z(nv*dv)
    conv1d [q+k+v_dim, 1, kernel] causal depthwise
    in_proj_ba [nv+nk, H] = beta(nv) + dt/alpha(nk)
    A_log [nk] and dt_bias [nk]
    """
    B, T, H = hidden_states.shape
    n_k = cfg.linear_num_key_heads
    n_v = cfg.linear_num_value_heads
    d_k = cfg.linear_key_head_dim
    d_v = cfg.linear_value_head_dim

    q_dim = n_k * d_k      # 2048
    k_dim = n_k * d_k      # 2048
    v_dim = n_v * d_v       # 4096
    qkv_dim = q_dim + k_dim + v_dim  # 8192

    # Fused QKVZ projection
    qkvz = hidden_states @ p["in_proj_qkvz.weight"].T  # [B, T, 12288]
    q_raw = qkvz[:, :, :q_dim]
    k_raw = qkvz[:, :, q_dim:q_dim + k_dim]
    v_raw = qkvz[:, :, q_dim + k_dim:q_dim + k_dim + v_dim]
    z_raw = qkvz[:, :, q_dim + k_dim + v_dim:]  # gate for output

    # Causal conv1d on q, k, v (concatenated)
    qkv_cat = jnp.concatenate([q_raw, k_raw, v_raw], axis=-1)  # [B, T, 8192]
    conv_w = p["conv1d.weight"]  # [8192, 1, 4]
    conv_w = conv_w.squeeze(1)   # [8192, 4]
    kernel_size = conv_w.shape[-1]
    # Causal pad + depthwise conv
    qkv_padded = jnp.concatenate([jnp.zeros((B, kernel_size - 1, qkv_dim), dtype=qkv_cat.dtype), qkv_cat], axis=1)
    windows = jnp.stack([qkv_padded[:, i:i + T, :] for i in range(kernel_size)], axis=2)
    qkv_conv = jnp.sum(windows * conv_w.T[None, None, :, :].transpose(0, 1, 2, 3), axis=2)

    q = jax.nn.silu(qkv_conv[:, :, :q_dim])
    k = jax.nn.silu(qkv_conv[:, :, q_dim:q_dim + k_dim])
    v = qkv_conv[:, :, q_dim + k_dim:]  # no activation on v

    # Beta and dt projections
    ba = hidden_states @ p["in_proj_ba.weight"].T  # [B, T, 64]
    beta = jax.nn.sigmoid(ba[:, :, :n_v])  # [B, T, 32]
    dt_raw = ba[:, :, n_v:]                # [B, T, 32]

    # Decay: A = -exp(A_log), dt = softplus(dt_raw + dt_bias)
    A = -jnp.exp(p["A_log"].astype(jnp.float32))
    dt = jax.nn.softplus(dt_raw + p["dt_bias"])
    decay = jnp.exp(A[None, None, :] * dt)

    # Reshape to heads
    q = q.reshape(B, T, n_k, d_k)
    k = k.reshape(B, T, n_k, d_k)
    v = v.reshape(B, T, n_v, d_v)

    # Sequential scan over time
    q = q.transpose(0, 2, 1, 3).astype(jnp.float32)
    k = k.transpose(0, 2, 1, 3).astype(jnp.float32)
    v = v.transpose(0, 2, 1, 3).astype(jnp.float32)
    beta_t = beta.transpose(0, 2, 1).astype(jnp.float32)
    decay_t = decay.transpose(0, 2, 1).astype(jnp.float32)

    # Broadcast k heads to v heads (n_v / n_k = 2)
    head_ratio = n_v // n_k
    k_v = jnp.repeat(k, head_ratio, axis=1)

    def scan_fn(state, t_inputs):
        k_t, v_t, beta_t, decay_t = t_inputs
        kv_mem = jnp.einsum("bnd,bnde->bne", k_t, state)
        delta_v = v_t - kv_mem
        update = jnp.einsum("bnd,bne->bnde", k_t, delta_v) * beta_t[:, :, None, None]
        state = state * decay_t[:, :, None, None] + update
        return state, state

    init_state = jnp.zeros((B, n_v, d_k, d_v), dtype=jnp.float32)

    k_scan = jnp.moveaxis(k_v, 2, 0)
    v_scan = jnp.moveaxis(v, 2, 0)
    beta_scan = jnp.moveaxis(beta_t, 2, 0)
    decay_scan = jnp.moveaxis(decay_t, 2, 0)

    _, all_states = lax.scan(scan_fn, init_state, (k_scan, v_scan, beta_scan, decay_scan))

    q_v = jnp.repeat(q, head_ratio, axis=1)
    q_scan = jnp.moveaxis(q_v, 2, 0)
    output = jnp.einsum("tbnk,tbnkd->tbnd", q_scan, all_states)
    output = jnp.moveaxis(output, 0, 2)

    output = output.transpose(0, 2, 1, 3).reshape(B, T, n_v * d_v).astype(hidden_states.dtype)

    # Output: group norm, gate with z, then project
    norm_w = p["norm.weight"]
    output_grouped = output.reshape(B, T, n_v, d_v)
    output_normed = _per_head_rms_norm(output_grouped, norm_w, cfg.rms_norm_eps)
    output = output_normed.reshape(B, T, n_v * d_v)

    z = jax.nn.silu(z_raw.astype(jnp.float32)).astype(output.dtype)
    output = output * z

    output = output @ p["out_proj.weight"].T
    return output


# ---------------------------------------------------------------------------
# MoE (Qwen3-Next specific: shared expert gate)
# ---------------------------------------------------------------------------

def _qwen3_moe_forward(hidden_states, p, cfg):
    """MoE forward with Qwen3-Next shared expert gating."""
    return _moe_forward(
        hidden_states, p,
        cfg.num_experts_per_tok,
        routed_scaling_factor=1.0,  # Qwen3-Next doesn't use scaling
        norm_topk_prob=cfg.norm_topk_prob,
        has_shared_expert_gate=True,
    )


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

def decoder_layer_forward(hidden_states, attention_mask, cos, sin,
                          layer_idx, params, cfg):
    pfx = f"model.layers.{layer_idx}"

    normed = rms_norm(hidden_states, params[f"{pfx}.input_layernorm.weight"], cfg.rms_norm_eps)

    if cfg.is_attention_layer(layer_idx):
        attn_p = {k.removeprefix(f"{pfx}.self_attn."): v
                  for k, v in params.items()
                  if k.startswith(f"{pfx}.self_attn.")}
        attn_out = attention_forward(normed, attention_mask, cos, sin, attn_p, cfg)
    else:
        gdn_p = {k.removeprefix(f"{pfx}.linear_attn."): v
                 for k, v in params.items()
                 if k.startswith(f"{pfx}.linear_attn.")}
        attn_out = gdn_forward(normed, gdn_p, cfg)

    hidden_states = hidden_states + attn_out

    mlp_in = rms_norm(hidden_states, params[f"{pfx}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    moe_p = {k.removeprefix(f"{pfx}.mlp."): v
             for k, v in params.items()
             if k.startswith(f"{pfx}.mlp.")}
    mlp_out = _qwen3_moe_forward(mlp_in, moe_p, cfg)

    return hidden_states + mlp_out


# ---------------------------------------------------------------------------
# Full model forward pass
# ---------------------------------------------------------------------------

def qwen3_next_forward(
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
    params: dict,
    cfg: Qwen3NextConfig,
    aux_layer_indices: Optional[set] = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Full Qwen3-Coder-Next forward pass (training mode, no KV cache).

    Returns:
        last_hidden_state    [B, T, H]
        embed_tokens_weight  [vocab_size, H]
        aux_hidden_states    [B, T, 3*H]
        target_logits        [B, T, vocab_size]
    """
    B, T = input_ids.shape
    num_layers = cfg.num_hidden_layers

    if aux_layer_indices is None:
        aux_layer_indices = {3, num_layers // 2 - 1, num_layers - 1}

    embed_w = params["model.embed_tokens.weight"]
    hidden_states = embed_w[input_ids]

    rope_dim = cfg.rope_head_dim
    cos, sin = build_rope_freqs(rope_dim, T, cfg.rope_theta)
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
