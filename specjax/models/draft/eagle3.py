"""
EAGLE-3 Draft Model — pure JAX implementation.

Matches the SpecForge / SafeAILab EAGLE-3 architecture:

  fc.weight                                [2048, 6144]   projects [low_h || mid_h || high_h]
  midlayer.input_layernorm.weight          [2048]         normalises token embeddings
  midlayer.hidden_norm.weight              [2048]         normalises projected features
  midlayer.self_attn.q_proj.weight         [2048, 4096]   input = [emb_norm || hidden_norm]
  midlayer.self_attn.k_proj.weight         [512,  4096]
  midlayer.self_attn.v_proj.weight         [512,  4096]
  midlayer.self_attn.o_proj.weight         [2048, 2048]
  midlayer.post_attention_layernorm.weight [2048]
  midlayer.mlp.gate_proj.weight            [8192, 2048]
  midlayer.mlp.up_proj.weight              [8192, 2048]
  midlayer.mlp.down_proj.weight            [2048, 8192]
  norm.weight                              [2048]
  lm_head.weight                           [32000, 2048]  draft vocab (not 154880)
  t2d                                      [154880]  bool   target->draft mapping
  d2t                                      [32000]   int64  draft->target mapping

Architecture (matching SpecForge OnlineEagle3Model):
  1. FC projects multi-layer features [low||mid||high] (3H) -> H
  2. Token embedding via embed_tokens (separate path)
  3. Midlayer attention: Q,K,V from cat(emb_norm, hidden_norm) (2H)
     with multi-branch tree KV cache across TTT steps
  4. MLP (SiLU-gated)
  5. Norm + lm_head -> draft vocab logits

Loss: KL divergence vs target model's softmax distribution (not CE vs ground truth).
"""

import json
import logging
import os
import shutil
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from specjax.ops.norm import rms_norm
from specjax.ops.rope import build_rope_freqs, rotate_half

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Eagle3Config:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 154_880
    draft_vocab_size: int = 32_000
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-5
    ttt_decay: float = 0.8

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: dict) -> "Eagle3Config":
        field_map = {
            "hidden_size": "hidden_size",
            "intermediate_size": "intermediate_size",
            "num_attention_heads": "num_heads",
            "num_key_value_heads": "num_kv_heads",
            "head_dim": "head_dim",
            "vocab_size": "vocab_size",
            "draft_vocab_size": "draft_vocab_size",
            "rope_theta": "rope_theta",
            "rms_norm_eps": "rms_norm_eps",
        }
        kwargs = {}
        for src, dst in field_map.items():
            if src in d:
                kwargs[dst] = d[src]
        # Handle nested rope_parameters (HuggingFace config format)
        if "rope_theta" not in kwargs and "rope_parameters" in d:
            rp = d["rope_parameters"]
            if isinstance(rp, dict) and "rope_theta" in rp:
                kwargs["rope_theta"] = rp["rope_theta"]
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Weight initialisation helpers
# ---------------------------------------------------------------------------

def _xavier_uniform(key: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray:
    fan_in, fan_out = shape[-1], shape[0]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=jnp.bfloat16)


def init_eagle3_params(cfg: Eagle3Config, key: jax.random.PRNGKey) -> dict:
    """
    Initialise Eagle3 params from scratch with Xavier uniform on linear layers,
    ones on norms. Returns a flat dict matching the safetensors weight names.
    """
    keys = jax.random.split(key, 20)
    ki = iter(keys)

    H = cfg.hidden_size
    I = cfg.intermediate_size
    n_h = cfg.num_heads
    n_kv = cfg.num_kv_heads
    d = cfg.head_dim
    V_d = cfg.draft_vocab_size
    attn_in = 2 * H

    params = {
        "fc.weight":                                _xavier_uniform(next(ki), (H, 3 * H)),
        "midlayer.input_layernorm.weight":           jnp.ones(H, dtype=jnp.bfloat16),
        "midlayer.hidden_norm.weight":               jnp.ones(H, dtype=jnp.bfloat16),
        "midlayer.self_attn.q_proj.weight":         _xavier_uniform(next(ki), (n_h * d, attn_in)),
        "midlayer.self_attn.k_proj.weight":         _xavier_uniform(next(ki), (n_kv * d, attn_in)),
        "midlayer.self_attn.v_proj.weight":         _xavier_uniform(next(ki), (n_kv * d, attn_in)),
        "midlayer.self_attn.o_proj.weight":         _xavier_uniform(next(ki), (H, n_h * d)),
        "midlayer.post_attention_layernorm.weight":  jnp.ones(H, dtype=jnp.bfloat16),
        "midlayer.mlp.gate_proj.weight":            _xavier_uniform(next(ki), (I, H)),
        "midlayer.mlp.up_proj.weight":              _xavier_uniform(next(ki), (I, H)),
        "midlayer.mlp.down_proj.weight":            _xavier_uniform(next(ki), (H, I)),
        "norm.weight":                               jnp.ones(H, dtype=jnp.bfloat16),
        "lm_head.weight":                           _xavier_uniform(next(ki), (V_d, H)),
    }
    return params


def load_eagle3_params(
    checkpoint_path: Optional[str],
    key: Optional[jax.random.PRNGKey] = None,
    config_override: Optional[Eagle3Config] = None,
) -> tuple[dict, dict, Eagle3Config]:
    """
    Load Eagle3 params from an existing checkpoint, or initialise from scratch.

    Args:
        checkpoint_path: path to existing checkpoint dir, or None for from-scratch.
        key: PRNG key for random initialisation.
        config_override: if provided and checkpoint_path is None, use this config
            instead of default Eagle3Config(). This allows matching the draft
            model dimensions to the target model when training from scratch.

    Returns:
        params:  flat dict of trainable parameters {name: jnp.ndarray bfloat16}
        buffers: {"t2d": jnp.ndarray bool, "d2t": jnp.ndarray int64}
        config:  Eagle3Config
    """
    if checkpoint_path is not None:
        cfg_path = os.path.join(checkpoint_path, "config.json")
        cfg = Eagle3Config()
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = Eagle3Config.from_dict(json.load(f))

        weights_path = os.path.join(checkpoint_path, "model.safetensors")
        raw = load_file(weights_path)

        params = {}
        buffers = {}
        for k, v in raw.items():
            if k in ("t2d", "d2t"):
                buffers[k] = jnp.array(v)
            else:
                params[k] = jnp.array(v, dtype=jnp.bfloat16)

        logger.warning(
            f"Eagle3: loaded {len(params)} trainable params + {len(buffers)} buffers "
            f"from {checkpoint_path}"
        )
    else:
        if key is None:
            key = jax.random.PRNGKey(42)
        cfg = config_override if config_override is not None else Eagle3Config()
        params = init_eagle3_params(cfg, key)
        buffers = {
            "t2d": jnp.zeros(cfg.vocab_size, dtype=jnp.bool_),
            "d2t": jnp.zeros(cfg.draft_vocab_size, dtype=jnp.int64),
        }
        logger.warning(
            f"Eagle3: initialised from scratch (Xavier uniform) "
            f"hidden_size={cfg.hidden_size} vocab_size={cfg.vocab_size}"
        )

    return params, buffers, cfg


# ---------------------------------------------------------------------------
# RMSNorm / RoPE helpers — delegated to specjax.ops.{norm,rope}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Eagle3 forward pass — matches SpecForge architecture
# ---------------------------------------------------------------------------

def eagle3_attention_forward(
    combined: jnp.ndarray,            # [B, T, 2H]
    p: dict,                          # attention sub-dict
    cfg: Eagle3Config,
    cache_k: list | None = None,      # list of [B, n_h, T, d] from previous TTT steps
    cache_v: list | None = None,
    position_offset: int = 0,
) -> tuple[jnp.ndarray, list | None, list | None]:
    """
    Self-attention with multi-branch tree KV cache for TTT.

    At TTT step 0: standard causal attention (cache empty).
    At TTT step k>0: attend to step 0's full KV (causal) plus scalar
    branch weights to steps 1..k (matching SpecForge's tree attention).

    Returns: (attn_output [B, T, H], updated_cache_k, updated_cache_v)
    """
    B, T, _ = combined.shape
    n_h = cfg.num_heads
    n_kv = cfg.num_kv_heads
    d = cfg.head_dim
    n_groups = n_h // n_kv

    q = combined @ p["q_proj.weight"].T    # [B, T, n_h*d]
    k = combined @ p["k_proj.weight"].T    # [B, T, n_kv*d]
    v = combined @ p["v_proj.weight"].T    # [B, T, n_kv*d]

    q = q.reshape(B, T, n_h, d).transpose(0, 2, 1, 3)    # [B, n_h, T, d]
    k = k.reshape(B, T, n_kv, d).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_kv, d).transpose(0, 2, 1, 3)

    # RoPE with position offset for TTT steps
    total_len = T + position_offset
    cos, sin = build_rope_freqs(d, total_len, cfg.rope_theta)
    cos_slice = cos[position_offset:position_offset + T][None, None]
    sin_slice = sin[position_offset:position_offset + T][None, None]
    q = q * cos_slice + rotate_half(q) * sin_slice
    k = k * cos_slice + rotate_half(k) * sin_slice

    # GQA: expand KV heads
    if n_groups > 1:
        k = jnp.repeat(k, n_groups, axis=1)
        v = jnp.repeat(v, n_groups, axis=1)

    scale = d ** -0.5

    if cache_k is None:
        # No cache mode (single-step / inference) — standard causal attention
        attn_w = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn_w = jnp.where(causal[None, None], attn_w, jnp.finfo(jnp.float32).min)
        attn_p = jax.nn.softmax(attn_w.astype(jnp.float32), axis=-1).astype(combined.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn_p, v)
        new_cache_k, new_cache_v = None, None

    elif len(cache_k) == 0:
        # TTT step 0: standard causal attention, start the cache
        attn_w = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn_w = jnp.where(causal[None, None], attn_w, jnp.finfo(jnp.float32).min)
        attn_p = jax.nn.softmax(attn_w.astype(jnp.float32), axis=-1).astype(combined.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn_p, v)
        new_cache_k = [k]
        new_cache_v = [v]

    else:
        # TTT step k>0: multi-branch tree attention
        # Append current K,V to cache
        new_cache_k = cache_k + [k]
        new_cache_v = cache_v + [v]

        # Main branch: full causal attention to step 0's KV
        k0 = new_cache_k[0]
        v0 = new_cache_v[0]
        attn_w = jnp.einsum("bhqd,bhkd->bhqk", q, k0) * scale  # [B, n_h, T, T]
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn_w = jnp.where(causal[None, None], attn_w, jnp.finfo(jnp.float32).min)

        # Branch attention: scalar dot products with steps 1..current
        branch_weights = []
        for i in range(1, len(new_cache_k)):
            ki = new_cache_k[i]
            # Per-position scalar weight: dot(q, ki) summed over head_dim
            wi = jnp.sum(q * ki, axis=-1, keepdims=True) * scale  # [B, n_h, T, 1]
            branch_weights.append(wi)

        if branch_weights:
            branch_w = jnp.concatenate(branch_weights, axis=-1)  # [B, n_h, T, num_branches]
            attn_w = jnp.concatenate([attn_w, branch_w], axis=-1)

        # Softmax over all positions (main branch + tree branches)
        attn_p = jax.nn.softmax(attn_w.astype(jnp.float32), axis=-1).astype(combined.dtype)

        # Weighted sum of values
        attn_p0 = attn_p[..., :T]
        out = jnp.einsum("bhqk,bhkd->bhqd", attn_p0, v0)
        for i in range(1, len(new_cache_k)):
            vi = new_cache_v[i]
            wi = attn_p[..., T + i - 1:T + i]   # [B, n_h, T, 1]
            out = out + wi * vi

    out = out.transpose(0, 2, 1, 3).reshape(B, T, n_h * d)
    return out @ p["o_proj.weight"].T, new_cache_k, new_cache_v


def eagle3_forward(
    params: dict,
    hidden_states: jnp.ndarray,     # [B, T, H] — projected multi-layer features (after FC)
    input_ids: jnp.ndarray,         # [B, T]
    embed_w: jnp.ndarray,           # [V, H]  embedding table
    cfg: Eagle3Config,
    cache_k: list | None = None,
    cache_v: list | None = None,
    position_offset: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, list | None, list | None]:
    """
    EAGLE-3 draft model forward pass matching SpecForge architecture.

    Args:
        hidden_states: projected multi-layer features [B, T, H]
        input_ids: token IDs for embedding lookup [B, T]
        embed_w: target model's embedding table [V, H]
        cache_k, cache_v: multi-branch KV cache from previous TTT steps
        position_offset: RoPE offset for TTT step (= len(cache_k) or 0)

    Returns: (logits [B, T, V_draft], out_hidden [B, T, H],
              new_cache_k, new_cache_v)
    """
    eps = cfg.rms_norm_eps

    # Token embedding (separate path, matching SpecForge)
    inputs_embeds = embed_w[input_ids]   # [B, T, H]

    # Midlayer: attention sublayer
    residual = hidden_states
    hidden_norm = rms_norm(hidden_states, params["midlayer.hidden_norm.weight"], eps)
    input_emb_norm = rms_norm(inputs_embeds, params["midlayer.input_layernorm.weight"], eps)
    combined = jnp.concatenate([input_emb_norm, hidden_norm], axis=-1)  # [B, T, 2H]

    attn_p = {k[len("midlayer.self_attn."):]: v
              for k, v in params.items() if k.startswith("midlayer.self_attn.")}
    attn_out, new_cache_k, new_cache_v = eagle3_attention_forward(
        combined, attn_p, cfg, cache_k, cache_v, position_offset,
    )
    hidden_states = residual + attn_out

    # Midlayer: MLP sublayer
    residual = hidden_states
    x_post_norm = rms_norm(hidden_states, params["midlayer.post_attention_layernorm.weight"], eps)
    gate = x_post_norm @ params["midlayer.mlp.gate_proj.weight"].T
    up   = x_post_norm @ params["midlayer.mlp.up_proj.weight"].T
    mlp_out = (jax.nn.silu(gate) * up) @ params["midlayer.mlp.down_proj.weight"].T
    hidden_states = residual + mlp_out

    # Output: norm + lm_head
    out = rms_norm(hidden_states, params["norm.weight"], eps)
    logits = out @ params["lm_head.weight"].T
    return logits, hidden_states, new_cache_k, new_cache_v


# ---------------------------------------------------------------------------
# Loss — KL divergence vs target model distribution (matching SpecForge)
# ---------------------------------------------------------------------------

def _compute_target_p(
    target_logits: jnp.ndarray,    # [B, T, V_target]
    d2t: jnp.ndarray,             # [V_draft] int64 — draft->target index
    t2d: jnp.ndarray,             # [V_target] int32 — target->draft index (-1 for OOV)
    attention_mask: jnp.ndarray,  # [B, T]
    offset: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute target probability distribution in draft vocab space.

    Mirrors SpecForge's _compute_target_p: select draft vocab subset from
    target logits, softmax, and build position mask (only where target argmax
    is in draft vocab).

    Returns:
        target_p:      [B, T-offset, V_draft] softmax probabilities
        position_mask: [B, T-offset] bool — valid positions for loss
        n_valid:       scalar — number of valid positions
    """
    # Shift for offset (TTT step k uses offset=k+1)
    shift_logits = target_logits[:, :-offset, :]   # [B, T-offset, V_target]
    shift_mask = attention_mask[:, offset:]          # [B, T-offset]

    # Target argmax in full vocab, mapped to draft vocab
    target_argmax = jnp.argmax(shift_logits, axis=-1)   # [B, T-offset]
    target_argmax_draft = t2d[target_argmax]              # -1 for OOV
    position_mask = (target_argmax_draft >= 0) & shift_mask.astype(jnp.bool_)

    # Select draft vocab subset: target_logits[:, :, d2t] -> [B, T-offset, V_draft]
    target_logits_draft = shift_logits[:, :, d2t]
    target_p = jax.nn.softmax(target_logits_draft.astype(jnp.float32), axis=-1)

    n_valid = jnp.maximum(position_mask.sum().astype(jnp.float32), 1.0)
    return target_p, position_mask, n_valid


def _kl_loss_and_acc(
    draft_logits: jnp.ndarray,    # [B, T, V_draft]
    target_logits: jnp.ndarray,   # [B, T, V_target]
    d2t: jnp.ndarray,             # [V_draft] int64
    t2d: jnp.ndarray,             # [V_target] int32
    attention_mask: jnp.ndarray,  # [B, T]
    offset: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    KL divergence loss + top-1 accuracy for one TTT step.

    Loss: -sum(target_p * log_softmax(draft_logits))  (matching SpecForge)
    Acc:  draft_argmax == target_argmax  (in draft vocab space)

    offset=1  -> standard next-token (TTT step 0)
    offset=k+1 -> k+1 positions ahead (TTT step k)
    """
    V_d = draft_logits.shape[-1]

    # Shift draft logits by offset
    shift_draft = draft_logits[:, :-offset, :]   # [B, T-offset, V_draft]

    # Compute target distribution and position mask
    target_p, position_mask, n_valid = _compute_target_p(
        target_logits, d2t, t2d, attention_mask, offset,
    )

    # KL loss: -sum(target_p * log_softmax(draft_logits))
    # Normalize by total positions (matching SpecForge LogSoftmaxLoss .mean() over B*T)
    draft_log_p = jax.nn.log_softmax(shift_draft.astype(jnp.float32), axis=-1)
    per_position_loss = -jnp.sum(target_p * draft_log_p, axis=-1)  # [B, T-offset]
    total_positions = jnp.float32(shift_draft.shape[0] * shift_draft.shape[1])
    loss = jnp.sum(jnp.where(position_mask, per_position_loss, 0.0)) / total_positions

    # Accuracy: draft_argmax == target_argmax (in draft vocab)
    draft_top1 = jnp.argmax(shift_draft, axis=-1)           # [B, T-offset]
    target_top1 = jnp.argmax(target_p, axis=-1)             # [B, T-offset]
    correct = (draft_top1 == target_top1).astype(jnp.float32)
    acc = jnp.sum(jnp.where(position_mask, correct, 0.0)) / total_positions

    return loss, acc


def compute_loss(
    logits: jnp.ndarray,           # [B, T, V_draft]
    target_logits: jnp.ndarray,    # [B, T, V_target]
    d2t: jnp.ndarray,              # [V_draft] int64
    t2d: jnp.ndarray,              # [V_target] int32
    attention_mask: jnp.ndarray,   # [B, T]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single-step KL divergence loss (next-token prediction).
    Returns: (loss scalar, accuracy)
    """
    return _kl_loss_and_acc(logits, target_logits, d2t, t2d, attention_mask, offset=1)


# ---------------------------------------------------------------------------
# TTT (multi-step) loss — matches SpecForge OnlineEagle3Model.forward
# ---------------------------------------------------------------------------

def _padding_right(x: jnp.ndarray) -> jnp.ndarray:
    """Shift sequence left by 1 (drop first, pad zero on right). Matches SpecForge padding(x, left=False)."""
    return jnp.concatenate([x[:, 1:], jnp.zeros_like(x[:, :1])], axis=1)


def compute_ttt_loss(
    params: dict,
    aux_hidden_states: jnp.ndarray,   # [B, T, 3*H] from glm_forward
    embed_w: jnp.ndarray,             # [V, H]
    input_ids: jnp.ndarray,           # [B, T]
    attention_mask: jnp.ndarray,      # [B, T]
    target_logits: jnp.ndarray,       # [B, T, V_target]
    d2t: jnp.ndarray,                 # [V_draft] int64
    t2d: jnp.ndarray,                 # [V_target] int32
    cfg: Eagle3Config,
    ttt_length: int = 7,
) -> tuple[jnp.ndarray, tuple, tuple]:
    """
    Multi-step TTT rollout matching SpecForge.

    At each step k:
      - FC projects aux_hidden_states (step 0) or draft hidden (step k>0)
      - Token embedding from input_ids (shifted right by k each step)
      - Attention with multi-branch KV cache
      - KL loss: draft logits vs target logits (pre-shifted by offset)
      - Hidden state feeds back for next step

    Returns: (total_loss, plosses tuple[ttt_length], acces tuple[ttt_length])
    """
    # Project multi-layer features through FC (only once, at step 0)
    hidden_states = aux_hidden_states @ params["fc.weight"].T   # [B, T, H]

    # Pre-shift target logits for next-token: logits[t] predicts token at t+1
    target_logits_shifted = _padding_right(target_logits)

    # Pre-compute shifted target_p for each TTT step
    # Step k uses target_logits shifted by k additional positions
    # We pre-build a padded version: [B, T + ttt_length, V_target]
    B, T, V_t = target_logits_shifted.shape
    pad = jnp.zeros((B, ttt_length, V_t), dtype=target_logits_shifted.dtype)
    target_logits_padded = jnp.concatenate([target_logits_shifted, pad], axis=1)

    cur_input_ids = _padding_right(input_ids)     # shifted for next-token
    cur_attention_mask = attention_mask

    cache_k = []
    cache_v = []
    plosses = []
    acces = []

    for k in range(ttt_length):
        # Target distribution for this TTT step
        step_target_logits = target_logits_padded[:, k:k + T, :]

        # Forward through draft model
        logits, draft_hidden, cache_k, cache_v = eagle3_forward(
            params, hidden_states, cur_input_ids, embed_w, cfg,
            cache_k, cache_v, position_offset=k,
        )

        # KL loss and accuracy for this step
        # Use offset=1 since target_logits are already pre-shifted per step
        target_p, position_mask, n_valid = _compute_target_p(
            step_target_logits, d2t, t2d, cur_attention_mask, offset=1,
        )
        shift_draft = logits[:, :-1, :]
        draft_log_p = jax.nn.log_softmax(shift_draft.astype(jnp.float32), axis=-1)
        per_pos_loss = -jnp.sum(target_p * draft_log_p, axis=-1)
        total_positions = jnp.float32(shift_draft.shape[0] * shift_draft.shape[1])
        ploss_k = jnp.sum(jnp.where(position_mask, per_pos_loss, 0.0)) / total_positions

        draft_top1 = jnp.argmax(shift_draft, axis=-1)
        target_top1 = jnp.argmax(target_p, axis=-1)
        correct = (draft_top1 == target_top1).astype(jnp.float32)
        acc_k = jnp.sum(jnp.where(position_mask, correct, 0.0)) / total_positions

        plosses.append(ploss_k)
        acces.append(acc_k)

        # Feedback: draft hidden becomes next step's hidden_states
        hidden_states = draft_hidden

        # Shift input_ids and masks for next TTT step (matching SpecForge padding)
        if k < ttt_length - 1:
            cur_input_ids = _padding_right(cur_input_ids)
            cur_attention_mask = _padding_right(cur_attention_mask)

    # Geometric weighting: decay^0, decay^1, ..., decay^(k-1)
    decay = cfg.ttt_decay
    total_loss = sum(decay ** k * plosses[k] for k in range(ttt_length))

    return total_loss, tuple(plosses), tuple(acces)


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_eagle3_checkpoint(
    params: dict,
    buffers: dict,
    output_dir: str,
    source_config_path: Optional[str] = None,
    cfg: Optional[Eagle3Config] = None,
    is_primary: bool = True,
) -> None:
    """
    Save Eagle3 checkpoint compatible with inference servers.
    Writes model.safetensors (CPU numpy) + config.json.

    IMPORTANT: When TP spans multiple hosts (tp > local_device_count), ALL
    processes must call this function (process_allgather is a collective).
    Only the primary process writes files to disk.
    """
    from jax.experimental.multihost_utils import process_allgather

    def _to_numpy(v):
        """Convert JAX array to numpy, handling multi-host TP sharding."""
        if hasattr(v, 'addressable_shards') and not v.is_fully_addressable:
            v = process_allgather(v)
        return np.array(v)

    # All processes participate in gathering (collective op)
    state = {}
    for k, v in params.items():
        arr = _to_numpy(v)
        if arr.dtype == np.float32 and ("norm" in k or "layernorm" in k):
            import ml_dtypes
            arr = arr.astype(ml_dtypes.bfloat16)
        state[k] = arr
    for k, v in buffers.items():
        arr = _to_numpy(v)
        if k == "d2t":
            arr = arr - np.arange(len(arr), dtype=arr.dtype)
        state[k] = arr

    # Only primary process writes to disk
    if not is_primary:
        return

    from safetensors.numpy import save_file

    os.makedirs(output_dir, exist_ok=True)
    save_file(state, os.path.join(output_dir, "model.safetensors"))

    if source_config_path and os.path.exists(source_config_path):
        shutil.copy(source_config_path, os.path.join(output_dir, "config.json"))
    elif cfg is not None:
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": 1,
            "num_attention_heads": cfg.num_heads,
            "num_key_value_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim,
            "vocab_size": cfg.vocab_size,
            "draft_vocab_size": cfg.draft_vocab_size,
            "rope_theta": cfg.rope_theta,
            "rms_norm_eps": cfg.rms_norm_eps,
            "torch_dtype": "bfloat16",
            "tie_word_embeddings": False,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    logger.warning(f"[Eagle3] Saved checkpoint -> {output_dir}")


def validate_eagle3_checkpoint(output_dir: str) -> list[str]:
    """
    Validate an Eagle3 checkpoint for sglang-jax compatibility.

    Returns a list of warnings (empty if everything looks good).
    """
    from safetensors.numpy import load_file

    warnings = []
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    config_path = os.path.join(output_dir, "config.json")

    if not os.path.exists(safetensors_path):
        warnings.append(f"Missing model.safetensors in {output_dir}")
        return warnings
    if not os.path.exists(config_path):
        warnings.append(f"Missing config.json in {output_dir}")
        return warnings

    # Check config.json fields
    with open(config_path) as f:
        cfg = json.load(f)

    required_fields = [
        "architectures", "model_type", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "vocab_size", "draft_vocab_size", "rope_theta", "rms_norm_eps",
    ]
    for field in required_fields:
        if field not in cfg:
            warnings.append(f"config.json missing required field: {field}")

    if cfg.get("num_hidden_layers") != 1:
        warnings.append(f"num_hidden_layers should be 1, got {cfg.get('num_hidden_layers')}")

    if cfg.get("tie_word_embeddings") is not False:
        warnings.append("config.json should have tie_word_embeddings: false for EAGLE3 with draft vocab")

    # Check safetensors keys
    state = load_file(safetensors_path)
    expected_params = {
        "fc.weight", "midlayer.input_layernorm.weight", "midlayer.hidden_norm.weight",
        "midlayer.self_attn.q_proj.weight", "midlayer.self_attn.k_proj.weight",
        "midlayer.self_attn.v_proj.weight", "midlayer.self_attn.o_proj.weight",
        "midlayer.post_attention_layernorm.weight",
        "midlayer.mlp.gate_proj.weight", "midlayer.mlp.up_proj.weight",
        "midlayer.mlp.down_proj.weight", "norm.weight", "lm_head.weight",
        "d2t", "t2d",
    }
    missing = expected_params - set(state.keys())
    extra = set(state.keys()) - expected_params
    if missing:
        warnings.append(f"Missing keys in safetensors: {missing}")
    if extra:
        warnings.append(f"Unexpected keys in safetensors: {extra}")

    # Check d2t shape and values
    if "d2t" in state:
        d2t = state["d2t"]
        draft_vocab = cfg.get("draft_vocab_size", 0)
        if d2t.shape[0] != draft_vocab:
            warnings.append(f"d2t shape {d2t.shape} doesn't match draft_vocab_size {draft_vocab}")

    # Check weight shapes against config
    H = cfg.get("hidden_size", 0)
    I = cfg.get("intermediate_size", 0)
    n_h = cfg.get("num_attention_heads", 0)
    n_kv = cfg.get("num_key_value_heads", 0)
    d = cfg.get("head_dim", 0)
    V_d = cfg.get("draft_vocab_size", 0)

    shape_checks = {
        "fc.weight": (H, 3 * H),
        "midlayer.self_attn.q_proj.weight": (n_h * d, 2 * H),
        "midlayer.self_attn.k_proj.weight": (n_kv * d, 2 * H),
        "midlayer.self_attn.v_proj.weight": (n_kv * d, 2 * H),
        "midlayer.self_attn.o_proj.weight": (H, n_h * d),
        "midlayer.mlp.gate_proj.weight": (I, H),
        "midlayer.mlp.up_proj.weight": (I, H),
        "midlayer.mlp.down_proj.weight": (H, I),
        "lm_head.weight": (V_d, H),
    }
    for key, expected_shape in shape_checks.items():
        if key in state and state[key].shape != expected_shape:
            warnings.append(f"{key}: shape {state[key].shape} != expected {expected_shape}")

    if warnings:
        for w in warnings:
            logger.warning(f"[Eagle3 compat] {w}")
    else:
        logger.warning(f"[Eagle3 compat] Checkpoint OK — compatible with sglang-jax")

    return warnings


# ---------------------------------------------------------------------------
# Config factories — one per target model family
# ---------------------------------------------------------------------------

def eagle3_config_for_qwen2(
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int = 128,
    vocab_size: int = 152_064,
    draft_vocab_size: int = 32_000,
    rope_theta: float = 1_000_000.0,
    rms_norm_eps: float = 1e-5,
) -> Eagle3Config:
    """Generic Eagle3 config for any Qwen2/Qwen2.5-family target model."""
    return Eagle3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        draft_vocab_size=draft_vocab_size,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
    )


def eagle3_config_for_ds_r1_qwen_14b() -> Eagle3Config:
    """Eagle3 config for DeepSeek-R1-Distill-Qwen-14B."""
    return eagle3_config_for_qwen2(
        hidden_size=5120,
        intermediate_size=13824,
        num_heads=40,
        num_kv_heads=8,
    )


def eagle3_config_for_ds_r1_qwen_7b() -> Eagle3Config:
    """Eagle3 config for DeepSeek-R1-Distill-Qwen-7B."""
    return eagle3_config_for_qwen2(
        hidden_size=3584,
        intermediate_size=18944,
        num_heads=28,
        num_kv_heads=4,
    )
