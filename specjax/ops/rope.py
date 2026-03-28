"""Rotary position embedding helpers shared across all SpecJAX models."""

import jax.numpy as jnp


def build_rope_freqs(rope_head_dim: int, max_seq_len: int, theta: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns cos/sin tables [max_seq_len, rope_head_dim]."""
    half = rope_head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)              # [T, half]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # [T, rope_head_dim]
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Standard rotate-half for RoPE: [-x2, x1]."""
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope_interleaved(
    q_rope: jnp.ndarray,   # [B, heads, T, rope_head_dim]
    k_rope: jnp.ndarray,   # [B, kv_heads, T, rope_head_dim]
    cos: jnp.ndarray,      # [T, rope_head_dim]
    sin: jnp.ndarray,      # [T, rope_head_dim]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Interleaved RoPE: cos/sin are broadcast over batch and head dims.

    GLM uses interleaved layout: within each head the rope dims are stored as
    [x0, x1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}] and rotated as pairs
    (x_i, x_{i+d/2}). This is equivalent to standard rotate_half applied to the
    rope slice of the head dimension.
    """
    cos = cos[None, None, :, :]   # [1, 1, T, D]
    sin = sin[None, None, :, :]
    q_rope = q_rope * cos + rotate_half(q_rope) * sin
    k_rope = k_rope * cos + rotate_half(k_rope) * sin
    return q_rope, k_rope


def apply_partial_rope(
    q: jnp.ndarray,   # [B, heads, T, head_dim]
    k: jnp.ndarray,   # [B, kv_heads, T, head_dim]
    cos: jnp.ndarray,      # [T, rope_dim]
    sin: jnp.ndarray,      # [T, rope_dim]
    rope_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE only to the first rope_dim dimensions, pass-through the rest."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rope, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    q_rope = q_rope * cos + rotate_half(q_rope) * sin
    q = jnp.concatenate([q_rope, q_pass], axis=-1)
    k_rope, k_pass = k[..., :rope_dim], k[..., rope_dim:]
    k_rope = k_rope * cos + rotate_half(k_rope) * sin
    k = jnp.concatenate([k_rope, k_pass], axis=-1)
    return q, k
