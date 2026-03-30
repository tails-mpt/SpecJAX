"""RMSNorm variants used across all SpecJAX models."""

import jax.numpy as jnp
from jax import lax


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """RMSNorm over last dimension. Works on any rank (3D hidden, 4D per-head)."""
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
    x_norm = x_f32 * lax.rsqrt(var + eps)
    return (weight * x_norm).astype(x.dtype)
