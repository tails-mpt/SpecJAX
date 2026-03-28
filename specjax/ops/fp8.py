"""FP8 dequantization routines for quantized target models."""

import ml_dtypes
import numpy as np
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# Block-wise FP8 (GLM-5-FP8, Qwen3-Next)
# ---------------------------------------------------------------------------

def dequant_fp8_block(
    w_fp8: np.ndarray,
    scale: np.ndarray,
    block_h: int = 128,
    block_w: int = 128,
) -> np.ndarray:
    """
    Block-wise FP8 e4m3 -> bfloat16 dequantization (vectorized).

    Args:
        w_fp8: [M, N] in float8_e4m3fn (or uint8 view)
        scale: [M//block_h, N//block_w] in float32
        block_h, block_w: block dimensions (default 128x128)

    Returns:
        [M, N] in bfloat16
    """
    w_f32 = w_fp8.astype(np.float32)
    M, N = w_f32.shape

    # Handle non-divisible dimensions by padding
    pad_h = (block_h - M % block_h) % block_h
    pad_w = (block_w - N % block_w) % block_w
    if pad_h > 0 or pad_w > 0:
        w_f32 = np.pad(w_f32, ((0, pad_h), (0, pad_w)), mode="constant")

    Mp, Np = w_f32.shape
    bh, bw = Mp // block_h, Np // block_w

    # Reshape into blocks, multiply by per-block scale, reshape back
    w_blocks = w_f32.reshape(bh, block_h, bw, block_w)
    scale_exp = scale[:bh, :bw, np.newaxis, np.newaxis]   # [bh, bw, 1, 1]
    scale_bc = np.transpose(scale_exp, (0, 2, 1, 3))      # [bh, 1, bw, 1]
    w_dequant = (w_blocks * scale_bc).reshape(Mp, Np)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        w_dequant = w_dequant[:M, :N]

    return w_dequant.astype(ml_dtypes.bfloat16)


def dequant_fp8_1d(w_fp8: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize a 1D FP8 tensor using a scalar or per-block scale."""
    w_f32 = w_fp8.astype(np.float32)
    if scale.ndim == 0 or scale.size == 1:
        return (w_f32 * float(scale)).astype(ml_dtypes.bfloat16)
    # Per-block 1D: treat as [1, N] and dequant
    return dequant_fp8_block(
        w_f32.reshape(1, -1), scale.reshape(1, -1), block_h=1, block_w=128
    ).reshape(-1)


# ---------------------------------------------------------------------------
# Channel-wise FP8 (GLM-4.7-FP8)
# ---------------------------------------------------------------------------

def dequant_fp8_channel(w_fp8: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Channel-wise FP8 e4m3 -> bfloat16 dequantization.

    For compressed-tensors format with strategy="channel":
    scale shape is [out_features, 1] (2D column vector).
    dequant: w[i,j] = w_fp8[i,j] * scale[i, 0]
    """
    w_f32 = w_fp8.astype(np.float32)
    scale_1d = scale.squeeze()
    if scale_1d.ndim == 0:
        return (w_f32 * float(scale_1d)).astype(ml_dtypes.bfloat16)
    return (w_f32 * scale_1d[:, np.newaxis]).astype(ml_dtypes.bfloat16)


# ---------------------------------------------------------------------------
# Qwen3-Next block-wise FP8 (scale_inv variant)
# ---------------------------------------------------------------------------

def dequant_fp8_qwen(
    weight: np.ndarray,
    scale_inv: np.ndarray,
    block_size: int = 128,
) -> np.ndarray:
    """
    Block-wise FP8 -> bf16 dequantization (Qwen3-Next format).

    weight: [out, in] float8_e4m3fn
    scale_inv: [ceil(out/128), ceil(in/128)] bfloat16

    Returns: [out, in] bfloat16
    """
    out_dim, in_dim = weight.shape
    w_f32 = weight.astype(np.float32)

    # Expand scale_inv to match weight shape via block repetition
    s = scale_inv.astype(np.float32)
    s = np.repeat(s, block_size, axis=0)[:out_dim]
    s = np.repeat(s, block_size, axis=1)[:, :in_dim]

    return (w_f32 * s).astype(ml_dtypes.bfloat16)


# ---------------------------------------------------------------------------
# JIT-friendly FP8 dequant (for MoE expert weights during forward pass)
# ---------------------------------------------------------------------------

def dequant_expert_jit(
    w_uint8: jnp.ndarray,   # [E, out, in] stored as uint8 (raw FP8 bytes)
    scale: jnp.ndarray,     # [E, out, 1] channel-wise scale
) -> jnp.ndarray:
    """JIT-friendly FP8->bf16 dequantization for MoE expert weights.

    FP8 e4m3 values are stored as uint8 on TPU (no native FP8 support on v4).
    We reinterpret the uint8 bit pattern as float8_e4m3fn, convert to f32,
    multiply by the channel-wise scale, and return bf16.

    On XLA/TPU this compiles to efficient element-wise ops.
    """
    w_f32 = lax.bitcast_convert_type(w_uint8, jnp.float8_e4m3fn).astype(jnp.float32)
    return (w_f32 * scale).astype(jnp.bfloat16)
