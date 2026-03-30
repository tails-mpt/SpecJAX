"""Shared pure-function building blocks for SpecJAX models."""

from specjax.ops.norm import rms_norm
from specjax.ops.rope import build_rope_freqs, rotate_half, apply_rope_interleaved, apply_partial_rope
from specjax.ops.moe import topk_router, moe_experts_forward, moe_forward, mlp_forward
from specjax.ops.fp8 import (
    dequant_fp8_block,
    dequant_fp8_1d,
    dequant_fp8_channel,
    dequant_fp8_qwen,
    dequant_expert_jit,
)
from specjax.ops.loading import discover_shards, stack_moe_experts
