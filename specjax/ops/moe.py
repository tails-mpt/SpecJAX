"""MoE routing and expert dispatch shared across all SpecJAX target models."""

from typing import Optional

import jax
import jax.numpy as jnp


def topk_router(
    hidden_states: jnp.ndarray,   # [B*T, H]
    gate_w: jnp.ndarray,          # [n_experts, H]
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool = True,
    gate_bias: Optional[jnp.ndarray] = None,  # [n_experts] fp32 correction bias
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sigmoid-based top-k router (noaux_tc method).

    Returns:
        routing_weights  [B*T, num_experts_per_tok]   (normalised, bfloat16)
        selected_experts [B*T, num_experts_per_tok]   (int32)
    """
    logits = hidden_states.astype(jnp.float32) @ gate_w.T  # [T, E]
    if gate_bias is not None:
        logits = logits + gate_bias.astype(jnp.float32)
    scores = jax.nn.sigmoid(logits)                         # [T, E]

    selected_experts = jnp.argsort(-scores, axis=-1)[..., :num_experts_per_tok]  # [T, K]
    routing_weights = jnp.take_along_axis(scores, selected_experts, axis=-1)  # [T, K]

    if norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(axis=-1, keepdims=True) + 1e-9)

    routing_weights = routing_weights * routed_scaling_factor
    return routing_weights.astype(jnp.bfloat16), selected_experts.astype(jnp.int32)


def moe_experts_forward(
    hidden_states: jnp.ndarray,         # [T, H]  (B*T flattened)
    routing_weights: jnp.ndarray,       # [T, K]
    selected_experts: jnp.ndarray,      # [T, K]  int32
    gate_up_proj: jnp.ndarray,          # [E, 2I, D]
    down_proj: jnp.ndarray,             # [E, D, I]  (D=hidden, I=moe_intermediate)
    gate_up_scale: Optional[jnp.ndarray] = None,  # [E, 2I, 1] for FP8 JIT dequant
    down_scale: Optional[jnp.ndarray] = None,     # [E, D, 1]  for FP8 JIT dequant
    act_fn=jax.nn.silu,
) -> jnp.ndarray:                       # [T, H]
    """
    Batched einsum MoE dispatch (JAX port of the static-shape patch).

    Processes all experts simultaneously to avoid data-dependent indexing
    (no jnp.nonzero, no dynamic gather that would require host sync).

    If scale arrays are provided, expert weights are dequantized from FP8
    just-in-time during the forward pass.
    """
    from specjax.ops.fp8 import dequant_expert_jit

    # JIT FP8 dequant if scales provided
    if gate_up_scale is not None:
        gate_up_proj = dequant_expert_jit(gate_up_proj, gate_up_scale)
    if down_scale is not None:
        down_proj = dequant_expert_jit(down_proj, down_scale)

    E = gate_up_proj.shape[0]

    # Expert selection weights [T, E] from one-hot aggregation
    one_hot = jax.nn.one_hot(selected_experts, num_classes=E, dtype=routing_weights.dtype)  # [T, K, E]
    token_expert_w = jnp.einsum("tke,tk->te", one_hot, routing_weights)   # [T, E]

    # All-expert gate+up projection [T, E, 2I]
    gate_up_all = jnp.einsum("td,eod->teo", hidden_states, gate_up_proj)
    I = gate_up_all.shape[-1] // 2
    gate_all = gate_up_all[..., :I]                  # [T, E, I]
    up_all   = gate_up_all[..., I:]                  # [T, E, I]
    mid_all  = act_fn(gate_all) * up_all             # [T, E, I]

    # All-expert down projection [T, E, D]
    out_all = jnp.einsum("tes,eos->teo", mid_all, down_proj)

    # Weighted sum over experts -> [T, D]
    return jnp.einsum("teo,te->to", out_all, token_expert_w)


def moe_forward(
    hidden_states: jnp.ndarray,   # [B, T, H]
    p: dict,                      # layer weights
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool = True,
    has_shared_expert_gate: bool = False,
) -> jnp.ndarray:
    """Complete MoE block: router + experts + shared expert."""
    B, T, H = hidden_states.shape
    flat = hidden_states.reshape(B * T, H)

    # Gate bias may or may not be present
    gate_bias = p.get("gate.e_score_correction_bias", None)

    routing_weights, selected_experts = topk_router(
        flat, p["gate.weight"], num_experts_per_tok,
        routed_scaling_factor, norm_topk_prob, gate_bias,
    )

    # Routed experts
    routed_out = moe_experts_forward(
        flat,
        routing_weights,
        selected_experts,
        p["experts.gate_up_proj"],   # [E, 2I, D]
        p["experts.down_proj"],      # [E, D, I]
        gate_up_scale=p.get("experts.gate_up_proj.scale"),
        down_scale=p.get("experts.down_proj.scale"),
    )

    # Shared expert (always active for all tokens)
    shared_gate = flat @ p["shared_expert.gate_proj.weight"].T
    shared_up   = flat @ p["shared_expert.up_proj.weight"].T
    shared_out  = (jax.nn.silu(shared_gate) * shared_up) @ p["shared_expert.down_proj.weight"].T

    # Optional shared expert gate (Qwen3-Next style)
    if has_shared_expert_gate and "shared_expert_gate.weight" in p:
        se_gate = jax.nn.sigmoid(flat @ p["shared_expert_gate.weight"].T)
        shared_out = shared_out * se_gate

    return (routed_out + shared_out).reshape(B, T, H)


def mlp_forward(x: jnp.ndarray, p: dict) -> jnp.ndarray:
    """SiLU-gated MLP: gate_proj, up_proj, down_proj."""
    gate = x @ p["gate_proj.weight"].T
    up   = x @ p["up_proj.weight"].T
    return (jax.nn.silu(gate) * up) @ p["down_proj.weight"].T
