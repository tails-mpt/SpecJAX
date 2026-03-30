"""Optimizer and LR schedule utilities for SpecJAX training."""

import optax


def cosine_warmup(peak_lr: float, warmup_steps: int, total_steps: int) -> optax.Schedule:
    """Cosine decay with linear warmup."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )


def build_optimizer(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Build AdamW optimizer with cosine warmup schedule and gradient clipping.

    Returns:
        (optimizer, schedule) tuple
    """
    schedule = cosine_warmup(learning_rate, warmup_steps, total_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay, b1=0.9, b2=0.999),
    )
    return optimizer, schedule
