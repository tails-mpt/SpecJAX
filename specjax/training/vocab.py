"""Vocabulary mapping utilities shared between training and evaluation."""

import json
import logging
from collections import Counter

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def build_d2t_from_data(
    data_path: str,
    tokenizer,
    draft_vocab_size: int,
    max_length: int = 512,
    n_samples: int = 5000,
) -> np.ndarray:
    """
    Build d2t (draft_id -> target_id) by counting token frequencies in the training
    data and selecting the top-draft_vocab_size most frequent tokens.

    Called when no pre-trained Eagle3 checkpoint is available.
    Returns: int64 np.ndarray of shape [draft_vocab_size]
    """
    counter: Counter = Counter()
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            try:
                item = json.loads(line)
                convs = item.get("conversations") or item.get("messages") or []
                for turn in convs:
                    text = turn.get("value") or turn.get("content") or ""
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    counter.update(ids)
            except Exception:
                continue

    top_tokens = [tok for tok, _ in counter.most_common(draft_vocab_size)]
    if len(top_tokens) < 1000:
        logger.warning(
            f"Only {len(top_tokens)} unique tokens found in {n_samples} samples — "
            f"check that data_path is correct and contains tokenizable text"
        )
    # Pad with sequential IDs if we have fewer than draft_vocab_size unique tokens
    seen = set(top_tokens)
    extra_id = 0
    while len(top_tokens) < draft_vocab_size:
        if extra_id not in seen:
            top_tokens.append(extra_id)
            seen.add(extra_id)
        extra_id += 1

    # Sort by token ID ascending (matches SpecForge's used_tokens.sort())
    top_tokens = sorted(top_tokens[:draft_vocab_size])

    d2t = np.array(top_tokens, dtype=np.int64)
    logger.warning(
        f"Built d2t vocab mapping: top {draft_vocab_size} tokens from {n_samples} samples"
    )
    return d2t


def build_t2d_map(d2t: jnp.ndarray, vocab_size: int, draft_vocab_size: int) -> jnp.ndarray:
    """
    [vocab_size] int32 array: target_token_id -> draft_token_id (-1 for OOV).
    Mirrors Eagle3DraftModel.build_target_to_draft_map() from PyTorch version.
    """
    mapping = np.full((vocab_size,), -1, dtype=np.int32)
    d2t_np = np.array(d2t)
    for draft_idx in range(draft_vocab_size):
        tgt_idx = int(d2t_np[draft_idx])
        if 0 <= tgt_idx < vocab_size:
            mapping[tgt_idx] = draft_idx
    return jnp.array(mapping, dtype=jnp.int32)


def setup_vocab_mappings(buffers: dict, e3_cfg) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reconstruct d2t (actual target indices) and t2d (target->draft mapping)
    from checkpoint buffers.

    Returns:
        d2t: [draft_vocab_size] int64  -- draft_id -> target_id
        t2d: [vocab_size] int32        -- target_id -> draft_id (-1 for OOV)
    """
    t2d_bool_np = np.array(buffers["t2d"], dtype=bool)
    d2t_actual_np = np.where(t2d_bool_np)[0].astype(np.int64)
    assert len(d2t_actual_np) == e3_cfg.draft_vocab_size, (
        f"t2d bool mask has {len(d2t_actual_np)} True entries, "
        f"expected draft_vocab_size={e3_cfg.draft_vocab_size}"
    )
    d2t = jnp.array(d2t_actual_np, dtype=jnp.int64)
    t2d = build_t2d_map(d2t, e3_cfg.vocab_size, e3_cfg.draft_vocab_size)
    return d2t, t2d
