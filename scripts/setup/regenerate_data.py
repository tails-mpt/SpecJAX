#!/usr/bin/env python3
"""
Regenerate training data using the target model via JAX-native inference on TPU.

Reads conversations from the base dataset, extracts user prompts, generates
responses using the Qwen2.5 target model already loaded in JAX, and writes
a new dataset aligned with the target model's output distribution.

This improves EAGLE3 draft head acceptance rates by aligning training data
with the target model.

Usage:
    source /path/to/venv/bin/activate
    python scripts/setup/regenerate_data.py \
        --model-path /path/to/models/Qwen2.5-7B-Instruct \
        --input-file training/data/mixed_54k.jsonl \
        --output-file training/data/mixed_54k_regen_qwen25_7b.jsonl \
        --temperature 0.8 --max-new-tokens 512 --max-samples 54000
"""

from specjax.env import configure_tpu_env
configure_tpu_env()

import argparse
import json
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from specjax.models.target.qwen25 import load_params, qwen25_forward, Qwen25Config
from specjax.models.sharding import make_mesh

logger = logging.getLogger(__name__)


def extract_prompts(input_path: str, tokenizer, max_samples: int = 0) -> list[dict]:
    """Extract first user turn from each conversation and tokenize."""
    prompts = []
    with open(input_path) as f:
        for line in f:
            if max_samples and len(prompts) >= max_samples:
                break
            item = json.loads(line)
            messages = item.get("messages") or item.get("conversations") or []
            if not messages:
                continue

            # Normalize role names
            user_turns = []
            for m in messages:
                role = m.get("role") or m.get("from", "")
                content = m.get("content") or m.get("value", "")
                role = {"human": "user", "gpt": "assistant"}.get(role, role)
                if role == "user" and content:
                    user_turns.append({"role": "user", "content": content})
                    break

            if not user_turns:
                continue

            try:
                prompt_text = tokenizer.apply_chat_template(
                    user_turns, tokenize=False, add_generation_prompt=True,
                )
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                prompts.append({
                    "messages": user_turns,
                    "prompt_ids": prompt_ids,
                })
            except Exception:
                continue

    return prompts


def generate_response(
    prompt_ids: list[int],
    params: dict,
    cfg: Qwen25Config,
    temperature: float,
    max_new_tokens: int,
    eos_token_id: int,
    rng_key: jax.Array,
) -> list[int]:
    """
    Autoregressive generation using the full forward pass (no KV cache).

    This is intentionally simple — no KV cache, just re-run the forward pass
    for each new token. Slow but correct and avoids any new code complexity.
    For 54K samples with avg 200 tokens each, this will be slow (~hours).
    We compensate by capping max_new_tokens reasonably.
    """
    # Start with prompt tokens
    ids = list(prompt_ids)
    max_total = min(len(ids) + max_new_tokens, 2048)  # Hard cap at 2048

    for step in range(max_new_tokens):
        if len(ids) >= max_total:
            break

        # Prepare input (batch size 1)
        input_ids = jnp.array([ids], dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids)

        # Forward pass — we only need logits
        hidden, _, _, logits = qwen25_forward(input_ids, attention_mask, params, cfg)

        # Get logits for last position
        next_logits = logits[0, -1, :]  # [vocab_size]

        if temperature <= 0 or temperature < 1e-6:
            # Greedy
            next_token = int(jnp.argmax(next_logits))
        else:
            # Temperature sampling
            scaled = next_logits.astype(jnp.float32) / temperature
            rng_key, subkey = jax.random.split(rng_key)
            next_token = int(jax.random.categorical(subkey, scaled))

        ids.append(next_token)

        if next_token == eos_token_id:
            break

    return ids[len(prompt_ids):]  # Return only generated tokens


def main():
    p = argparse.ArgumentParser(description="Regenerate training data via JAX inference on TPU")
    p.add_argument("--model-path", required=True,
                   help="Path to target model (e.g., Qwen2.5-7B-Instruct)")
    p.add_argument("--input-file", required=True,
                   help="Input JSONL dataset")
    p.add_argument("--output-file", required=True,
                   help="Output JSONL with regenerated responses")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max new tokens per response (default: 512)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max samples to process (0=all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print(f"=== Data Regeneration (JAX-native on TPU) ===", flush=True)
    print(f"  Model: {args.model_path}", flush=True)
    print(f"  Input: {args.input_file}", flush=True)
    print(f"  Output: {args.output_file}", flush=True)
    print(f"  Temp: {args.temperature}, max_new_tokens: {args.max_new_tokens}", flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id or 151643  # Qwen2.5 EOS

    # Load prompts
    print(f"Loading prompts ...", flush=True)
    prompts = extract_prompts(args.input_file, tokenizer, args.max_samples)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    # Load model
    mesh = make_mesh()
    print(f"Loading model ...", flush=True)
    params, cfg = load_params(args.model_path, mesh=mesh)
    print(f"Model loaded ({len(params)} tensors)", flush=True)

    # Generate
    rng_key = jax.random.PRNGKey(args.seed)
    t_start = time.perf_counter()
    completed = 0

    with open(args.output_file, "w") as out_f:
        for i, prompt_data in enumerate(prompts):
            rng_key, gen_key = jax.random.split(rng_key)

            try:
                response_ids = generate_response(
                    prompt_data["prompt_ids"], params, cfg,
                    args.temperature, args.max_new_tokens, eos_token_id, gen_key,
                )
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

                if response_text.strip():
                    result = {
                        "messages": prompt_data["messages"] + [
                            {"role": "assistant", "content": response_text}
                        ]
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    completed += 1
            except Exception as e:
                logger.warning(f"Sample {i} failed: {e}")

            if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / elapsed
                eta = (len(prompts) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i+1:5d}/{len(prompts)}]  "
                    f"ok={completed}  "
                    f"rate={rate:.2f} samples/s  "
                    f"elapsed={elapsed/60:.1f}m  "
                    f"ETA={eta/60:.1f}m",
                    flush=True,
                )

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. {completed} samples written to {args.output_file}", flush=True)
    print(f"Wall time: {elapsed/60:.1f}m ({elapsed/3600:.1f}h)", flush=True)


if __name__ == "__main__":
    main()
