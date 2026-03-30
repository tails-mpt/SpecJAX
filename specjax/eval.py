#!/usr/bin/env python3
"""
EAGLE3 Draft Model Evaluation — pure JAX on TPU.

Computes teacher-forcing acc_0 (single-step acceptance rate) on a neutral
dataset (MT-Bench) for any EAGLE-3 checkpoint against any supported target
model. No SGLang needed.

Reports both SpecForge-style metric (/total_positions) and per-valid-position
metric (/n_valid) for comparability.

Usage:
  pip install -e .
  python -m specjax.eval \\
    --target-model-path /path/to/target-model \\
    --target-model-type glm_flash \\
    --draft-checkpoint  /path/to/draft-checkpoint \\
    --eval-data         data/mt_bench_questions.jsonl \\
    --max-length 1024
"""

from specjax.env import configure_tpu_env
configure_tpu_env()  # must run before JAX import

import argparse
import json
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from specjax.models.draft.eagle3 import (
    Eagle3Config,
    eagle3_forward,
    load_eagle3_params,
    _compute_target_p,
)
from specjax.models.sharding import make_mesh
from specjax.models.target import get_target
from specjax.training.vocab import setup_vocab_mappings, build_t2d_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mt_bench(path: str, tokenizer, max_length: int) -> list[dict]:
    """
    Load MT-Bench questions (first turn only) and tokenize.
    NOTE: these are prompt-only — response_mask = attention_mask (all real positions).
    Returns list of dicts with 'input_ids', 'attention_mask', 'response_mask'.
    """
    samples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            turns = item.get("turns", [])
            if not turns:
                continue
            prompt = turns[0]

            # Format as a chat message for the GLM tokenizer
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)

            if len(ids) > max_length:
                ids = ids[:max_length]

            attention_mask = [1] * len(ids)
            samples.append({
                "input_ids": np.array(ids, dtype=np.int32),
                "attention_mask": np.array(attention_mask, dtype=np.int32),
                "response_mask": np.array(attention_mask, dtype=np.int32),  # all positions
                "length": len(ids),
                "question_id": item.get("question_id", "?"),
            })

    print(f"Loaded {len(samples)} MT-Bench prompts "
          f"(avg {np.mean([s['length'] for s in samples]):.0f} tokens)", flush=True)
    return samples


def load_pregenerated(path: str, max_length: int) -> list[dict]:
    """
    Load pre-generated responses saved by generate_mt_bench_responses.py.
    Format: JSONL with fields input_ids, attention_mask, response_mask, prompt_len, response_len.
    """
    samples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            ids = np.array(item["input_ids"], dtype=np.int32)
            mask = np.array(item["attention_mask"], dtype=np.int32)
            resp = np.array(item["response_mask"], dtype=np.int32)
            if len(ids) > max_length:
                ids = ids[:max_length]
                mask = mask[:max_length]
                resp = resp[:max_length]
            samples.append({
                "input_ids": ids,
                "attention_mask": mask,
                "response_mask": resp,
                "length": len(ids),
                "prompt_len": item.get("prompt_len", 0),
                "response_len": item.get("response_len", int(resp.sum())),
            })
    resp_lens = [s["response_len"] for s in samples]
    print(f"Loaded {len(samples)} pre-generated samples "
          f"(avg response {np.mean(resp_lens):.0f} tokens)", flush=True)
    return samples


def load_conversations(path: str, tokenizer, max_length: int,
                       n_samples: int = 200, skip: int = 0) -> list[dict]:
    """
    Load full conversations (user + assistant) and tokenize.
    response_mask = 1 only for assistant response tokens, 0 for prompt/special tokens.
    This correctly measures speculative decoding acceptance on generated text.

    Expects JSONL with 'messages' or 'conversations' field (ShareGPT/UltraChat format).
    """
    samples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            if skipped < skip:
                skipped += 1
                continue
            if len(samples) >= n_samples:
                break
            try:
                item = json.loads(line)
                messages = item.get("messages") or item.get("conversations") or []
                if not messages:
                    continue
                # Normalize role names (ShareGPT uses 'human'/'gpt', UltraChat uses 'user'/'assistant')
                normalized = []
                for m in messages:
                    role = m.get("role") or m.get("from", "")
                    content = m.get("content") or m.get("value", "")
                    role = {"human": "user", "gpt": "assistant"}.get(role, role)
                    if role in ("user", "assistant") and content:
                        normalized.append({"role": role, "content": content})
                if len(normalized) < 2:
                    continue

                # Tokenize prompt only (up to and including last user turn + assistant tag)
                # Find the last user turn before an assistant turn
                prompt_msgs = []
                for m in normalized:
                    prompt_msgs.append(m)
                    if m["role"] == "assistant":
                        break
                if not prompt_msgs or prompt_msgs[-1]["role"] != "assistant":
                    continue

                # Tokenize full conversation
                full_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=False,
                )
                full_ids = tokenizer.encode(full_text, add_special_tokens=False)

                # Tokenize prompt only to find where response starts
                prompt_only = prompt_msgs[:-1]  # all turns except the last assistant turn
                prompt_text = tokenizer.apply_chat_template(
                    prompt_only, tokenize=False, add_generation_prompt=True,
                )
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                prompt_len = len(prompt_ids)

                if len(full_ids) > max_length:
                    full_ids = full_ids[:max_length]

                L = len(full_ids)
                if L <= prompt_len:
                    continue  # empty response, or truncation eliminated all response tokens
                attention_mask = [1] * L
                # response_mask: 1 only for response tokens (positions prompt_len..L-1)
                response_mask = [0] * prompt_len + [1] * (L - prompt_len)

                samples.append({
                    "input_ids": np.array(full_ids, dtype=np.int32),
                    "attention_mask": np.array(attention_mask, dtype=np.int32),
                    "response_mask": np.array(response_mask, dtype=np.int32),
                    "length": L,
                    "prompt_len": prompt_len,
                    "response_len": L - prompt_len,
                })
            except Exception:
                continue

    resp_lens = [s["response_len"] for s in samples]
    print(f"Loaded {len(samples)} conversations "
          f"(avg total {np.mean([s['length'] for s in samples]):.0f} tokens, "
          f"avg response {np.mean(resp_lens):.0f} tokens)", flush=True)
    return samples


def pad_batch(samples: list[dict], max_length: int) -> dict:
    """Pad a list of samples to the same length and stack into a batch."""
    batch_ids = []
    batch_mask = []
    batch_resp = []
    for s in samples:
        ids = s["input_ids"]
        mask = s["attention_mask"]
        resp = s["response_mask"]
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids = np.concatenate([ids, np.zeros(pad_len, dtype=np.int32)])
            mask = np.concatenate([mask, np.zeros(pad_len, dtype=np.int32)])
            resp = np.concatenate([resp, np.zeros(pad_len, dtype=np.int32)])
        batch_ids.append(ids)
        batch_mask.append(mask)
        batch_resp.append(resp)
    return {
        "input_ids": np.stack(batch_ids),
        "attention_mask": np.stack(batch_mask),
        "response_mask": np.stack(batch_resp),
    }


# ---------------------------------------------------------------------------
# Eval step (jit-compiled, no gradients)
# ---------------------------------------------------------------------------

def make_eval_step(target_forward_fn, target_cfg, e3_cfg: Eagle3Config,
                   aux_layer_indices=None):
    """
    Single-step eval: forward through target + draft, compute acc_0.
    response_mask: [B, T] int32 — 1 for positions whose PREDICTED next token should
    be counted (use attention_mask for all positions, or a response-only mask).
    Returns: (loss, acc_specforge, acc_per_valid, n_valid, total_positions)
    """

    @jax.jit
    def eval_step(draft_params, target_params, d2t, t2d, input_ids, attention_mask,
                  response_mask):
        # Target model forward (frozen) — always uses full attention_mask
        if aux_layer_indices is not None:
            _last_hidden, embed_w, aux_hidden, target_logits = target_forward_fn(
                input_ids, attention_mask, target_params, target_cfg,
                aux_layer_indices=aux_layer_indices,
            )
        else:
            _last_hidden, embed_w, aux_hidden, target_logits = target_forward_fn(
                input_ids, attention_mask, target_params, target_cfg,
            )

        # Draft model forward
        hidden_states = aux_hidden @ draft_params["fc.weight"].T
        logits, _out_hidden, _, _ = eagle3_forward(
            draft_params, hidden_states, input_ids, embed_w, e3_cfg,
        )

        # Compute target distribution and position mask (offset=1, next-token)
        # Use response_mask so we only count positions predicting response tokens
        target_p, position_mask, n_valid = _compute_target_p(
            target_logits, d2t, t2d, response_mask, offset=1,
        )
        shift_draft = logits[:, :-1, :]

        # KL loss (over response positions only)
        draft_log_p = jax.nn.log_softmax(shift_draft.astype(jnp.float32), axis=-1)
        per_position_loss = -jnp.sum(target_p * draft_log_p, axis=-1)
        total_positions = jnp.float32(shift_draft.shape[0] * shift_draft.shape[1])
        loss = jnp.sum(jnp.where(position_mask, per_position_loss, 0.0)) / total_positions

        # Accuracy: draft_argmax == target_argmax
        draft_top1 = jnp.argmax(shift_draft, axis=-1)
        target_top1 = jnp.argmax(target_p, axis=-1)
        correct = (draft_top1 == target_top1).astype(jnp.float32)
        n_correct = jnp.sum(jnp.where(position_mask, correct, 0.0))

        # SpecForge metric: correct / total_positions
        acc_specforge = n_correct / total_positions
        # Per-valid metric: correct / n_valid
        acc_per_valid = n_correct / n_valid

        return loss, acc_specforge, acc_per_valid, n_valid, total_positions

    return eval_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    mesh = make_mesh(tp=args.tp)

    # Tokenizer
    print(f"Loading tokenizer from {args.target_model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=True,
    )

    # Target model (frozen)
    target_type = args.target_model_type
    aux_layer_indices = None
    if args.aux_layers:
        aux_layer_indices = set(int(x) for x in args.aux_layers.split(","))

    load_target_params, target_forward = get_target(target_type)
    print(f"Loading {target_type} target model ...", flush=True)

    target_params, target_cfg = load_target_params(args.target_model_path, mesh=mesh)
    print(f"Target loaded: {len(target_params)} tensors (type={target_type})", flush=True)

    # Draft model
    print(f"Loading draft model from {args.draft_checkpoint} ...", flush=True)
    draft_params, buffers, e3_cfg = load_eagle3_params(args.draft_checkpoint)

    # Vocab mappings
    d2t, t2d = setup_vocab_mappings(buffers, e3_cfg)

    # Eval data
    print(f"Loading eval data from {args.eval_data} ...", flush=True)
    if args.pregenerated:
        samples = load_pregenerated(args.eval_data, args.max_length)
        eval_mode = "pre-generated responses (response_mask from file)"
    elif args.response_only:
        samples = load_conversations(
            args.eval_data, tokenizer, args.max_length,
            n_samples=args.n_samples, skip=args.skip,
        )
        eval_mode = "response-only (assistant tokens)"
    else:
        samples = load_mt_bench(args.eval_data, tokenizer, args.max_length)
        eval_mode = "prompt-only (WARNING: measures prompt tokens, not generation)"
    if not samples:
        print("ERROR: No samples loaded", flush=True)
        return

    # Compile eval step
    eval_step = make_eval_step(target_forward, target_cfg, e3_cfg, aux_layer_indices)

    # Run evaluation (batch_size=1 for simplicity, variable sequence lengths)
    total_loss = 0.0
    total_correct = 0.0
    total_n_valid = 0.0
    total_positions = 0.0
    n_samples = 0

    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating: {args.draft_checkpoint}", flush=True)
    print(f"Dataset:    {args.eval_data} ({len(samples)} samples)", flush=True)
    print(f"Mode:       {eval_mode}", flush=True)
    print(f"Max length: {args.max_length}", flush=True)
    aux_str = str(sorted(aux_layer_indices)) if aux_layer_indices else "(default)"
    print(f"Aux layers: {aux_str}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t_start = time.perf_counter()

    for i, sample in enumerate(samples):
        # Pad to max_length for consistent XLA compilation
        batch = pad_batch([sample], args.max_length)
        input_ids = jnp.array(batch["input_ids"])
        attention_mask = jnp.array(batch["attention_mask"])
        response_mask = jnp.array(batch["response_mask"])

        loss, acc_sf, acc_pv, n_valid, n_total = eval_step(
            draft_params, target_params, d2t, t2d, input_ids, attention_mask,
            response_mask,
        )

        loss_val = float(loss)
        n_valid_val = float(n_valid)
        n_total_val = float(n_total)
        acc_sf_val = float(acc_sf)
        acc_pv_val = float(acc_pv)

        # Accumulate for global metrics
        # To get correct global average, we accumulate raw counts
        n_correct_sf = acc_sf_val * n_total_val
        total_correct += n_correct_sf
        total_n_valid += n_valid_val
        total_positions += n_total_val
        total_loss += loss_val * n_total_val
        n_samples += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{i+1:3d}/{len(samples)}]  "
                f"acc_0(SF)={total_correct/total_positions*100:.1f}%  "
                f"acc_0(valid)={total_correct/total_n_valid*100:.1f}%  "
                f"loss={total_loss/total_positions:.4f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    # Final results
    elapsed = time.perf_counter() - t_start
    global_acc_sf = total_correct / total_positions * 100
    global_acc_pv = total_correct / total_n_valid * 100
    global_loss = total_loss / total_positions
    valid_ratio = total_n_valid / total_positions * 100

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS: {args.draft_checkpoint}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Samples evaluated:       {n_samples}", flush=True)
    print(f"  Total positions:         {int(total_positions)}", flush=True)
    print(f"  Valid positions:         {int(total_n_valid)} ({valid_ratio:.1f}%)", flush=True)
    print(f"  Loss (SpecForge-style):  {global_loss:.4f}", flush=True)
    print(f"  acc_0 (SpecForge-style): {global_acc_sf:.2f}%  [correct / total_positions]", flush=True)
    print(f"  acc_0 (per-valid):       {global_acc_pv:.2f}%  [correct / n_valid]", flush=True)
    print(f"  Wall time:               {elapsed:.1f}s ({elapsed/n_samples:.1f}s/sample)", flush=True)
    print(f"{'='*60}\n", flush=True)


def main():
    p = argparse.ArgumentParser(description="EAGLE3 JAX evaluation on TPU")
    p.add_argument("--target-model-path", required=True,
                   help="Path to target model (GLM-4.7-Flash or Qwen3-Coder-Next)")
    p.add_argument("--target-model-type", type=str, default="glm_flash",
                   choices=["glm_flash", "glm5_fp8", "glm47_fp8", "qwen3", "qwen25"],
                   help="Target model type")
    p.add_argument("--aux-layers", type=str, default=None,
                   help="Comma-separated 0-indexed aux layer indices")
    p.add_argument("--draft-checkpoint", required=True,
                   help="Path to EAGLE-3 draft model checkpoint")
    p.add_argument("--eval-data", required=True,
                   help="Path to eval JSONL dataset")
    p.add_argument("--max-length", type=int, default=1024,
                   help="Maximum sequence length for padding")
    p.add_argument("--pregenerated", action="store_true",
                   help="Load pre-generated responses (from generate_mt_bench_responses.py). "
                        "JSONL with input_ids/attention_mask/response_mask already set.")
    p.add_argument("--response-only", action="store_true",
                   help="Load full conversations and evaluate only on assistant response "
                        "tokens (correct metric for speculative decoding acceptance rate)")
    p.add_argument("--n-samples", type=int, default=200,
                   help="Number of samples to evaluate (--response-only mode)")
    p.add_argument("--skip", type=int, default=50000,
                   help="Skip first N lines before sampling (default: 50000, "
                        "avoids overlap with training data start)")
    p.add_argument("--tp", type=int, default=4,
                   help="Tensor-parallel size (default 4). Use 8/16 for large models.")
    args = p.parse_args()

    print(f"=== EAGLE3 JAX Eval (process {jax.process_index()}) ===", flush=True)
    print(f"  jax.device_count(): {jax.device_count()} "
          f"(local: {jax.local_device_count()})", flush=True)
    print(f"  target model:  {args.target_model_path}", flush=True)
    print(f"  draft ckpt:    {args.draft_checkpoint}", flush=True)
    print(f"  eval data:     {args.eval_data}", flush=True)
    print(f"  max length:    {args.max_length}", flush=True)

    evaluate(args)


if __name__ == "__main__":
    main()
