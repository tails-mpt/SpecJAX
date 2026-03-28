#!/usr/bin/env python3
"""
EAGLE3 Draft Model Training — pure JAX/Optax on TPU.

Trains an EAGLE3 draft model against any supported frozen target model
using KL divergence loss. Produces safetensors checkpoints compatible
with inference servers (SpecForge / SGLang).

Usage:
  pip install -e .
  python -m specjax.train \\
    --target-model-path /path/to/target-model \\
    --target-model-type glm_flash \\
    --data-path         data/sharegpt.jsonl \\
    --output-dir        /path/to/checkpoints \\
    --exp-name          my-experiment
"""

from specjax.env import configure_tpu_env
configure_tpu_env()  # must run before JAX import

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer

from specjax.models.target import get_target
from specjax.models.draft.eagle3 import (
    Eagle3Config,
    eagle3_forward,
    compute_loss,
    compute_ttt_loss,
    load_eagle3_params,
    save_eagle3_checkpoint,
    validate_eagle3_checkpoint,
)
from specjax.models.sharding import make_mesh, batch_sharding as make_batch_sharding
from specjax.data.dataset import Eagle3Dataset, BucketBatchSampler, DataLoader
from specjax.training.vocab import build_d2t_from_data, build_t2d_map
from specjax.training.optimizer import build_optimizer

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train step (jit-compiled micro-batch forward+backward)
# ---------------------------------------------------------------------------

def make_train_step(target_forward_fn, target_cfg, e3_cfg: Eagle3Config):
    """
    Single-step train step with KL divergence loss (matching SpecForge).
    Returns: (loss, acc, grads)
    """

    @jax.jit
    def train_step(params, target_params, d2t, t2d, batch):
        input_ids      = jnp.array(batch["input_ids"])
        attention_mask = jnp.array(batch["attention_mask"])

        def loss_fn(p):
            _last_hidden, embed_w, aux_hidden, target_logits = target_forward_fn(
                input_ids, attention_mask, target_params, target_cfg,
            )
            # FC projects multi-layer features [B, T, 3H] -> [B, T, H]
            hidden_states = aux_hidden @ p["fc.weight"].T
            logits, _out_hidden, _, _ = eagle3_forward(
                p, hidden_states, input_ids, embed_w, e3_cfg,
            )
            loss, acc = compute_loss(logits, target_logits, d2t, t2d, attention_mask)
            return loss, acc

        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, acc, grads

    return train_step


def make_train_step_ttt(target_forward_fn, target_cfg, e3_cfg: Eagle3Config, ttt_length: int = 7):
    """
    Multi-step TTT train step with KL divergence loss (matching SpecForge).

    Returns: (total_loss, plosses[ttt_length], acces[ttt_length], grads)

    Gradient accumulation is handled in Python (same as single-step version).
    The entire TTT loop is traced once by JAX jit and compiled as one XLA program.
    """

    @jax.jit
    def train_step(params, target_params, d2t, t2d, batch):
        input_ids      = jnp.array(batch["input_ids"])
        attention_mask = jnp.array(batch["attention_mask"])

        def loss_fn(p):
            _last_hidden, embed_w, aux_hidden, target_logits = target_forward_fn(
                input_ids, attention_mask, target_params, target_cfg,
            )
            total_loss, plosses, acces = compute_ttt_loss(
                p, aux_hidden, embed_w, input_ids, attention_mask,
                target_logits, d2t, t2d, e3_cfg, ttt_length,
            )
            return total_loss, (plosses, acces)

        (total_loss, (plosses, acces)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        return total_loss, plosses, acces, grads

    return train_step


# ---------------------------------------------------------------------------
# Eagle3Config from target model
# ---------------------------------------------------------------------------

def _eagle3_config_from_target(target_cfg, target_type: str) -> Eagle3Config | None:
    """Build an Eagle3Config matching the target model's dimensions.

    Returns None if the target type uses the default Eagle3Config (GLM models).
    """
    if target_type in ("qwen2", "qwen25", "qwen3"):
        return Eagle3Config(
            hidden_size=target_cfg.hidden_size,
            intermediate_size=target_cfg.intermediate_size,
            num_heads=target_cfg.num_attention_heads,
            num_kv_heads=target_cfg.num_key_value_heads,
            head_dim=target_cfg.head_dim,
            vocab_size=target_cfg.vocab_size,
            draft_vocab_size=32_000,
            rope_theta=target_cfg.rope_theta,
            rms_norm_eps=target_cfg.rms_norm_eps,
        )
    if target_type == "llama":
        return Eagle3Config(
            hidden_size=target_cfg.hidden_size,
            intermediate_size=target_cfg.intermediate_size,
            num_heads=target_cfg.num_attention_heads,
            num_kv_heads=target_cfg.num_key_value_heads,
            head_dim=target_cfg.head_dim,
            vocab_size=target_cfg.vocab_size,
            draft_vocab_size=32_000,
            rope_theta=target_cfg.rope_theta,
            rms_norm_eps=target_cfg.rms_norm_eps,
        )
    if target_type == "minimax_m2":
        return Eagle3Config(
            hidden_size=target_cfg.hidden_size,
            intermediate_size=target_cfg.intermediate_size,
            num_heads=target_cfg.num_attention_heads,
            num_kv_heads=target_cfg.num_key_value_heads,
            head_dim=target_cfg.head_dim,
            vocab_size=target_cfg.vocab_size,
            draft_vocab_size=32_000,
            rope_theta=target_cfg.rope_theta,
            rms_norm_eps=target_cfg.rms_norm_eps,
        )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ttt_mode = args.ttt_length > 1

    # Multi-host / DP setup
    mesh = make_mesh(tp=args.tp)
    dp_size  = mesh.shape["dp"] if "dp" in mesh.axis_names else 1
    # DP rank: when TP spans multiple hosts (tp > local_device_count),
    # multiple processes share the same DP rank. Compute from process count.
    processes_per_dp_rank = max(1, jax.process_count() // max(dp_size, 1))
    dp_index = jax.process_index() // processes_per_dp_rank
    is_primary = jax.process_index() == 0  # only process 0 logs/saves

    _batch_sharding = make_batch_sharding(mesh)
    logger.warning(
        f"Process {dp_index}/{dp_size}  |  "
        f"local_devices={jax.local_device_count()}  total_devices={jax.device_count()}"
    )

    use_wandb = _WANDB_AVAILABLE and args.wandb_project
    if use_wandb and is_primary:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or args.exp_name,
                config={
                    "target_model": args.target_model_path,
                    "target_model_type": args.target_model_type,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "effective_batch": args.batch_size * args.grad_accum_steps * dp_size,
                    "dp_size": dp_size,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "max_length": args.max_length,
                    "warmup_ratio": args.warmup_ratio,
                    "ttt_length": args.ttt_length,
                    "hardware": f"tpu-{jax.device_count()}chips-jax-dp{dp_size}tp{jax.device_count()//dp_size}",
                },
            )
        except Exception as e:
            logger.warning(f"W&B init failed ({e}) — continuing without W&B")
            use_wandb = False

    # Tokenizer
    logger.warning(f"Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)

    # Target model (frozen, SPMD sharded)
    target_type = args.target_model_type
    load_target_params, target_forward_fn = get_target(target_type)
    logger.warning(f"Loading {target_type} target model ...")
    target_params, target_cfg = load_target_params(args.target_model_path, mesh=mesh)
    logger.warning(f"Target loaded: {len(target_params)} tensors (type={target_type})")

    # Draft model — build Eagle3Config from target model when training from scratch
    e3_config_override = None
    if args.draft_init_path is None:
        e3_config_override = _eagle3_config_from_target(target_cfg, target_type)
        if e3_config_override is not None:
            logger.warning(
                f"Eagle3 config from target: hidden={e3_config_override.hidden_size} "
                f"vocab={e3_config_override.vocab_size}"
            )

    logger.warning("Loading Eagle3 draft model ...")
    draft_params, buffers, e3_cfg = load_eagle3_params(
        args.draft_init_path,
        key=jax.random.PRNGKey(0),
        config_override=e3_config_override,
    )

    source_config_path = (
        os.path.join(args.draft_init_path, "config.json")
        if args.draft_init_path else None
    )

    # Vocabulary mappings (constant, not differentiated)
    if args.draft_init_path is None:
        # From scratch: build d2t (actual target indices) from training data frequencies
        logger.warning("Building Eagle3 vocab mapping from training data ...")
        d2t_actual_np = build_d2t_from_data(
            args.data_path, tokenizer,
            draft_vocab_size=e3_cfg.draft_vocab_size,
            max_length=args.max_length,
        )
        # Build t2d bool mask from d2t
        t2d_bool_np = np.zeros(e3_cfg.vocab_size, dtype=bool)
        t2d_bool_np[d2t_actual_np] = True
        # Store in buffers for checkpoint saving
        buffers["d2t"] = jnp.array(d2t_actual_np, dtype=jnp.int64)
        buffers["t2d"] = jnp.array(t2d_bool_np)
        d2t = buffers["d2t"]
        logger.warning(
            f"d2t: built from data frequencies. "
            f"Range [{d2t_actual_np.min()}, {d2t_actual_np.max()}]"
        )
    else:
        # From checkpoint: SpecForge stores d2t as OFFSETS, not actual indices.
        # Reconstruct actual d2t from the t2d bool mask (authoritative source).
        t2d_bool_np = np.array(buffers["t2d"], dtype=bool)
        d2t_actual_np = np.where(t2d_bool_np)[0].astype(np.int64)
        assert len(d2t_actual_np) == e3_cfg.draft_vocab_size, (
            f"t2d bool mask has {len(d2t_actual_np)} True entries, "
            f"expected draft_vocab_size={e3_cfg.draft_vocab_size}"
        )
        d2t = jnp.array(d2t_actual_np, dtype=jnp.int64)
        logger.warning(
            f"d2t: reconstructed from t2d bool mask. "
            f"Range [{d2t_actual_np.min()}, {d2t_actual_np.max()}]"
        )

    t2d = build_t2d_map(d2t, e3_cfg.vocab_size, e3_cfg.draft_vocab_size)
    logger.warning(
        f"Draft vocab coverage: {float((t2d >= 0).mean())*100:.1f}% of target vocab"
    )

    # Dataset
    dataset = Eagle3Dataset(args.data_path, tokenizer, max_length=args.max_length)

    def make_dataloader(seed: int = 0):
        random.seed(seed)
        sampler = BucketBatchSampler(
            dataset.bucket_lengths, batch_size=args.batch_size, drop_last=True
        )
        return DataLoader(dataset, sampler)

    # steps_per_epoch = batches each DP rank processes (1/dp_size of total)
    total_batches_per_epoch = len(make_dataloader(seed=0))
    steps_per_epoch  = total_batches_per_epoch // dp_size   # batches per rank per epoch
    total_opt_steps  = (steps_per_epoch // args.grad_accum_steps) * args.num_epochs
    warmup_steps     = max(1, int(total_opt_steps * args.warmup_ratio))
    logger.warning(
        f"Process {dp_index}: {len(dataset)} samples, "
        f"{total_batches_per_epoch} total micro-batches/epoch → "
        f"{steps_per_epoch} per DP rank, "
        f"{total_opt_steps} optimizer steps, {warmup_steps} warmup steps"
    )

    # Promote norm/layernorm weights to float32 to avoid bfloat16 precision loss.
    # Norm weights are initialised to 1.0; small gradient updates (~1e-5) vanish
    # in bfloat16 due to rounding (1.0 + 1e-5 → 1.0).  Keeping them in float32
    # lets the optimizer accumulate updates accurately.  They are cast back to
    # bfloat16 in save_eagle3_checkpoint for checkpoint compatibility.
    for k in list(draft_params.keys()):
        if "layernorm" in k or k == "norm.weight" or "hidden_norm" in k:
            draft_params[k] = draft_params[k].astype(jnp.float32)
            logger.warning(f"  Promoted {k} to float32 (was bfloat16)")

    # Optimizer
    optimizer, schedule = build_optimizer(
        args.learning_rate, warmup_steps, total_opt_steps,
    )
    opt_state = optimizer.init(draft_params)

    # Compile train step
    if ttt_mode:
        train_step = make_train_step_ttt(target_forward_fn, target_cfg, e3_cfg, args.ttt_length)
        logger.warning(f"Using TTT mode: ttt_length={args.ttt_length}")
    else:
        train_step = make_train_step(target_forward_fn, target_cfg, e3_cfg)
        logger.warning("Using single-step mode")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step  = 0
    micro_step   = 0
    acc_grads: Optional[dict] = None
    acc_loss  = 0.0
    acc_cov   = 0.0   # accuracy (single-step) or avg acc across TTT positions
    # Per-TTT-position accumulators (ttt_mode only)
    acc_plosses: list = [0.0] * args.ttt_length
    acc_acces:   list = [0.0] * args.ttt_length
    n_log_steps = 0
    t_start   = time.perf_counter()

    for epoch in range(args.num_epochs):
        logger.warning(f"=== Epoch {epoch + 1}/{args.num_epochs} (process {dp_index}/{dp_size}) ===")

        # All DP ranks use the same epoch seed so they see the same shuffled order,
        # then each rank strides through a disjoint subset.
        #
        # CRITICAL for multi-host: group batches by bucket length before striding.
        # The bucket batch sampler produces batches of varying sequence lengths
        # (128, 256, 512, 1024). If workers stride through a randomly-shuffled
        # list, they may get different bucket lengths on the same step, causing
        # XLA program mismatches ("unexpected peer with different launch id").
        # Sorting by length ensures that strided workers always process the
        # same bucket length on each step.
        epoch_seed = epoch * 1000
        all_batches = list(make_dataloader(seed=epoch_seed))
        # Sort by sequence length so all DP ranks process matching shapes.
        all_batches.sort(key=lambda b: b["input_ids"].shape[1])
        # Align each bucket boundary to dp_size. Without this, if K_bucket mod dp_size != 0,
        # strided workers cross the bucket boundary at different iteration indices, causing
        # shape mismatches across DP ranks ("unexpected peer with different launch id").
        aligned: list = []
        i = 0
        while i < len(all_batches):
            cur_len = all_batches[i]["input_ids"].shape[1]
            j = i
            while j < len(all_batches) and all_batches[j]["input_ids"].shape[1] == cur_len:
                j += 1
            aligned.extend(all_batches[i : i + ((j - i) // dp_size) * dp_size])
            i = j
        all_batches = aligned
        n_rank_batches = len(all_batches) // dp_size   # ensure equal counts across ranks
        my_batches = [all_batches[dp_index + i * dp_size] for i in range(n_rank_batches)]

        for batch in my_batches:
            # Shard batch across DP ranks so XLA inserts all-reduce on gradients.
            # With dp_size==1 this is a no-op (the sharding replicates across the
            # single TP mesh and jnp.array() already handles numpy input).
            if dp_size > 1:
                batch = {
                    k: jax.make_array_from_process_local_data(_batch_sharding, v)
                    for k, v in batch.items()
                }
            if ttt_mode:
                loss, plosses, acces, grads = train_step(draft_params, target_params, d2t, t2d, batch)
            else:
                loss, acc, grads = train_step(draft_params, target_params, d2t, t2d, batch)

            if acc_grads is None:
                acc_grads = grads
            else:
                acc_grads = jax.tree.map(lambda a, b: a + b, acc_grads, grads)

            acc_loss += float(loss) / args.grad_accum_steps
            if ttt_mode:
                for k in range(args.ttt_length):
                    acc_plosses[k] += float(plosses[k]) / args.grad_accum_steps
                    acc_acces[k]   += float(acces[k])   / args.grad_accum_steps
                acc_cov += float(sum(acces) / len(acces)) / args.grad_accum_steps
            else:
                acc_cov += float(acc) / args.grad_accum_steps
            micro_step += 1

            # Optimizer step
            if micro_step % args.grad_accum_steps == 0:
                scaled_grads = jax.tree.map(lambda g: g / args.grad_accum_steps, acc_grads)
                updates, opt_state = optimizer.update(scaled_grads, opt_state, draft_params)
                draft_params = optax.apply_updates(draft_params, updates)

                global_step += 1
                n_log_steps += 1
                acc_grads = None

                if global_step % args.log_every == 0 and is_primary:
                    lr_now   = float(schedule(global_step))
                    elapsed  = time.perf_counter() - t_start
                    sps      = elapsed / global_step
                    avg_loss = acc_loss / max(n_log_steps, 1)
                    avg_cov  = acc_cov  / max(n_log_steps, 1)

                    if ttt_mode:
                        avg_plosses = [acc_plosses[k] / max(n_log_steps, 1) for k in range(args.ttt_length)]
                        avg_acces   = [acc_acces[k]   / max(n_log_steps, 1) for k in range(args.ttt_length)]
                        pos_str = "  ".join(
                            f"acc_{k}={avg_acces[k]*100:.1f}%" for k in range(min(3, args.ttt_length))
                        )
                        print(
                            f"[step {global_step:6d}] "
                            f"loss={avg_loss:.4f}  {pos_str}  "
                            f"lr={lr_now:.2e}  elapsed={elapsed:.0f}s  s/step={sps:.1f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[step {global_step:6d}] "
                            f"loss={avg_loss:.4f}  acc={avg_cov*100:.1f}%  "
                            f"lr={lr_now:.2e}  elapsed={elapsed:.0f}s  s/step={sps:.1f}",
                            flush=True,
                        )

                    if use_wandb:
                        log_dict = {
                            "train/loss": avg_loss,
                            "train/lr": lr_now,
                            "train/step_time_s": sps,
                            "train/elapsed_s": elapsed,
                        }
                        if ttt_mode:
                            for k in range(args.ttt_length):
                                log_dict[f"train/ploss_{k}"] = avg_plosses[k]
                                log_dict[f"train/acc_{k}"]   = avg_acces[k]
                        else:
                            log_dict["train/acc"] = avg_cov
                        wandb.log(log_dict, step=global_step)

                    acc_loss    = 0.0
                    acc_cov     = 0.0
                    acc_plosses = [0.0] * args.ttt_length
                    acc_acces   = [0.0] * args.ttt_length
                    n_log_steps = 0

                if args.save_every > 0 and global_step % args.save_every == 0:
                    ckpt_dir = output_dir / f"step_{global_step}"
                    save_eagle3_checkpoint(
                        draft_params, buffers, str(ckpt_dir), source_config_path, e3_cfg,
                        is_primary=is_primary,
                    )
                    if is_primary:
                        print(f"[step {global_step}] Checkpoint → {ckpt_dir}", flush=True)

            if args.max_steps and global_step >= args.max_steps:
                logger.warning(f"Reached --max-steps {args.max_steps}, stopping.")
                break

        # End-of-epoch checkpoint (all processes participate in gather, primary writes)
        ckpt_dir = output_dir / f"epoch_{epoch + 1}"
        save_eagle3_checkpoint(
            draft_params, buffers, str(ckpt_dir), source_config_path, e3_cfg,
            is_primary=is_primary,
        )
        if is_primary:
            print(f"[epoch {epoch + 1}] Checkpoint → {ckpt_dir}", flush=True)

        if args.max_steps and global_step >= args.max_steps:
            break

    # Final checkpoint (all processes participate in gather, primary writes)
    final_dir = output_dir / "final"
    save_eagle3_checkpoint(
        draft_params, buffers, str(final_dir), source_config_path, e3_cfg,
        is_primary=is_primary,
    )
    if is_primary:
        validate_eagle3_checkpoint(str(final_dir))
        print(f"Training complete. Final checkpoint → {final_dir}", flush=True)

    if use_wandb and is_primary:
        wandb.finish()


def main():
    p = argparse.ArgumentParser(description="EAGLE3 JAX training on TPU")
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON config file (CLI args override config values)")
    p.add_argument("--target-model-path", required=True)
    p.add_argument("--target-model-type", type=str, default="glm_flash",
                   choices=["glm_flash", "glm5_fp8", "glm47_fp8", "qwen3", "qwen2", "qwen25", "llama"],
                   help="Target model type (default: glm_flash)")
    p.add_argument("--draft-init-path", default=None)
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True,
                   help="Use persistent storage for preemption durability")
    p.add_argument("--exp-name", default="exp-jax")
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Stop after N optimizer steps (dry-run)")
    p.add_argument("--ttt-length", type=int, default=1,
                   help="TTT unroll steps (1=single-step baseline, 7=full SpecForge multi-step)")
    p.add_argument("--tp", type=int, default=4,
                   help="Tensor-parallel size (default 4). Use 8/16 for large models.")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)

    # Two-pass parse: load config file defaults first, then CLI overrides
    args, _ = p.parse_known_args()
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        # Apply config values as defaults (CLI args take precedence)
        config_mapped = {k.replace("-", "_"): v for k, v in config.items()}
        p.set_defaults(**config_mapped)
    args = p.parse_args()

    print(f"=== EAGLE3 JAX Training: {args.exp_name} (process {jax.process_index()}) ===", flush=True)
    print(f"  jax.device_count() : {jax.device_count()}  (local: {jax.local_device_count()})", flush=True)
    print(f"  target model : {args.target_model_path} (type={args.target_model_type})", flush=True)
    print(f"  draft init   : {args.draft_init_path or '(scratch)'}", flush=True)
    print(f"  data         : {args.data_path}", flush=True)
    print(f"  output dir   : {args.output_dir}", flush=True)
    print(f"  batch×accum  : {args.batch_size}×{args.grad_accum_steps}="
          f"{args.batch_size*args.grad_accum_steps} per-dp-rank", flush=True)
    print(f"  epochs       : {args.num_epochs}  lr={args.learning_rate}  "
          f"max_len={args.max_length}", flush=True)

    train(args)


if __name__ == "__main__":
    main()
