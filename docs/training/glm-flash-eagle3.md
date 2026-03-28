# Eagle3 JAX/TPU Training for GLM-4.7-Flash

**W&B**: [glm4-eagle3-experiments](https://wandb.ai/your-wandb-entity/glm4-eagle3-experiments)
**Hardware**: TPU v6e-4 (Trillium), `ct6e-standard-4t`, GCP SPOT VM

---

## Quick Start (after preemption)

```bash
# 1. Bootstrap venv + download models + data
bash scripts/setup_jax_env.sh

# 2. Verify TPU
source /path/to/workspace/.venv-train-jax/bin/activate
python3 -c "import jax; print(jax.devices())"
# → [TpuDevice(id=0,...), ..., TpuDevice(id=3,...)]

# 3. Pre-flight checks
bash scripts/preflight_jax.sh

# 4. Launch (run with sudo for GCS write permissions)
sudo bash -c 'source ~/.specjax.env && source /path/to/workspace/.venv-train-jax/bin/activate && \
  bash scripts/run_exp_jax_i.sh'

# Monitor
tail -f /path/to/workspace/logs/exp_jax_i_*.log
```

---

## Hardware & Paths

| Resource | Path | Persistence |
|---|---|---|
| Code repo | (repo root) | GCS (persistent) |
| Checkpoints | `/path/to/checkpoints/exp-jax-{a..i}/` | GCS (persistent) |
| Secrets | `~/.specjax.env` (WANDB_API_KEY, HF_TOKEN) | GCS (persistent) |
| Venv | `/path/to/workspace/.venv-train-jax` | NVMe (ephemeral) |
| Target model | `/path/to/models/GLM-4.7-Flash` | NVMe (ephemeral) |
| Training logs | `/path/to/workspace/logs/` | NVMe (ephemeral) |
| W&B cache | `/path/to/workspace/wandb/` | NVMe (ephemeral) |
| SpecForge ref | `/path/to/shared-storage/SpecForge-internal/` | GCS (persistent) |

**SPOT VM**: `/path/to/workspace/` is wiped on preemption. Run `setup_jax_env.sh` to recreate venv and re-download models. Code and checkpoints on `/path/to/shared-storage/` survive.

---

## Architecture

### Target Model

GLM-4.7-Flash: 9B params, 47 layers, MoE (64 routed experts), vocab 154,880, hidden 2048.

### Eagle3 Draft Head

Single-layer draft model that predicts the target's next-token distribution for speculative decoding.

**Weight structure** (15 tensors, ~278MB):

| Tensor | Shape | Purpose |
|---|---|---|
| `fc.weight` | [2048, 6144] | Projects [low \|\| mid \|\| high] target hidden states (3H → H) |
| `midlayer.input_layernorm.weight` | [2048] | RMSNorm on token embeddings |
| `midlayer.hidden_norm.weight` | [2048] | RMSNorm on FC-projected features |
| `midlayer.self_attn.{q,k,v,o}_proj.weight` | varies | GQA: 16 heads, 4 KV heads, head_dim=128 |
| `midlayer.post_attention_layernorm.weight` | [2048] | RMSNorm after attention |
| `midlayer.mlp.{gate,up,down}_proj.weight` | varies | SiLU-gated MLP (intermediate=8192) |
| `norm.weight` | [2048] | Final RMSNorm |
| `lm_head.weight` | [32000, 2048] | Draft vocab logits |
| `d2t` | [32000] int32 | Draft→target token mapping (stored as offsets) |
| `t2d` | [154880] bool | Target→draft token mask (authoritative source) |

**Config**: `rope_theta=1,000,000`, `rms_norm_eps=1e-5`

### Forward Pass

```
Target model (frozen):
  input_ids → 47 decoder layers → extract hidden states at layers {1, 22, 43}

Eagle3 draft head:
  FC:       [low_h || mid_h || high_h] (3H=6144) → projected_features (H=2048)
  Embed:    input_ids → embed_tokens (H)
  Midlayer: Q = cat(norm(embed), norm(projected)) (2H=4096)
            Standard causal attention + GQA + RoPE
            MLP → hidden_out (H)
  Norm + lm_head → draft logits [B, T, 32000]
```

### TTT (Test-Time Training) Rollout

`ttt_length=7`: 7 sequential forward passes unrolled at compile time via `jax.jit`.

- **Loss**: KL divergence vs target model's softmax distribution: `-sum(target_p * log_softmax(draft_logits))`
- **Loss weighting**: `total_loss = sum(0.8^k * loss_k for k in range(7))`
- **Loss normalization**: Divide by `B * T` (total positions), matching SpecForge's `.mean()`
- **KV cache**: Multi-branch tree attention — step 0 uses full causal attention, steps 1..k use scalar dot-product branches
- **Metric**: `acc_k = (draft_argmax == target_argmax)` — measures draft-target agreement, not ground-truth accuracy

### Draft Vocabulary

32K tokens selected by frequency from training data, then **sorted by target token ID ascending** (matching SpecForge's `used_tokens.sort()`).

- **d2t storage**: Offsets `d2t[i] = actual_target_id - i` (SpecForge convention)
- **t2d storage**: Boolean mask over full 154,880 vocab (authoritative source)
- **Loading**: Always reconstruct actual IDs from t2d: `d2t_actual = np.where(t2d_bool)[0]`
- **Coverage**: ~20.7% of target vocab → 96.4% of training tokens

### SPMD Sharding

1D tensor-parallel across 4 TPU chips: `Mesh(jax.devices(), ("tp",))`. Dense weights shard on leading dim, MoE experts shard on expert dim.

---

## GPU/TPU Interoperability

Seven incompatibilities found and fixed (2026-03-22) to ensure checkpoint interchangeability between SpecForge GPU and JAX TPU training:

| # | Issue | Was (JAX) | Fixed to (SpecForge) | File |
|---|-------|-----------|---------------------|------|
| 1 | rope_theta | 500,000 | **1,000,000** | `specjax/models/draft/eagle3.py` |
| 2 | rms_norm_eps | 1e-6 | **1e-5** | `specjax/models/draft/eagle3.py` |
| 3 | Aux layer indices | {2, 23, 44} | **{1, 22, 43}** | `specjax/models/target/glm_flash.py` |
| 4 | d2t token ordering | frequency order | **sorted by token ID** | `specjax/train.py` |
| 5 | d2t storage format | actual IDs | **offsets** | `specjax/models/draft/eagle3.py` |
| 6 | Norm weights stuck at 1.0 | bfloat16 (precision loss) | **float32 during training** | `specjax/train.py` |
| 7 | SpecForge GLM config | head_dim=102, heads=20 | **head_dim=128, heads=16** | SpecForge-internal |

**Aux layer detail**: SpecForge uses HF `outputs.hidden_states` with offset=1: `low=1+1=2, mid=N//2-1+1=23, high=N-4+1=44` → 0-indexed layers {1, 22, 43}. SafeAILab uses {2, 23, 44}. We match SpecForge.

**Attention parity**: SpecForge sdpa path, SpecForge flex_attention (diagonal mask), and JAX all produce identical multi-branch tree attention patterns at any TTT length.

**GPU checkpoint eval caveat**: The HF checkpoint scores 36.83% on our eval despite reporting 79% training accuracy. This is likely because SpecForge's `process_token_dict_to_mappings` produces a different d2t mapping than our `build_d2t_from_data` — the lm_head weights are trained against a different token ordering. Our eval reconstructs d2t from the t2d bool mask, which may not match the GPU training's internal mapping. The 79% training metric is self-consistent but not comparable to our eval pipeline.

---

## Experiment Results

| Experiment | Mode | LR | Epochs | acc_0 (train) | acc@pos0 ShareGPT | acc@pos0 MT-bench | Notes |
|---|---|---|---|---|---|---|---|
| exp-jax-a | Single-step | 1e-4 | 1 | — | — | — | Baseline, ShareGPT 10K |
| exp-jax-b | Single-step | 5e-5 | 2 | — | — | — | Loss=5.505, coverage 96.4% |
| exp-jax-c | TTT-7 | 5e-5 | 2 | 18% | — | — | acc_0 inversion (acc_1-4 ≈ 39%) |
| exp-jax-d | Single-step | 1e-4 | 3 | — | — | — | 54K mixed, matches GPU Exp K params |
| exp-jax-e | TTT-7 | 5e-5 | 3 | 30.5% | — | — | Fine-tuned from exp-jax-d |
| exp-jax-f | TTT-7 | 1e-4 | 3 | ~30% | — | — | Architecture rewrite (SpecForge-compatible) |
| exp-jax-g | TTT-7 | 1e-4 | 3 | ~56% | — | — | From scratch; pre-interop fixes |
| **exp-jax-h** | TTT-7 | 1e-4 | 3 | 51.0% | **50.49%** | **47.44%** | All 7 interop fixes; first honest number |
| **exp-jax-i** | TTT-7 | **8e-4** | 3 | 59.0% | **59.71%** | **55.95%** | LR scaling (+9.2pp over h) |
| GPU Exp K | — | 1e-4 | 3 | 79% (train) | **36.83%** | **30.89%** | HF checkpoint; root cause under investigation |

**Eval datasets**: ShareGPT = 200 samples from `training/data/sharegpt.jsonl`, max_length=512. MT-bench = 80 questions from `benchmark/data/mt_bench_sharegpt.jsonl`, max_length=512.

**Key observations**:
- **exp-jax-i consistently outperforms GPU Exp K** on both eval datasets (ShareGPT: 59.71% vs 36.83%, MT-bench: 55.95% vs 30.89%)
- GPU Exp K's 79% training acc vs 31-37% eval acc gap is under investigation — code audit found no clear bug in d2t reconstruction, config loading, forward pass, or metric computation. Possible causes: eval data distribution mismatch, or numerical differences between PyTorch and JAX target model implementations. Requires GPU access to verify.
- LR scaling from 1e-4 → 8e-4 gained +9.2pp (exp-jax-h → exp-jax-i)
- MT-bench scores are ~3-6pp lower than ShareGPT across all checkpoints (MT-bench has more diverse/complex prompts)

All checkpoints: `/path/to/checkpoints/exp-jax-{a..i}/`

---

## Code Layout

```
specjax/
├── train.py                     # Training loop (TTT, checkpointing, W&B)
├── eval.py                      # Evaluation script
├── models/
│   ├── draft/
│   │   └── eagle3.py            # Eagle3 draft head (forward, loss, save/load)
│   ├── target/
│   │   └── glm_flash.py         # GLM-4.7-Flash target model (frozen)
│   └── sharding.py              # SPMD mesh + PartitionSpec helpers
├── data/
│   └── dataset.py               # Tokenisation, bucketing, batch sampling

scripts/
├── run_exp_jax_{a..i}.sh        # Experiment launch scripts
├── prep_dataset_jax_d.sh        # Build 54K mixed dataset
├── setup_jax_env.sh             # Venv + model download (run after preemption)
├── preflight_jax.sh             # Pre-training validation
├── download_models_jax.sh       # HuggingFace model download
├── download_train_data_jax.sh   # ShareGPT dataset download
├── sanity_check_arch.py         # Architecture parity check
├── worker_start_training.sh     # Multi-host worker launcher
└── requirements-train-jax.txt   # Python dependencies

benchmark/scripts/
├── eval_acceptance_pos0.py      # Offline acceptance rate evaluation
├── eval_models/                 # JAX model copies for eval
│   ├── eagle3.py
│   ├── glm_flash.py
│   └── sharding.py
└── run_acceptance_comparison.sh  # Side-by-side GPU vs TPU comparison
```

---

## SpecForge Reference

Key files in `/path/to/shared-storage/SpecForge-internal/` for cross-referencing:

| File | What to look for |
|------|-----------------|
| `specforge/core/eagle3.py:325-336` | Aux layer indices (offset=1 convention) |
| `specforge/core/loss.py:173-201` | LogSoftmaxLoss (`.mean()` over B*T) |
| `specforge/data/preprocessing.py:568-616` | d2t offset format, `used_tokens.sort()` |
| `specforge/modeling/draft/llama3_eagle.py:620-760` | Multi-branch tree attention |
| `specforge/configs/glm4-flash-eagle3.json` | Draft model config (head_dim=128, heads=16) |
