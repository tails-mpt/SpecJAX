# Eagle3 JAX/TPU Training for GLM-5-FP8

**W&B**: [glm5-eagle3-experiments](https://wandb.ai/your-wandb-entity/glm5-eagle3-experiments)
**Hardware**: TPU v4e-32, `ct4e-standard-32t`, GCP SPOT VM

---

## Quick Start (after preemption)

```bash
# 1. Bootstrap venv + download models + data
bash scripts/setup_glm5_env.sh

# 2. Verify TPU
source /path/to/workspace/.venv-train-jax/bin/activate
python3 -c "import jax; print(jax.devices())"
# → [TpuDevice(id=0,...), ..., TpuDevice(id=31,...)]

# 3. Pre-flight checks
bash scripts/preflight_glm5.sh

# 4. Launch (run with sudo for GCS write permissions)
sudo bash -c 'source ~/.specjax.env && source /path/to/workspace/.venv-train-jax/bin/activate && \
  bash scripts/run_exp_glm5_a.sh'

# Monitor
tail -f /path/to/workspace/logs/exp_glm5_a_*.log
```

---

## Hardware & Paths

| Resource | Path | Persistence |
|---|---|---|
| Code repo | (repo root) | GCS (persistent) |
| Checkpoints | `/path/to/checkpoints/exp-glm5-{a..}/` | GCS (persistent) |
| Secrets | `~/.specjax.env` (WANDB_API_KEY, HF_TOKEN) | GCS (persistent) |
| Venv | `/path/to/workspace/.venv-train-jax` | NVMe (ephemeral) |
| Target model | `/path/to/models/GLM-5-FP8` | NVMe (ephemeral) |
| Training logs | `/path/to/workspace/logs/` | NVMe (ephemeral) |
| W&B cache | `/path/to/workspace/wandb/` | NVMe (ephemeral) |
| SpecForge ref | `/path/to/shared-storage/SpecForge-internal/` | GCS (persistent) |

**SPOT VM**: `/path/to/workspace/` is wiped on preemption. Run `setup_glm5_env.sh` to recreate venv and re-download models. Code and checkpoints on `/path/to/shared-storage/` survive.

---

## Architecture

### Target Model

GLM-5-FP8 (GlmMoeDsaForCausalLM): 744B total params, 40B active, MoE with DSA.

| Parameter | Value |
|-----------|-------|
| hidden_size | 6144 |
| num_hidden_layers | 78 |
| num_attention_heads | 64 |
| num_key_value_heads | 64 |
| vocab_size | 154,880 |
| q_lora_rank | 2048 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 192 |
| qk_rope_head_dim | 64 |
| v_head_dim | 256 |
| n_routed_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 2048 |
| routed_scaling_factor | 2.5 |
| first_k_dense_replace | 3 (layers 0-2 dense, 3-77 MoE) |
| rope_theta | 1,000,000 |
| rms_norm_eps | 1e-5 |
| Quantization | FP8 e4m3, block_size=[128,128] |

### Eagle3 Draft Head

Single-layer draft model scaled for GLM-5's hidden_size=6144.

**Weight structure** (~762M params, ~1.5GB):

| Tensor | Shape | Purpose |
|---|---|---|
| `fc.weight` | [6144, 18432] | Projects [low \|\| mid \|\| high] target hidden states (3H → H) |
| `midlayer.input_layernorm.weight` | [6144] | RMSNorm on token embeddings |
| `midlayer.hidden_norm.weight` | [6144] | RMSNorm on FC-projected features |
| `midlayer.self_attn.{q,k,v,o}_proj.weight` | varies | GQA: 32 heads, 8 KV heads, head_dim=192 |
| `midlayer.post_attention_layernorm.weight` | [6144] | RMSNorm after attention |
| `midlayer.mlp.{gate,up,down}_proj.weight` | varies | SiLU-gated MLP (intermediate=16384) |
| `norm.weight` | [6144] | Final RMSNorm |
| `lm_head.weight` | [32000, 6144] | Draft vocab logits |
| `d2t` | [32000] int32 | Draft→target token mapping (stored as offsets) |
| `t2d` | [154880] bool | Target→draft token mask |

**Config**: `rope_theta=1,000,000`, `rms_norm_eps=1e-5`

### Forward Pass

```
Target model (frozen, FP8 dequantized per-layer):
  input_ids → 78 decoder layers → extract hidden states at layers {1, 38, 74}

Eagle3 draft head:
  FC:       [low_h || mid_h || high_h] (3H=18432) → projected_features (H=6144)
  Embed:    input_ids → embed_tokens (H)
  Midlayer: Q = cat(norm(embed), norm(projected)) (2H=12288)
            Standard causal attention + GQA + RoPE
            MLP → hidden_out (H)
  Norm + lm_head → draft logits [B, T, 32000]
```

### DSA (Dynamic Sparse Attention) — SKIPPED

GLM-5 includes a DSA indexer that selects top-2048 positions per query for long-context efficiency. For Eagle3 training with `max_length <= 1024`, `topk=2048 >= seq_len` means ALL positions are selected, making DSA equivalent to full causal attention. The indexer is not implemented; its weights are skipped during loading.

### FP8 Weight Loading

GLM-5-FP8 stores weights in `float8_e4m3fn` format with block-wise scaling (`block_size=[128,128]`). During loading:

1. Each safetensors shard is read sequentially (memory management for ~400GB+ on disk)
2. FP8 weights are dequantized to bf16 using their paired `*.weight_scale` tensors
3. Non-quantized weights (embeddings, norms, router, lm_head) are cast directly to bf16
4. DSA indexer and MTP weights are skipped
5. Per-expert weights are stacked into `[E, 2I, D]` / `[E, D, I]` tensors
6. All tensors are sharded onto the SPMD mesh

### SPMD Sharding

1D tensor-parallel across 32 TPU chips: `Mesh(jax.devices(), ("tp",))`. Dense weights shard on leading dim, MoE experts shard on intermediate dim.

---

## Key Differences from GLM-Flash Eagle3

| Aspect | GLM-Flash | GLM-5-FP8 |
|--------|-----------|-----------|
| Target model size | 9B | 744B (40B active) |
| Hidden size | 2048 | 6144 |
| Aux layer indices | {1, 22, 43} | {1, 38, 74} |
| MoE experts | 64 | 256 |
| Experts per token | 4 | 8 |
| Dense layers | Layer 0 | Layers 0-2 |
| Weight format | bf16 | FP8 e4m3 (dequantized at load) |
| TPU config | v6e-4 (4 chips) | v4e-32 (32 chips) |
| TP size | 4 | 32 |
| Attention | MLA | MLA (DSA skipped) |
| Eagle3 fc.weight | [2048, 6144] | [6144, 18432] |
| Eagle3 heads | 16h, 4kv, dim=128 | 32h, 8kv, dim=192 |

---

## Code Layout

```
specjax/
├── train.py                     # Training loop (--target-model-type glm5)
├── eval.py                      # Evaluation script
├── models/
│   ├── draft/
│   │   └── eagle3.py            # Eagle3 draft head + eagle3_config_for_glm5()
│   ├── target/
│   │   ├── glm5_fp8.py          # GLM-5-FP8 target model (frozen, FP8 dequant)
│   │   └── glm_flash.py         # GLM-4.7-Flash target model (frozen)
│   └── sharding.py              # SPMD mesh + PartitionSpec (tp_size param)
├── data/
│   └── dataset.py               # Tokenisation, bucketing, batch sampling

scripts/
├── run_exp_glm5_a.sh            # First GLM-5 experiment script
├── setup_glm5_env.sh            # Venv + model download for GLM-5
├── preflight_glm5.sh            # Pre-training validation for GLM-5
├── download_models_glm5.sh      # HuggingFace GLM-5-FP8 download
├── run_exp_jax_{a..i}.sh        # GLM-Flash experiment scripts
├── setup_jax_env.sh             # GLM-Flash environment setup
└── requirements-train-jax.txt   # Python dependencies
```

---

## Memory Budget (v4e-32, TP=32)

| Component | Total Size | Per-chip (32 chips) |
|-----------|-----------|-------------------|
| Target weights (FP8 on-device) | ~744GB | ~23GB |
| Eagle3 weights (bf16) | ~1.5GB | ~1.5GB (replicated) |
| Eagle3 optimizer (AdamW: 3x params) | ~4.5GB | ~4.5GB |
| Activations (B=1, T=512) | ~2GB est. | ~2GB |
| **Total** | | **~31GB** |
| **Available per chip** | | **32GB** |

If OOM: reduce `max_length` to 256, or shard Eagle3 params across TP.

---

## Experiment Results

| Experiment | Mode | LR | Epochs | Notes |
|---|---|---|---|---|
| exp-glm5-a | Single-step | 1e-4 | 3 | First validation run, B=1, maxlen=512 |

All checkpoints: `/path/to/checkpoints/exp-glm5-{a..}/`
