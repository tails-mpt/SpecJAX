# Eagle3 JAX/TPU Training for Qwen3-Coder-Next

**Status**: Planning — not yet implemented
**Date**: 2026-03-23
**Predecessor**: [GLM-4.7-Flash Eagle3 JAX training](glm-flash-eagle3.md)

---

## Goal

Train an EAGLE3 speculative-decoding draft head for `Qwen/Qwen3-Coder-Next` using pure JAX/XLA on Cloud TPU, following the same architecture and patterns established for GLM-4.7-Flash. This is the "second model" that triggers Phase 2 of the JAX SpecForge framework — the point where model-specific vs. reusable abstractions become clear.

Additionally, test **Hypothesis 3**: that selecting hidden states from attention layers (instead of the default GDN layers) significantly improves EAGLE3 accuracy for hybrid GDN architectures.

---

## Hardware Requirements

### Why This Needs a Bigger TPU

GLM-4.7-Flash is ~57GB in bf16 and fits on TPU v6e-4 (128GB HBM). Qwen3-Coder-Next is **~556GB in bf16** — roughly 10× larger, driven by 512 MoE experts (vs 64 for GLM).

**Weight breakdown (bf16)**:

| Component | GLM-4.7-Flash | Qwen3-Coder-Next |
|-----------|---------------|-------------------|
| Embeddings + lm_head | ~1.3GB | ~4.2GB |
| Attention/GDN weights | ~0.9GB | ~9.5GB (12 ATT + 36 GDN layers) |
| MoE experts | ~55GB (46 layers × 64 experts × 1536 intermediate) | **~542GB** (48 layers × 512 experts × 512 intermediate) |
| **Total** | **~57GB** | **~556GB** |

### TPU Sizing

| TPU Slice | HBM | bf16 (556GB) | FP8 (~278GB) |
|-----------|-----|--------------|--------------|
| v6e-4 | 128GB | ✗ | ✗ |
| v6e-8 | 256GB | ✗ | ✗ |
| **v6e-16** | **512GB** | Tight (no activation headroom) | **✓ (~230GB for activations)** |
| **v6e-32** | **1024GB** | **✓ (comfortable)** | ✓ (overkill) |

**Recommendation**: Start with **v6e-16 + FP8 weights** (Qwen3-Coder-Next-FP8 available on HuggingFace). Fall back to v6e-32 + bf16 if FP8 JAX loading proves problematic.

### GPU Reference

SpecForge GPU training used **8× H100 80GB** (640GB total), TP=4, DP=2.

---

## Qwen3-Coder-Next Architecture Reference

### Overview

| Property | Value |
|----------|-------|
| Total parameters | ~80B |
| Active parameters | ~3B per token |
| Total layers | 48 |
| Hidden size | 7168 |
| Vocabulary size | 151,936 |
| `rms_norm_eps` | 1e-6 |
| `full_attention_interval` | 4 |
| MoE experts | 512 total / 10 active + 1 shared |
| MoE intermediate size | 512 per expert |
| Shared expert intermediate size | 512 |
| Max position embeddings | 32,768 (rope-extended to 256K) |

### Hybrid Layer Architecture (3:1 GDN-to-Attention)

The model alternates 3 Gated DeltaNet (GDN) layers with 1 full attention layer, repeated 12 times:

```
Layers 0-3:   GDN  GDN  GDN  ATT
Layers 4-7:   GDN  GDN  GDN  ATT
...
Layers 44-47: GDN  GDN  GDN  ATT
```

**Layer type rule**: layer `i` is attention if `(i + 1) % 4 == 0`, else GDN.

| Layer Type | Count | 0-indexed Indices |
|------------|-------|-------------------|
| **Attention** | 12 | 3, 7, 11, 15, 19, **23**, 27, 31, 35, 39, 43, **47** |
| **GDN** | 36 | 0, 1, 2, 4, 5, 6, 8, ..., 44, 45, 46 |

### Full Attention Layers

- **Type**: Grouped Query Attention (GQA) with gating
- **Heads**: 16 query heads, 2 KV heads (8× GQA compression)
- **Head dim**: 256
- **RoPE**: Partial — `partial_rotary_factor=0.25`, so 64 of 256 dims get RoPE
- **Gating**: Q projection outputs 2× dim, split into `(q, gate)`. Output = `attn_output * sigmoid(gate)`
- **QK normalization**: RMSNorm applied per-head to Q and K before attention

### GDN Layers (Gated DeltaNet)

- **Type**: Linear recurrent attention with gated delta rule
- **Value heads**: 32, **Key heads**: 16
- **Key/Value dim**: 128 each
- **Convolution**: 1D causal conv (kernel_size=4, depthwise)
- **Recurrent state**: `[batch, heads, key_dim, value_dim]` — O(1) per position
- **Update rule**: `state = state * gate + k * (v - kv_mem) * beta`
- **Output normalization**: `RMSNormGated` — applies `hidden * w * silu(gate)`

**Key insight**: GDN hidden states are compressed recurrent states optimized for long-range tracking, not the immediate retrieval signal needed for next-token prediction. Attention layer hidden states are weighted token combinations better suited for EAGLE3 feature fusion.

### MoE

- Every layer has MoE (`decoder_sparse_step=1`)
- 512 routed experts, 10 selected per token + 1 shared expert
- `norm_topk_prob=True` — routing probabilities are normalized
- JAX implementation must use **static-shape einsum dispatch** (same pattern as GLM) to avoid XLA recompilation from data-dependent routing

---

## Hypothesis 3: Attention vs GDN Layer Selection

### Background

EAGLE3's tri-layer feature fusion concatenates hidden states from three target model layers (low, mid, high) as input to the FC projection. The SpecForge default formula selects:

```python
# SpecForge eagle3.py lines 329-332 (0-indexed layer numbers)
low  = 1       # layer 1
mid  = N//2 - 1  # layer 23 (for N=48)
high = N - 4   # layer 44
```

For Qwen3-Coder-Next (N=48), this gives layers **{1, 23, 44}** (0-indexed):

| Layer | Type | Suitable for EAGLE3? |
|-------|------|---------------------|
| 1 | **GDN** | ✗ Compressed recurrent state |
| 23 | **Attention** | ✓ Retrieval-oriented |
| 44 | **GDN** | ✗ Compressed recurrent state |

2 of 3 selected layers are GDN. The all-12 attention layers (at positions 3, 7, 11, ..., 47) are entirely skipped.

### Hypothesis

Selecting features from attention layers should yield higher EAGLE3 accuracy because attention hidden states encode immediate retrieval signals (weighted token combinations) rather than compressed long-range recurrent states.

### Proposed Layer Configurations

**Config A — All attention** (hypothesis-informed):
- Layers **{3, 23, 47}** (0-indexed) — first, middle, last attention layers
- Matches the SGLang inference patch already implemented in `specjax/models/target/qwen3_next.py`

**Config B — SpecForge default** (baseline):
- Layers **{1, 23, 44}** (0-indexed) — standard SpecForge formula
- 2 GDN + 1 attention (mixed)

### Experiment Design

Pre-compute features for BOTH configurations in the same extraction pass (save aux_hidden_states for all 6 unique layer indices: {1, 3, 23, 44, 47}). Train separate draft heads and compare acc@pos0.

---

## Key Differences from GLM Implementation

| Dimension | GLM-4.7-Flash | Qwen3-Coder-Next | Impact |
|-----------|---------------|-------------------|--------|
| Target hidden_size | 2048 | **7168** | FC: 3×2048=6144→2048 becomes 3×7168=21504→2048 |
| Vocab size | 154,880 | **151,936** | Different d2t/t2d mappings, different tokenizer |
| rms_norm_eps | 1e-5 | **1e-6** | Config change |
| Attention type | MLA (all layers) | **GQA + gating** (12 layers) | New attention implementation |
| GDN layers | None | **36 layers** | Entirely new component in JAX |
| MoE scale | 64 experts, 4 active | **512 experts, 10 active** | 10× more expert weights, different dispatch |
| Model size | ~57GB bf16 | **~556GB bf16** | Requires v6e-16+ (vs v6e-4) |
| Embedding reuse | Draft uses target embed (same H=2048) | **Draft needs own embed** (target H=7168 ≠ draft H=2048) | New `embed_tokens.weight` in draft params |
| RoPE | Full (64-dim head) | **Partial** (64 of 256 dims) | Different RoPE application |
| Shared expert | 1 shared, standard size | 1 shared, intermediate=512 | Minor config change |

### Critical: Draft Model Embedding

For GLM, the draft model reuses the target model's embedding table (`embed_w[input_ids]`) because both have hidden_size=2048. For Qwen3, the target embedding is [151936, 7168] while the draft expects H=2048.

SpecForge solves this by giving the draft model its **own embedding table**:
```python
# llama3_eagle.py line 1343
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, ...)
# config.hidden_size = 2048 (draft), NOT 7168 (target)
```

The JAX implementation must add `embed_tokens.weight [151936, 2048]` to the Eagle3 params, trained alongside the other draft weights. This tensor is NOT in the current GLM checkpoint format — it's a new addition for models where target and draft hidden sizes differ.

---

## Implementation Plan

### New Files

#### 1. `specjax/models/target/qwen3_next.py` — Frozen target model

Pure JAX forward pass for Qwen3-Coder-Next. Pattern: follow `specjax/models/target/glm_flash.py`.

Must implement:
- **`Qwen3NextConfig`** — dataclass from HF `config.json`
- **`load_params(model_path, mesh)`** — load safetensors shards, stack MoE experts into `[E, 2I, D]` / `[E, D, I]`, apply SPMD sharding
- **`gdn_forward()`** — Gated DeltaNet layer: causal conv1d → gated delta rule (chunk mode for training)
- **`attention_forward()`** — GQA with gating, partial RoPE, QK norm
- **`moe_forward()`** — Static-shape einsum dispatch for 512 experts (same pattern as GLM, scaled up)
- **`qwen3_next_forward(params, input_ids, attention_mask, aux_layer_indices)`** — Full forward returning `(last_hidden, embed_w, aux_hidden_states[B,T,3*7168], target_logits[B,T,151936])`

Key challenge: the GDN layer requires implementing the **gated delta rule** in JAX. For training (full sequences, not autoregressive), use the **chunk mode** (`torch_chunk_gated_delta_rule` equivalent) which processes fixed-size chunks in parallel. This avoids the sequential recurrent mode.

**Reference implementations**:
- PyTorch: `transformers/models/qwen3_next/modular_qwen3_next.py` (in your venv's `site-packages/transformers/models/qwen3_next/`)
- HuggingFace `fla` (Flash Linear Attention) library for chunked delta rule

#### 2. `data/offline_dataset.py` — Pre-computed feature dataset (optional)

If memory requires two-phase training (extract features → train draft), this loads pre-computed `aux_hidden_states` and `target_logits` from safetensors. Reuses `BucketBatchSampler` pattern from `specjax/data/dataset.py`.

Not needed if v6e-32 can run the full target model online.

#### 3. Shell scripts

- **`setup_qwen3_jax_env.sh`** — Bootstrap venv, download Qwen3-Coder-Next-FP8 model
- **`run_exp_qwen3_jax_a.sh`** — First experiment: attention layers [3, 23, 47]
- **`run_exp_qwen3_jax_b.sh`** — Second experiment: default layers [1, 23, 44]

### Files to Modify

#### 1. `specjax/models/draft/eagle3.py` — Make model-agnostic

Changes:
1. **Add `target_hidden_size` to Eagle3Config** (default 0 = same as hidden_size, for backward compat)
2. **FC init**: `fc.weight` shape becomes `[H, 3 * target_H]` when `target_hidden_size > 0`
3. **Add `embed_tokens.weight`** to params when `target_hidden_size != hidden_size`:
   - Shape: `[vocab_size, hidden_size]` (draft's own 2048-dim embedding)
   - Initialized with Xavier uniform (or projected from target embedding)
4. **`eagle3_forward`**: Use `params["embed_tokens.weight"][input_ids]` instead of `embed_w[input_ids]` when draft has own embedding
5. **`save_eagle3_checkpoint`**: Include `target_hidden_size` in config.json, save `embed_tokens.weight`
6. **`load_eagle3_params`**: Infer `target_hidden_size` from FC weight shape: `fc.weight.shape[1] // 3`
7. **Config defaults for Qwen3**: `vocab_size=151936, rms_norm_eps=1e-6`

#### 2. `specjax/train.py` — Support multiple target models

Changes:
1. **Add `--target-model-type`** argument: `glm` (default) or `qwen3`
2. **Conditional import**: `from specjax.models.target.qwen3_next import ...` when `--target-model-type qwen3`
3. **Pass `aux_layer_indices`** to target forward: `[3, 23, 47]` or `[1, 23, 44]` (new arg `--aux-layers`)
4. **Handle `embed_w` difference**: when target and draft hidden sizes differ, the draft model uses its own embedding (already in `params`), not the target's
5. **Update `make_train_step_ttt`**: pass `embed_w` from draft params, not target model

#### 3. `specjax/models/sharding.py` — Scale for 512 experts

The existing sharding logic should mostly work (3D MoE tensors shard on expert dim). May need to handle the larger mesh (16+ chips) with 2D DP×TP.

### Files That Don't Change

- `specjax/data/dataset.py` — Model-agnostic (just tokenization + bucketing). Only the tokenizer changes.
- `specjax/eval.py` — Draft head evaluation is model-agnostic (it just runs the draft model + target model). Will need the Qwen3 target model implemented but the eval logic is reusable.

---

## Interoperability: SpecForge Checkpoint Compatibility

The trained JAX checkpoint must be loadable by SpecForge/SGLang for GPU inference.

### Draft Head Config (`config.json`)

```json
{
  "architectures": ["LlamaForCausalLMEagle3"],
  "model_type": "llama",
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "num_hidden_layers": 1,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "vocab_size": 151936,
  "draft_vocab_size": 32000,
  "target_hidden_size": 7168,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-06,
  "torch_dtype": "bfloat16"
}
```

Key: `target_hidden_size: 7168` tells SpecForge to create `fc = Linear(7168*3, 2048)` (see `llama3_eagle.py` line 1348-1351).

### Weight Tensors (`model.safetensors`)

| Tensor | Shape | Notes |
|--------|-------|-------|
| `fc.weight` | [2048, **21504**] | 3 × 7168 = 21504 (vs 6144 for GLM) |
| `embed_tokens.weight` | [151936, 2048] | **NEW** — draft's own embedding (not in GLM checkpoints) |
| `midlayer.*` | (same as GLM) | Architecture unchanged |
| `norm.weight` | [2048] | Same |
| `lm_head.weight` | [32000, 2048] | Same |
| `d2t` | [32000] int64 | Stored as offsets (SpecForge convention) |
| `t2d` | [151936] bool | Authoritative source for vocab mapping |

### Interop Checklist (from GLM lessons)

Apply all 7 fixes from the GLM experience from day one:

| # | Fix | Value for Qwen3 |
|---|-----|-----------------|
| 1 | rope_theta | 1,000,000 |
| 2 | rms_norm_eps | **1e-6** (Qwen3-specific, different from GLM's 1e-5) |
| 3 | Aux layer indices | Config A: {3, 23, 47} / Config B: {1, 23, 44} — NOT SafeAILab's formula |
| 4 | d2t ordering | Sorted by target token ID ascending |
| 5 | d2t format | Offsets: `d2t[i] = actual_id - i` |
| 6 | Norm weights | float32 during training, cast to bfloat16 on save |
| 7 | Draft config | head_dim=128, num_heads=16, num_kv_heads=4 |

---

## Existing GPU Results (SpecForge Baseline)

### Exp A: SpecForge Training

- **Data**: 1,156 samples (qwen3_next_phase4_train_clean.jsonl)
- **Config**: TP=4, DP=2, LR=1e-4, batch=1, epochs=3, TTT=7, max_len=2048
- **Checkpoint**: `/path/to/workspace/SpecForge-internal/outputs/qwen3-coder-next-eagle3-exp-a/epoch_2_step_432/`
- **Layer selection**: Default SpecForge formula → layers {1, 23, 44} (2 GDN + 1 attention)

### Inference Results

**B=1 (single-user, latency-sensitive)**:

| Dataset | Baseline (tok/s) | Aurora Draft | Exp A Draft | Aurora vs Base | Exp A vs Base |
|---------|------------------|-------------|-------------|----------------|---------------|
| HumanEval | 127.6 | 132.1 | **136.96** | +1.04× | **+1.07×** |
| Aider | 131.6 | **155.8** | 112.9 | **+1.18×** | 0.86× |
| Terminal-Bench | 128.4 | **136.7** | 122.9 | **+1.06×** | 0.96× |
| SWEBench-Pro | 131.5 | **132.3** | 109.7 | **+1.01×** | 0.83× |
| MT-Bench | 131.3 | 114.4 | 93.5 | 0.87× | 0.71× |

**Acceptance rates**: Aurora 22–31%, Exp A 20–31%.

**B=32**: Both regress uniformly (0.41–0.62×) — structural MoE expert dispatch overhead, not a training gap.

### Aurora Reference (Together Computer)

- 10K training steps, 80K code requests
- Architecture: `LlamaForCausalLMEagle3`, hidden_size=2048, draft_vocab=32K
- Best B=1: 1.18× speedup on Aider

### Why Acceptance Rates Are Low

1. **Coding token entropy** — code tokens are harder to predict than prose
2. **GDN feature quality** — 2/3 of tri-layer features come from GDN (compressed recurrent state), not attention (Hypothesis 3)
3. **Insufficient data** — Exp A: 1,156 samples vs Aurora's 80K

---

## Experiment Plan

### Dataset

Use **54K mixed dataset** (proven effective for GLM: 59.71% acc@pos0) tokenized with Qwen3 tokenizer:
- 45% ShareGPT (`anon8231489123/ShareGPT_Vicuna_unfiltered`)
- 35% UltraChat (`HuggingFaceH4/ultrachat_200k`)
- 20% PerfectBlend (`mlabonne/open-perfectblend`)

Build script: adapt `prep_dataset_jax_d.sh` to use Qwen3 tokenizer (`Qwen/Qwen3-Coder-Next`).

### Experiments

| Experiment | Aux Layers | LR | Epochs | TTT | Dataset | Purpose |
|------------|-----------|-----|--------|-----|---------|---------|
| **exp-qwen3-jax-a** | **{3, 23, 47}** (attention) | 1e-4 | 3 | 7 | 54K mixed | Hypothesis 3: attention layers |
| **exp-qwen3-jax-b** | **{1, 23, 44}** (default) | 1e-4 | 3 | 7 | 54K mixed | Hypothesis 3: default (GDN) layers |
| exp-qwen3-jax-c | Best of a/b | **8e-4** | 3 | 7 | 54K mixed | LR scaling (if linear scaling helps like GLM) |

**Hyperparameters** (matching GLM exp-jax-h baseline):

| Parameter | Value |
|-----------|-------|
| Batch size per chip | 4 (may need reduction due to larger aux_hidden_states) |
| Gradient accumulation | 8 |
| Effective batch size | varies with TPU slice (32 on v6e-4, scales with DP) |
| Max sequence length | 1024 |
| Warmup | 3% of total steps |
| Optimizer | AdamW (optax) |
| Loss | KL divergence vs target softmax |
| Loss weighting | 0.8^k geometric decay |

### Evaluation

1. **TPU eval** (offline): Run `python3 -m specjax.eval` with Qwen3 target model on the same TPU slice
2. **Compare**: acc@pos0 between Config A and Config B, and vs SpecForge Exp A baseline

---

## Critical File References

### Existing Code (to reuse/modify)

| Component | Path | Action |
|-----------|------|--------|
| Eagle3 draft model | `specjax/models/draft/eagle3.py` | Modify: add `target_hidden_size`, own embedding |
| Training script | `specjax/train.py` | Modify: add `--target-model-type`, aux layer config |
| SPMD sharding | `specjax/models/sharding.py` | May need update for larger mesh |
| Dataset pipeline | `specjax/data/dataset.py` | Reuse as-is (change tokenizer only) |
| Eval script | `specjax/eval.py` | Reuse with Qwen3 target |
| GLM target model | `specjax/models/target/glm_flash.py` | **Reference pattern** for Qwen3 target |

### SpecForge References

| Component | Path | What to look for |
|-----------|------|-----------------|
| Draft model (PyTorch) | `SpecForge-internal/specforge/modeling/draft/llama3_eagle.py` | `target_hidden_size` FC (line 1348), `embed_tokens` (line 1343) |
| Online training | `SpecForge-internal/specforge/core/eagle3.py` | Aux layer formula (lines 329-332), TTT loop |
| Qwen3 draft config | `SpecForge-internal/configs/qwen3-coder-next-eagle3.json` | Draft model hyperparameters |
| Qwen3 target backend | `SpecForge-internal/specforge/modeling/target/custom_backend/qwen3.py` | Target model integration |
| Loss function | `SpecForge-internal/specforge/core/loss.py` | LogSoftmaxLoss (lines 173-201) |

### Qwen3 Model Implementation (HuggingFace)

| File | Path (in venv) | What to look for |
|------|----------------|-----------------|
| Config | `.venv-train-jax/.../transformers/models/qwen3_next/configuration_qwen3_next.py` | `full_attention_interval`, layer_types, MoE config |
| Model | `.venv-train-jax/.../transformers/models/qwen3_next/modeling_qwen3_next.py` | Full attention, cache, MoE block |
| Modular | `.venv-train-jax/.../transformers/models/qwen3_next/modular_qwen3_next.py` | GDN implementation (`Qwen3NextGatedDeltaNet`), chunk delta rule |

### Documentation

| Doc | Path |
|-----|------|
| GLM Eagle3 JAX training | `docs/training/glm-flash-eagle3.md` |

---

## Implementation Complexity Estimate

| Component | Effort | Notes |
|-----------|--------|-------|
| `specjax/models/target/qwen3_next.py` (target model) | **High** | ~800-1000 lines. GDN chunked delta rule is the hardest part. MoE dispatch reuses GLM pattern. |
| `specjax/models/draft/eagle3.py` modifications | Low | ~50 lines changed. Add target_hidden_size, embed_tokens. |
| `specjax/train.py` modifications | Medium | ~100 lines. Conditional imports, aux layer config, embed handling. |
| Shell scripts + dataset prep | Low | Adapt existing scripts. |
| **Total new code** | | ~1000-1200 lines (mostly the target model) |

### GDN Implementation Strategy

The GDN (Gated DeltaNet) layer is the only fundamentally new component. Two approaches:

1. **Chunk mode** (recommended for training): Process fixed-size chunks in parallel. The HuggingFace `fla` library provides `torch_chunk_gated_delta_rule`. Port the chunked scan to JAX using `jax.lax.scan` over chunks.

2. **Recurrent mode** (for reference): Token-by-token recurrent state update. Simpler to implement but O(T) sequential — not suitable for training.

The chunk mode computes intra-chunk attention (quadratic within chunk) and inter-chunk recurrent state (linear scan across chunks). Chunk size 64-128 balances parallelism vs memory.

---

## Open Questions

1. **FP8 in JAX**: Does `jax.numpy.float8_e4m3fn` work natively on TPU v6e? Or do we need to dequantize to bf16 during loading?
2. **GDN chunk implementation**: Is there an existing JAX implementation of chunked delta rule (e.g., in the `fla` library)? If not, porting from PyTorch is the main implementation risk.
3. **Draft embedding initialization**: Should we initialize `embed_tokens.weight` from a truncated/projected version of the target embedding, or random Xavier? SpecForge likely uses random init since dimensions differ.
4. **Activation memory**: With hidden_size=7168 and 48 layers, target model activations for aux layer extraction may be significant. Need to estimate per-batch activation memory to determine max batch size.
5. **Shared expert**: Qwen3's shared expert runs alongside routed experts for every token. The static-shape dispatch needs to handle this (separate einsum path, then add).
