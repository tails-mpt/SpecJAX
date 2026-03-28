# MiniMax-M2.5 EAGLE3 Training Plan

## Context

**What**: Implement MiniMax-M2.5 as a frozen target model in SpecJAX and train an EAGLE3 speculative decoding draft head for it.

**Why**: MiniMax-M2.5 is a 229B MoE model (256 experts, top-8, 10B active) gaining traction (~511K DL/mo). Two experimental vLLM-only EAGLE3 drafts exist (`novita/Eagle3-Spec-Minimax-M2.5-Exp{14,15}`) with negligible downloads (~230/mo each) — there's a quality gap to fill. Getting the code in place now is a head start even if training waits for hardware.

**Known blockers**: sglang-jax has no `minimax_m2` backend (fork required for deployment), custom `model_type` needs `trust_remote_code`, built-in MTP (3 modules) reduces urgency. These are deployment concerns — code and training can proceed independently.

## Model Architecture

Source: [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) `config.json`

| Parameter | Value |
|-----------|-------|
| Total params | ~229B |
| Active params/token | ~10B (top-8 of 256 experts) |
| `hidden_size` | 3072 |
| `num_hidden_layers` | 62 |
| `num_attention_heads` | 48 |
| `num_key_value_heads` | 8 (GQA ratio 6:1) |
| `head_dim` | 128 |
| `rotary_dim` | 64 (partial RoPE — first 64 of 128 dims) |
| `rope_theta` | 5,000,000 |
| `num_local_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `intermediate_size` | 1536 (per expert) |
| `shared_intermediate_size` | 0 (no shared expert) |
| `vocab_size` | 200,064 |
| `rms_norm_eps` | 1e-6 |
| `use_qk_norm` | true (`per_layer` — RMSNorm on flat Q/K before reshape) |
| `scoring_func` | sigmoid (with `e_score_correction_bias`) |
| `max_position_embeddings` | 196,608 |
| FP8 format | `float8_e4m3fn`, block-wise 128×128 |
| Modules NOT quantized | `gate`, `e_score_correction_bias`, `lm_head` |

### Weight naming convention

```
model.embed_tokens.weight                                          [200064, 3072]
model.layers.{L}.input_layernorm.weight                            [3072]
model.layers.{L}.self_attn.{q,k,v,o}_proj.weight                  FP8 + scale_inv
model.layers.{L}.self_attn.q_norm.weight                           [6144]  (48*128)
model.layers.{L}.self_attn.k_norm.weight                           [1024]  (8*128)
model.layers.{L}.post_attention_layernorm.weight                   [3072]
model.layers.{L}.block_sparse_moe.gate.weight                     [256, 3072]  (not quantized)
model.layers.{L}.block_sparse_moe.e_score_correction_bias         [256]  (not quantized)
model.layers.{L}.block_sparse_moe.experts.{E}.w1.weight           FP8 [1536, 3072]  (gate_proj)
model.layers.{L}.block_sparse_moe.experts.{E}.w3.weight           FP8 [1536, 3072]  (up_proj)
model.layers.{L}.block_sparse_moe.experts.{E}.w2.weight           FP8 [3072, 1536]  (down_proj)
model.norm.weight                                                  [3072]
lm_head.weight                                                     [200064, 3072]  (not quantized)
```

125 safetensors shards, ~466 GB total (FP8 + scales), 96K+ tensors.

### Key architectural notes

1. **Partial RoPE**: Only first 64 of 128 head_dim dimensions get rotary encoding. Uses existing `apply_partial_rope()`.

2. **QK-Norm (per_layer)**: RMSNorm applied to the FLAT projected Q/K tensors (before reshape to multi-head). Weight shapes: q_norm `[6144]`, k_norm `[1024]`. This is different from per-head QK-norm in Qwen3-Next.

3. **MoE routing with post-sigmoid bias**: MiniMax adds `e_score_correction_bias` AFTER sigmoid, only for expert selection. Final routing weights use original sigmoid scores (without bias), then renormalize. This differs from the shared `topk_router()` which adds bias pre-sigmoid, so we implement `_minimax_topk_router()`.

4. **No shared expert**: Unlike GLM-Flash (1 shared) or Qwen3-Next (1 shared), MiniMax has `shared_intermediate_size=0`.

5. **Lightning Attention**: MiniMax uses a hybrid linear/softmax attention mechanism. For EAGLE3 training (teacher forcing, short sequences 512-2048 tokens), standard softmax attention with causal mask is functionally equivalent. The Q/K/V/O projection weights are identical regardless — no need to implement the linear-attention fast path.

6. **MTP modules**: Config has `num_mtp_modules=3` (built-in speculative decoding). These weights are skipped during loading — not needed for EAGLE3 training.

7. **Aux layers for EAGLE3**: Layers {1, 30, 58} (indices: 1, num_layers//2-1, num_layers-4).

## Hardware Analysis

### Can v6e-64 run this?

FP8 weights: ~229 GB (1 byte/param) + scales (~2-3 GB).

| TP | Weight/chip | Headroom (32 GB chip) | Verdict |
|----|------------|----------------------|---------|
| TP=4 (default) | 57.3 GB | **-25.3 GB** | Does NOT fit |
| TP=8 | 28.6 GB | 3.4 GB | **Extremely tight** — transient bf16 dequant of MoE layers during forward will likely OOM |
| TP=16 | 14.3 GB | 17.7 GB | **Fits comfortably** |
| TP=32 | 7.2 GB | 24.8 GB | Very comfortable |

**Recommendation: TP=16 on v6e-64** (64 chips / 16 = DP=4).

Per-chip memory breakdown at TP=16:
- Target model weights (FP8→bf16): ~14.3 GB
- Transient MoE dequant per layer: ~1.5 GB (freed after each layer)
- Draft head params + AdamW states: ~0.5 GB
- Forward activations (B=1, T=512): ~0.5 GB
- XLA compilation overhead: ~2-4 GB
- **Total: ~19 GB** → 13 GB headroom on 32 GB chip

### Training config (TP=16, DP=4, v6e-64)

```json
{
    "batch_size": 1,
    "max_length": 512,
    "ttt_length": 1,
    "learning_rate": 3e-4,
    "grad_accum_steps": 8,
    "num_epochs": 3
}
```

Effective batch size: DP(4) × B(1) × grad_accum(8) = 32.

Start with TTT=1 for memory validation, then try TTT=7 if headroom allows.

## Implementation

### Files created/modified

| File | Action | Notes |
|------|--------|-------|
| `specjax/models/target/minimax_m2.py` | Created | Config, load_params (FP8 dequant + key rename + expert stacking), forward pass |
| `specjax/models/target/__init__.py` | Modified | Added `minimax_m2` to `TARGET_MODELS` registry |
| `specjax/models/sharding.py` | Modified | `make_mesh(tp=4)` — configurable TP size (backward compatible) |
| `specjax/train.py` | Modified | Added `--tp` CLI flag, `minimax_m2` EAGLE3 config mapping |
| `specjax/eval.py` | Modified | Added `--tp` CLI flag |
| `configs/minimax_m2.json` | Created | Training hyperparameters |
| `scripts/download/download_minimax_m2.sh` | Created | HuggingFace model download |
| `scripts/run/run_exp_minimax_m2.sh` | Created | Training launcher with TP=16 |

### Configurable TP (prerequisite for all large models)

`sharding.py:make_mesh()` now accepts a `tp` argument (default=4, fully backward compatible). This also unblocks:
- Qwen3-32B (TP=8, v4-32)
- DeepSeek-R1-Distill-Llama-70B (TP=8, v4-64)
- Llama-3.3-70B-Instruct (TP=8, v4-64)
- Qwen2.5-72B-Instruct (TP=8, v4-64)

### Weight loading pipeline

1. Load 125 safetensors shards
2. Skip MTP module weights (`model.mtp_*`)
3. Dequantize FP8 → bf16 via `dequant_fp8_block(block_h=128, block_w=128)`
4. Rename keys: `block_sparse_moe` → `mlp`, `w1` → `gate_proj`, `w3` → `up_proj`, `w2` → `down_proj`
5. Stack experts via `stack_moe_experts()` → `[E, 2I, D]` gate_up_proj + `[E, D, I]` down_proj
6. Shard onto SPMD mesh via `shard_params()`

### Custom MoE router

```python
# MiniMax post-sigmoid bias routing:
scores = sigmoid(hidden @ gate.T)           # [T, 256]
selection_scores = scores + bias            # bias shifts selection only
top_k = argsort(-selection_scores)[:, :8]   # select top-8
weights = gather(scores, top_k)             # use ORIGINAL scores
weights = weights / sum(weights)            # renormalize
```

This differs from the shared `topk_router()` which adds bias to logits before sigmoid.

## Verification

1. **Dry run**: `python -m specjax.train --target-model-path /path/to/MiniMax-M2.5 --target-model-type minimax_m2 --data-path data/sharegpt.jsonl --output-dir /tmp/test --max-steps 5 --tp 16`
2. **Loss decreases** over 5 steps
3. **Memory**: Monitor per-chip HBM via `scripts/monitoring/tpu_monitor.py`
4. **Checkpoint round-trip**: Save + reload, verify no silent corruption

## Next steps

1. Download model weights (~230 GB) to training NFS
2. Run dry-run on v6e-64 to validate memory fit
3. If TTT=1 fits, try TTT=7 for higher draft accuracy
4. Train 3 epochs, target acc_0 ≥ 60%
5. Evaluate against existing `novita/Eagle3-Spec-Minimax-M2.5-Exp14` baseline
6. If sglang-jax gains `minimax_m2.py` support, release checkpoint
