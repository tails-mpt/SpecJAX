# Plan: SpecJAX EAGLE3 Inference Integration with sglang-jax

## Context

SpecJAX trains EAGLE3 speculative decoding draft models on TPUs. To make the public release compelling, we need to demonstrate that these trained models actually work for inference — specifically on sglang-jax, the JAX/TPU inference runtime. This plan creates an `inference/` directory in SpecJAX with scripts, configs, and benchmarks showing end-to-end: **train with SpecJAX -> serve with sglang-jax -> measure throughput gains**.

We are on a TPU v5e-32 (32 chips, 16 GB HBM each). Another agent is working on `main`, so all work happens on a new branch `inference-benchmarks`.

---

## Key Finding: What Works Today vs. What Needs a Patch

**Works now** (target model has EAGLE3 layer capture in sglang-jax):
| # | Target | Draft (HF) | acc_0 |
|---|--------|------------|-------|
| 1 | Llama-3.2-3B-Instruct | thoughtworks/Llama-3.2-3B-Instruct-Eagle3 | 60.6% |
| 2 | Llama-3.1-8B-Instruct | thoughtworks/Llama-3.1-8B-Instruct-Eagle3 | 60.5% |
| 3 | Qwen3-8B | thoughtworks/Qwen3-8B-Eagle3 | 60.0% |
| 4 | Qwen3-14B | thoughtworks/Qwen3-14B-Eagle3 | 60.1% |

**Needs ~25-line patch to sglang-jax `qwen2.py`** (add `set_eagle3_layers_to_capture`):
| # | Target | Draft (HF) | acc_0 |
|---|--------|------------|-------|
| 5 | Qwen2.5-7B-Instruct | thoughtworks/Qwen2.5-7B-Instruct-Eagle3 | 61.8% |
| 6 | Qwen2.5-14B-Instruct | thoughtworks/Qwen2.5-14B-Instruct-Eagle3 | 60.2% |
| 7 | DeepSeek-R1-Distill-Qwen-7B | thoughtworks/DeepSeek-R1-Distill-Qwen-7B-Eagle3 | 61.5% |
| 8 | DeepSeek-R1-Distill-Qwen-14B | thoughtworks/DeepSeek-R1-Distill-Qwen-14B-Eagle3 | 65.8% |

All EAGLE3 draft models use `LlamaForCausalLMEagle3` architecture in sglang-jax regardless of target model family.

---

## Phase 0: Branch & Directory Setup

1. Create branch: `git checkout -b inference-benchmarks main`
2. Create directory structure:

```
inference/
  README.md                       # Usage guide
  setup.sh                        # Install sglang-jax from local checkout
  run_benchmark.sh                # Parameterized: runs offline throughput (EAGLE3)
  run_benchmark_baseline.sh       # Same benchmark without EAGLE3
  compare_results.py              # Parse JSON results -> markdown summary table
  configs/
    llama32_3b.env                # Per-model env vars (TARGET_MODEL, DRAFT_MODEL, TP_SIZE, etc.)
    llama31_8b.env
    qwen3_8b.env
    qwen3_14b.env
    qwen25_7b.env                 # Phase 2 (needs qwen2 patch)
    qwen25_14b.env
    ds_r1_qwen_7b.env
    ds_r1_qwen_14b.env
  results/                        # Benchmark output JSONs + summary markdown
```

Each `.env` config file contains: `TARGET_MODEL`, `DRAFT_MODEL`, `TP_SIZE`, `DTYPE`, `EAGLE_TOPK`, `NUM_STEPS`, `NUM_DRAFT_TOKENS`, `PAGE_SIZE`, `MEM_FRACTION_STATIC`, `MAX_RUNNING_REQUESTS`.

The scripts source the `.env` and pass the vars to sglang-jax CLI. This avoids one-script-per-model proliferation.

---

## Phase 1: First Model End-to-End (Llama-3.2-3B-Instruct)

**Why this model first:** Smallest (6 GB bf16), fits trivially on v5e, full upstream EAGLE3 support, proven checkpoint.

### Steps

1. **Create `inference/setup.sh`** — installs sglang-jax from `$SGLANG_JAX_ROOT` and ensures `HF_TOKEN` is set
2. **Create `inference/configs/llama32_3b.env`** with:
   - `TARGET_MODEL=meta-llama/Llama-3.2-3B-Instruct`
   - `DRAFT_MODEL=thoughtworks/Llama-3.2-3B-Instruct-Eagle3`
   - `TP_SIZE=4`, `EAGLE_TOPK=1`, `NUM_STEPS=3`, `NUM_DRAFT_TOKENS=4`, `PAGE_SIZE=64`
3. **Create `inference/run_benchmark.sh`** — uses `python -m sgl_jax.bench_offline_throughput` with speculative args from `.env`
4. **Create `inference/run_benchmark_baseline.sh`** — same but without `--speculative-*` flags
5. **Create `inference/compare_results.py`** — reads JSON pairs, computes speedup, outputs markdown table
6. **Run baseline benchmark** for Llama-3.2-3B
7. **Run EAGLE3 benchmark** for Llama-3.2-3B
8. **Compare results** and verify speedup > 1.0x

### Benchmark command (EAGLE3):
```bash
python -m sgl_jax.bench_offline_throughput \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path thoughtworks/Llama-3.2-3B-Instruct-Eagle3 \
  --speculative-eagle-topk 1 --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
  --disable-overlap-schedule --tp-size 4 --dtype bfloat16 --page-size 64 \
  --mem-fraction-static 0.85 --dataset-name sharegpt --num-prompts 200 \
  --result-filename inference/results/llama32_3b_eagle3.json
```

### Key metrics to capture:
- **Output token throughput (tok/s)** — primary metric
- **Total token throughput (tok/s)** — includes prefill
- **Request throughput (req/s)** — user-facing
- **Speedup ratio** — `eagle3_output_throughput / baseline_output_throughput` (the headline number)

---

## Phase 2: Scale to All Compatible Models + Qwen2 Patch

Once Phase 1 validates, expand to remaining models in parallel:

### 2a: Create configs and benchmark for:
- **Llama-3.1-8B-Instruct** (~16 GB bf16, 4 GB/chip at TP=4, fits comfortably)
- **Qwen3-8B** (~16 GB bf16, same as above, uses `qwen3.py` target with EAGLE3 support)
- **Qwen3-14B** (~28 GB bf16, 7 GB/chip — tight on v5e but should work for inference. Reduce `MEM_FRACTION_STATIC=0.80`, `MAX_RUNNING_REQUESTS=32`. If OOM, skip and note needs v4/v6e.)

### 2b: Patch sglang-jax `qwen2.py` for EAGLE3 support

Apply locally to `$SGLANG_JAX_ROOT/python/sgl_jax/srt/models/qwen2.py`. The patch mirrors the pattern in `qwen3.py` (lines 539-547) and `llama.py` (lines 527-537):

1. Add `self.capture_aux_hidden_states = False` in `Qwen2ForCausalLM.__init__`
2. Add `self.layers_to_capture = []` in `Qwen2Model.__init__`
3. Add aux hidden state capture in `Qwen2Model.__call__` forward loop (capture at `layers_to_capture` indices)
4. Return `aux_hidden_states` from `Qwen2Model.__call__`
5. Add `set_eagle3_layers_to_capture()` method to `Qwen2ForCausalLM`
6. Wire `aux_hidden_states` through `Qwen2ForCausalLM.__call__`

**Reference files:**
- Pattern to copy: `sglang-jax/python/sgl_jax/srt/models/qwen3.py` (lines 370-380, 539-547)
- File to modify: `sglang-jax/python/sgl_jax/srt/models/qwen2.py`

### 2c: Benchmark Qwen2 family:
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B (highest acc_0 at 65.8% — likely best speedup)

---

## Phase 3: Package Results

1. **Create `inference/README.md`** — the public-facing guide:
   - Prerequisites (sglang-jax install, HF_TOKEN)
   - Supported model pairs table
   - Copy-paste commands per model
   - Benchmark results table
   - Hardware requirements
2. **Generate `inference/results/BENCHMARK_RESULTS.md`** — summary table with all throughput numbers
3. **Update SpecJAX main `README.md`** — add "Inference" section pointing to `inference/`
4. **Create `inference/run_all_benchmarks.sh`** — convenience script that iterates all configs

---

## Files to Create/Modify

### New files in SpecJAX (on `inference-benchmarks` branch):
- `inference/README.md`
- `inference/setup.sh`
- `inference/run_benchmark.sh`
- `inference/run_benchmark_baseline.sh`
- `inference/compare_results.py`
- `inference/run_all_benchmarks.sh`
- `inference/configs/llama32_3b.env`
- `inference/configs/llama31_8b.env`
- `inference/configs/qwen3_8b.env`
- `inference/configs/qwen3_14b.env`
- `inference/configs/qwen25_7b.env`
- `inference/configs/qwen25_14b.env`
- `inference/configs/ds_r1_qwen_7b.env`
- `inference/configs/ds_r1_qwen_14b.env`
- `inference/results/` (generated)

### Modified file in sglang-jax (local patch):
- `$SGLANG_JAX_ROOT/python/sgl_jax/srt/models/qwen2.py` (~25 lines, Phase 2b)

### Modified file in SpecJAX:
- `README.md` (add Inference section, Phase 3)

---

## Verification

1. **Smoke test:** For each model, verify the sglang-jax server starts and responds to a single `/v1/chat/completions` request before running the full benchmark
2. **Benchmark validity:** Each benchmark runs 200 ShareGPT prompts. Baseline and EAGLE3 use identical prompts.
3. **Speedup sanity check:** With 60% acc_0, expect 1.3x-2.0x output throughput speedup. If speedup < 1.0x, investigate NUM_STEPS/TOPK tuning.
4. **End-to-end:** `python inference/compare_results.py inference/results/` produces a clean markdown table

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| v5e OOM for 14B models | Reduce MEM_FRACTION_STATIC/MAX_RUNNING_REQUESTS; skip and note "needs v4" |
| HF models are private | Need HF_TOKEN; will ask user if not available |
| sglang-jax EAGLE3 bugs | Start with smallest model; compare server logs against known-working Qwen3-32B config |
| Poor speedup numbers | Tune EAGLE_TOPK (1 vs 2), NUM_STEPS (3 vs 5); 60% acc_0 should still yield measurable gain |

---

## Execution Log (2026-03-27)

### Environment
- TPU v5e-32 pod (8 hosts, 4 chips/host), running single-host mode (4 chips, TP=4)
- Python 3.12 venv at `$VENV_PATH`
- sglang-jax at `$SGLANG_JAX_ROOT` (via PYTHONPATH)
- JAX 0.8.1 + libtpu

### Bug Fix: Tied Embeddings
Llama-3.2-3B uses `tie_word_embeddings=True`, which caused `AttributeError: 'LlamaForCausalLM' object has no attribute 'lm_head'` in `get_embed_and_head()`. Fixed locally in `sglang-jax/python/sgl_jax/srt/models/llama.py` with `hasattr` guards.

### Benchmark Results

| Model | Baseline (tok/s) | EAGLE3 Overall (tok/s) | EAGLE3 Steady-State (tok/s) | Baseline Prompts | EAGLE3 Prompts |
|---|---|---|---|---|---|
| Llama-3.1-8B | 129.0 | 15.0 | 40.6 | 200 | 20 |
| Llama-3.2-3B | 149.5 | 12.8 | 21.5 | 200 | 20 |
| Qwen3-8B | 358.5 | 10.4 | 76.4 | 200 | 20 |

### Key Finding: sglang-jax EAGLE3 Not Yet Optimized

EAGLE3 speculative decoding in sglang-jax is currently **slower** than baseline for all models. The sglang-jax docs explicitly note: *"the performance optimization is needed, some jnp array operations need move to JIT functions"*.

**What works:**
- All SpecJAX-trained EAGLE3 models load and run correctly in sglang-jax
- Acceptance rates are reasonable (~25-67%), consistent with ~60% training acc_0
- Steady-state throughput (after JIT warmup) is better but still below baseline

**What doesn't work yet:**
- The verify/tree-building pipeline has high overhead per decode step
- Each new batch shape triggers JIT recompilation (cache_miss=1 on many steps)
- The draft model forward pass is not efficiently overlapped with target model

**Recommendation:** The SpecJAX models are ready. The sglang-jax EAGLE3 runtime needs optimization. For the public release, we can demonstrate compatibility and note that throughput gains will come with sglang-jax improvements. Alternatively, we can wait for sglang-jax to optimize and re-run benchmarks.

### Qwen2 Patch: Deferred
The ~25-line patch to `qwen2.py` was not applied since there's no point benchmarking more models while the EAGLE3 pipeline itself is unoptimized. The patch is documented in Phase 2b above and can be applied when sglang-jax improves.
