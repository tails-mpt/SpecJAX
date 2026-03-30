# SpecJAX Inference with sglang-jax

Run EAGLE3 draft models trained by SpecJAX on [sglang-jax](https://github.com/sgl-project/sglang-jax) for inference on TPU.

> **Recommended**: Use our fork [tails-mpt/sglang-jax](https://github.com/tails-mpt/sglang-jax) which includes EAGLE3 support for all SpecJAX-trained models out of the box (Qwen2/2.5 patch + Llama tied-embeddings fix).

## Supported Model Pairs

| Target Model | Draft Model (HuggingFace) | Training acc_0 | sglang-jax Support |
|---|---|---|---|
| meta-llama/Llama-3.2-3B-Instruct | thoughtworks/Llama-3.2-3B-Instruct-Eagle3 | 60.6% | Native |
| meta-llama/Llama-3.1-8B-Instruct | thoughtworks/Llama-3.1-8B-Instruct-Eagle3 | 60.5% | Native |
| Qwen/Qwen3-8B | thoughtworks/Qwen3-8B-Eagle3 | 60.0% | Native |
| Qwen/Qwen3-14B | thoughtworks/Qwen3-14B-Eagle3 | 60.1% | Native |
| Qwen/Qwen2.5-7B-Instruct | thoughtworks/Qwen2.5-7B-Instruct-Eagle3 | 61.8% | [Fork](https://github.com/tails-mpt/sglang-jax) or patch upstream |
| Qwen/Qwen2.5-14B-Instruct | thoughtworks/Qwen2.5-14B-Instruct-Eagle3 | 60.2% | [Fork](https://github.com/tails-mpt/sglang-jax) or patch upstream |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | thoughtworks/DeepSeek-R1-Distill-Qwen-7B-Eagle3 | 61.5% | [Fork](https://github.com/tails-mpt/sglang-jax) or patch upstream |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | thoughtworks/DeepSeek-R1-Distill-Qwen-14B-Eagle3 | 65.8% | [Fork](https://github.com/tails-mpt/sglang-jax) or patch upstream |

## Quick Start

### 1. Setup

```bash
# Prerequisites: Python 3.12, sglang-jax source
bash inference/setup.sh
```

### 2. Run a Benchmark

```bash
# Baseline (no speculative decoding)
bash inference/run_benchmark_baseline.sh inference/configs/qwen3_8b.env

# With EAGLE3 speculative decoding
bash inference/run_benchmark.sh inference/configs/qwen3_8b.env

# Compare results
python3.12 inference/compare_results.py inference/results/
```

### 3. Serve a Model

```bash
source inference/tpu_env.sh
source inference/configs/qwen3_8b.env

$PYTHON -m sgl_jax.launch_server \
    --model-path "$TARGET_MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-eagle-topk 1 \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --disable-overlap-schedule \
    --tp-size 4 --dtype bfloat16 --page-size 64 \
    --port 30000 --host 0.0.0.0
```

Then query with any OpenAI-compatible client:
```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Hardware Requirements

| Model Size | Minimum TPU | TP Size | Notes |
|---|---|---|---|
| 3B | v5e-4 (4 chips) | 4 | Fits easily |
| 7-8B | v5e-4 (4 chips) | 4 | ~16 GB bf16 |
| 14B | v4-8 or v6e-4 (4 chips) | 4 | ~28 GB bf16, tight on v5e |

## TPU Multi-Host Setup

On multi-host TPU pods (e.g., v5e-32), run benchmarks in single-host mode by sourcing `tpu_env.sh`:

```bash
source inference/tpu_env.sh   # Sets TPU_CHIPS_PER_PROCESS_BOUNDS etc.
```

This restricts JAX to the 4 local TPU chips without requiring coordination with other hosts.

## Benchmark Results

See [results/BENCHMARK_RESULTS.md](results/BENCHMARK_RESULTS.md) for full results.

**Current status (March 2026):** The EAGLE3 speculative decoding pipeline in sglang-jax is functional but not yet performance-optimized. The sglang-jax documentation notes: *"the performance optimization is needed, some jnp array operations need to move to JIT functions."*

Our SpecJAX-trained draft models:
- Load correctly into sglang-jax via `--speculative-algorithm EAGLE3`
- Produce correct output with reasonable acceptance rates
- Are ready to deliver throughput gains once sglang-jax optimizes the EAGLE3 pipeline

## Directory Structure

```
inference/
  README.md                    # This file
  setup.sh                     # Install sglang-jax dependencies
  tpu_env.sh                   # TPU single-host environment setup
  run_benchmark.sh             # Benchmark with EAGLE3
  run_benchmark_baseline.sh    # Benchmark without EAGLE3
  run_all_benchmarks.sh        # Run all model benchmarks
  compare_results.py           # Compare and summarize results
  configs/                     # Per-model configuration files
    llama32_3b.env
    llama31_8b.env
    qwen3_8b.env
    qwen3_14b.env
    qwen25_7b.env              # Needs qwen2 patch
    qwen25_14b.env             # Needs qwen2 patch
    ds_r1_qwen_7b.env          # Needs qwen2 patch
    ds_r1_qwen_14b.env         # Needs qwen2 patch
  results/                     # Benchmark output (JSON + summary)
```

## Qwen2/2.5 EAGLE3 Support

Upstream sglang-jax natively supports EAGLE3 for LLaMA and Qwen3 targets. For Qwen2/2.5 targets (including DeepSeek-R1-Distill-Qwen), a small patch is needed to add `set_eagle3_layers_to_capture()` to `qwen2.py`.

**Option 1 (recommended):** Use our fork which includes this patch:
```bash
git clone https://github.com/tails-mpt/sglang-jax.git
cd sglang-jax && pip install -e .
```

**Option 2:** Apply the patch to upstream sglang-jax yourself. The patch follows the exact pattern in `qwen3.py`. See the [integration plan](../docs/sglang-jax-inference-integration-plan.md) for details.
