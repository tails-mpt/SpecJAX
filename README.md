# SpecJAX: Train EAGLE3 Draft Models on TPU

SpecJAX is an open-source JAX/TPU framework for training [EAGLE3](https://arxiv.org/abs/2503.01840) speculative decoding draft models natively on Google Cloud TPUs — no PyTorch, no CUDA, no XLA recompilation surprises.

It is a TPU-native alternative to [SpecForge](https://github.com/sgl-project/SpecForge) (PyTorch/GPU). We've used it to train **9 production-grade EAGLE3 draft heads** across Llama, Qwen, and DeepSeek architectures, all publicly available on HuggingFace under the [Thoughtworks](https://huggingface.co/thoughtworks) organization.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-TPU-orange.svg)](https://github.com/google/jax)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-9%20released-yellow)](https://huggingface.co/thoughtworks)
[![arXiv](https://img.shields.io/badge/arXiv-2503.01840-b31b1b.svg)](https://arxiv.org/abs/2503.01840)

## Why SpecJAX?

If you have TPU access and want to train EAGLE3 draft heads, SpecForge doesn't run on TPUs — it's CUDA-only. SpecJAX fills that gap: the same EAGLE3 algorithm, rebuilt from the ground up in pure JAX with static-shape einsum so it compiles once and stays compiled. No PyTorch/XLA, no libtpu mmap tricks.

## Key Features

- **Pure JAX** — no Flax, no `nn.Module`, no mutable state. All functions are stateless pure functions; JIT compilation and SPMD sharding work reliably.
- **Static-shape einsum** — MoE expert dispatch processes all experts simultaneously, eliminating data-dependent shapes that cause XLA recompilation.
- **SPMD sharding** — 2D `(dp, tp)` mesh via `jax.sharding.NamedSharding`. Scales from TP=4 on a single v4-32 host to TP=8 across two v6e-8 hosts for 32B+ models.
- **Flat param dicts** — weights loaded directly from safetensors as `{str: jnp.ndarray}`, zero framework overhead.
- **TTT training** — multi-step speculative rollout with geometric loss weighting (0.8^k) for better multi-token acceptance at inference time.
- **Multi-architecture support** — standard GQA, GQA+MoE, MLA+MoE, and FP8 quantized targets all handled through a unified target model registry.

## Trained Models

All checkpoints are publicly available and compatible with [SGLang](https://github.com/sgl-project/sglang) (GPU) and [sglang-jax](https://github.com/tails-mpt/sglang-jax) (TPU).

| Target Model | Params | Hardware | acc_0 | HuggingFace |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 3.2B | TPU v6e-4 | 60.6% | [thoughtworks/Llama-3.2-3B-Instruct-Eagle3](https://huggingface.co/thoughtworks/Llama-3.2-3B-Instruct-Eagle3) |
| Llama-3.1-8B-Instruct | 8B | TPU v4-32 | 60.5% | [thoughtworks/Llama-3.1-8B-Instruct-Eagle3](https://huggingface.co/thoughtworks/Llama-3.1-8B-Instruct-Eagle3) |
| Qwen2.5-7B-Instruct | 7.1B | TPU v4-32 | 61.8% | [thoughtworks/Qwen2.5-7B-Instruct-Eagle3](https://huggingface.co/thoughtworks/Qwen2.5-7B-Instruct-Eagle3) |
| Qwen2.5-14B-Instruct | 14B | TPU v4-32 | 60.2% | [thoughtworks/Qwen2.5-14B-Instruct-Eagle3](https://huggingface.co/thoughtworks/Qwen2.5-14B-Instruct-Eagle3) |
| DeepSeek-R1-Distill-Qwen-7B | 7.6B | TPU v5e-32 | 61.5% | [thoughtworks/DeepSeek-R1-Distill-Qwen-7B-Eagle3](https://huggingface.co/thoughtworks/DeepSeek-R1-Distill-Qwen-7B-Eagle3) |
| DeepSeek-R1-Distill-Qwen-14B | 14B | TPU v4-32 | 65.8% | [thoughtworks/DeepSeek-R1-Distill-Qwen-14B-Eagle3](https://huggingface.co/thoughtworks/DeepSeek-R1-Distill-Qwen-14B-Eagle3) |
| Qwen3-8B | 8B | TPU v4-32 | 60.0% | [thoughtworks/Qwen3-8B-Eagle3](https://huggingface.co/thoughtworks/Qwen3-8B-Eagle3) |
| Qwen3-14B | 14B | TPU v4-32 | 60.1% | [thoughtworks/Qwen3-14B-Eagle3](https://huggingface.co/thoughtworks/Qwen3-14B-Eagle3) |
| Qwen3-32B | 32B | TPU v6e-16 (TP=8) | — | [thoughtworks/Qwen3-32B-Eagle3](https://huggingface.co/thoughtworks/Qwen3-32B-Eagle3) |

`acc_0` is the first-token acceptance rate on ShareGPT (temperature=0) measured during training evaluation.

## Inference

Draft models are served via standard EAGLE3 speculative decoding pipelines:

**GPU — SGLang (upstream)**
All Llama and Qwen3 models work with [upstream SGLang](https://github.com/sgl-project/sglang) out of the box. Qwen2.5 and DeepSeek-R1-Distill models require a small patch to `qwen2.py` (~25 lines) — see the [inference guide](inference/README.md).

**TPU — sglang-jax**
Use the fork [tails-mpt/sglang-jax](https://github.com/tails-mpt/sglang-jax), which ships with all EAGLE3 patches pre-applied. See [inference/](inference/) for benchmark scripts and per-model configs.

> **Status:** The sglang-jax EAGLE3 pipeline is functional (correct outputs, ~60–66% acceptance rates) but throughput gains are pending upstream optimization of the verify/tree-building path. See the [inference README](inference/README.md) for current numbers and known issues.

## Getting Started

```bash
# Install
pip install -e .
bash scripts/setup/setup_jax_env.sh

# Train a draft head
python -m specjax.train \
  --target-model-path /path/to/Qwen3-8B \
  --target-model-type qwen3 \
  --data-path data/sharegpt.jsonl \
  --output-dir /path/to/checkpoints

# Evaluate
python -m specjax.eval \
  --target-model-path /path/to/Qwen3-8B \
  --draft-checkpoint /path/to/checkpoints/final \
  --eval-data data/mt_bench.jsonl
```

See the [inference guide](inference/README.md) for serving configs and benchmark scripts.

## Architecture

SpecJAX uses a minimal stack: safetensors → flat param dict → pure JAX forward functions → SPMD-sharded `jax.jit`. There are no framework-managed modules; the target model is always frozen and loaded once, and only the small draft head accumulates gradients.

The EAGLE3 draft head is a single transformer block that takes the target model's last hidden state (plus an embed of the current token) and predicts the next token distribution. Training uses teacher-forcing rollouts of length 1–7 with geometric loss weighting.

For a deeper dive, see [`specjax/models/draft/eagle3.py`](specjax/models/draft/eagle3.py) and the [EAGLE3 paper](https://arxiv.org/abs/2503.01840).

## References

- [EAGLE3: Scaling up EAGLE with Multi-Token Prediction](https://arxiv.org/abs/2503.01840) — the algorithm we implement
- [SpecForge](https://github.com/sgl-project/SpecForge) — PyTorch/GPU reference implementation by the SGLang team
- [SGLang](https://github.com/sgl-project/sglang) — GPU inference serving with speculative decoding
- [sglang-jax](https://github.com/tails-mpt/sglang-jax) — TPU inference serving with EAGLE3 support

## Acknowledgments

SpecJAX builds on the work of the [EAGLE team](https://arxiv.org/abs/2503.01840) and the [SGLang / SpecForge team](https://github.com/sgl-project/SpecForge). TPU resources provided by Google Cloud TPU Research Cloud.

## License

MIT
