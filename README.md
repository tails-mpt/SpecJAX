# SpecJAX

Train speculative decoding draft models in JAX on TPUs.

SpecJAX is a JAX/TPU-native alternative to [SpecForge](https://github.com/sgl-project/SpecForge) for training [EAGLE3](https://arxiv.org/abs/2503.01840) draft models. While SpecForge targets GPU clusters with PyTorch, SpecJAX runs natively on Google Cloud TPU pods using JAX for efficient multi-host training.

## How It Works

EAGLE3 trains a lightweight draft model to predict the next token distribution of a larger frozen target model:

1. **Target forward** (frozen): run the full target model to get logits and intermediate hidden states from 3 auxiliary layers.
2. **Draft forward** (trainable): the EAGLE3 draft head projects the multi-layer features, runs one transformer layer, and predicts draft logits.
3. **KL loss**: minimize the KL divergence between the draft and target softmax distributions.
4. **TTT (optional)**: multi-step rollout with geometric loss weighting (0.8^k) for better multi-token acceptance.

## Supported Models

| Target Model | Params | Status | Hardware | HuggingFace |
|-------------|--------|--------|----------|-------------|
| Llama-3.2-3B-Instruct | 3.2B | Trained | TPU v6e-4 | [thoughtworks/Llama-3.2-3B-Instruct-Eagle3](https://huggingface.co/thoughtworks/Llama-3.2-3B-Instruct-Eagle3) |
| Llama-3.1-8B-Instruct | 8B | Trained | TPU v4-32 | [thoughtworks/Llama-3.1-8B-Instruct-Eagle3](https://huggingface.co/thoughtworks/Llama-3.1-8B-Instruct-Eagle3) |
| Qwen2.5-7B-Instruct | 7.1B | Trained | TPU v4-32 | [thoughtworks/Qwen2.5-7B-Instruct-Eagle3](https://huggingface.co/thoughtworks/Qwen2.5-7B-Instruct-Eagle3) |
| Qwen2.5-14B-Instruct | 14B | Trained | TPU v4-32 | [thoughtworks/Qwen2.5-14B-Instruct-Eagle3](https://huggingface.co/thoughtworks/Qwen2.5-14B-Instruct-Eagle3) |
| DeepSeek-R1-Distill-Qwen-7B | 7.6B | Trained | TPU v5e-32 | [thoughtworks/DeepSeek-R1-Distill-Qwen-7B-Eagle3](https://huggingface.co/thoughtworks/DeepSeek-R1-Distill-Qwen-7B-Eagle3) |
| DeepSeek-R1-Distill-Qwen-14B | 14B | Trained | TPU v4-32 | [thoughtworks/DeepSeek-R1-Distill-Qwen-14B-Eagle3](https://huggingface.co/thoughtworks/DeepSeek-R1-Distill-Qwen-14B-Eagle3) |
| Qwen3-8B | 8B | Trained | TPU v4-32 | [thoughtworks/Qwen3-8B-Eagle3](https://huggingface.co/thoughtworks/Qwen3-8B-Eagle3) |
| Qwen3-14B | 14B | Trained | TPU v4-32 | [thoughtworks/Qwen3-14B-Eagle3](https://huggingface.co/thoughtworks/Qwen3-14B-Eagle3) |
| GLM-4.7-Flash | 15.6B | Trained | TPU v6e-4 | — |
| Qwen3-Coder-Next (MoE) | 80B | In progress | TPU v4-64 | — |

## Project Structure

```
specjax/
├── specjax/                  # Core library
│   ├── env.py               # TPU environment configuration
│   ├── ops/                  # Shared pure-function building blocks
│   │   ├── norm.py           # RMSNorm
│   │   ├── rope.py           # Rotary position embeddings
│   │   ├── moe.py            # MoE routing + expert dispatch
│   │   ├── fp8.py            # FP8 dequantization (block, channel, JIT)
│   │   └── loading.py        # Safetensors shard loading, expert stacking
│   ├── models/
│   │   ├── draft/eagle3.py   # EAGLE3 draft model (trainable)
│   │   ├── target/           # Frozen target model implementations
│   │   └── sharding.py       # SPMD mesh + sharding utilities
│   ├── training/
│   │   ├── vocab.py          # Draft-target vocabulary mapping
│   │   └── optimizer.py      # Cosine warmup + AdamW
│   ├── data/dataset.py       # ShareGPT dataset with bucket batching
│   ├── train.py              # Training entry point
│   └── eval.py               # Evaluation entry point
├── configs/                  # Per-model training configurations
├── scripts/                  # Operational scripts
│   ├── setup/                # Environment and dataset preparation
│   ├── download/             # Model weight download
│   ├── preflight/            # Pre-training validation checks
│   ├── run/                  # Experiment launch scripts
│   └── monitoring/           # TPU metrics collection
└── docs/                     # Documentation and analysis
    ├── training/             # Step-by-step training guides per model
    └── hardware/             # TPU memory analysis and hardware constraints
```

## Key Findings

- **MoE + EAGLE3 on GPU**: 0.63x throughput despite 35% acceptance rate. Root cause: MoE verification cost is 3x dense models.
- **PyTorch/XLA on TPU**: Multiple blocking issues (mmap corruption, ICI crashes) motivated the pure JAX rewrite.
- **GLM-5-FP8 (744B) on v4-32**: Does not fit — 744B params need >512 GB HBM. See [hardware analysis](docs/hardware/glm5-v4-32-analysis.md).
- **GLM-4.7-Flash on JAX/TPU**: Working EAGLE3 training pipeline. See [training guide](docs/training/glm-flash-eagle3.md).

## Quick Start

```bash
# 1. Install the package
pip install -e .

# 2. Set up environment on TPU
bash scripts/setup/setup_jax_env.sh

# 3. Run preflight checks
bash scripts/preflight/preflight_jax.sh

# 4. Launch training
python -m specjax.train \
  --target-model-path /path/to/GLM-4.7-Flash \
  --target-model-type glm_flash \
  --data-path data/sharegpt.jsonl \
  --output-dir /path/to/checkpoints \
  --exp-name my-experiment

# 5. Evaluate
python -m specjax.eval \
  --target-model-path /path/to/GLM-4.7-Flash \
  --draft-checkpoint /path/to/checkpoints/final \
  --eval-data data/mt_bench.jsonl
```

Or use a config file with the experiment launch scripts:
```bash
python -m specjax.train --config configs/glm_flash.json \
  --target-model-path /path/to/GLM-4.7-Flash \
  --data-path data/sharegpt.jsonl \
  --output-dir /path/to/checkpoints
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Quick smoke test (5 optimizer steps)
python -m specjax.train \
  --target-model-path /path/to/GLM-4.7-Flash \
  --data-path data/sharegpt.jsonl \
  --output-dir /tmp/test \
  --max-steps 5
```

## References

- [EAGLE3: Scaling up EAGLE with Multi-Token Prediction](https://arxiv.org/abs/2503.01840)
- [SpecForge (PyTorch/GPU reference)](https://github.com/sgl-project/SpecForge)
- [SGLang](https://github.com/sgl-project/sglang) — inference serving with speculative decoding

## License

MIT
