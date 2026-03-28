# SpecJAX EAGLE3 Inference Benchmarks

Hardware: TPU v5e-32 (single host, 4 chips, TP=4)
Inference runtime: sglang-jax (local build)
Dataset: ShareGPT

| Model | Baseline (tok/s) | EAGLE3 (tok/s) | Steady-State EAGLE3 (tok/s) | Baseline Prompts | EAGLE3 Prompts |
|-------|-----------------|----------------|---------------------------|-----------------|----------------|
| llama31_8b | 129.0 | 15.0 | 40.6 | 200 | 20 |
| llama32_3b | 149.5 | 12.8 | 21.5 | 200 | 20 |
| qwen3_8b | 358.5 | 10.4 | 76.4 | 200 | 20 |

**Note:** EAGLE3 speculative decoding in sglang-jax is currently unoptimized.
The sglang-jax docs note: *"the performance optimization is needed, some jnp array
operations need move to JIT functions"*. The models load and produce correct output
with reasonable acceptance rates (~25-67%), but the overhead of the unoptimized
verify/tree-building pipeline currently outweighs the savings from speculative tokens.
Steady-state throughput (`last_gen_throughput`) shows the gap is smaller once JIT
compilation caches are warm. Once sglang-jax optimizes their EAGLE3 pipeline,
these SpecJAX-trained draft models will deliver the expected speedup.
