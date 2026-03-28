# GLM Large Model Eagle3 Training — Hardware & Environment Findings

**Date**: 2026-03-24
**Context**: Setting up Eagle3 JAX training for large GLM models on current TPU v4-32 instance.

---

## Current Hardware

| Property | Value | How determined |
|----------|-------|----------------|
| TPU type | **v4-32** | GCE metadata `accelerator-type` |
| Total chips | **16** (v4-32 = 32 TensorCores = 16 chips) | 4 worker endpoints × 4 local `/dev/accel*` devices |
| HBM per chip | **32 GB** (TPU v4 spec) | — |
| **Total HBM** | **512 GB** | 16 × 32GB |
| Workers | 4 hosts | `worker-network-endpoints` has 4 IPs: <worker-ips> |
| Chips per worker | 4 local (`/dev/accel0..3`) | `ls /dev/accel*` |
| Host RAM | 400 GB | `free -h` |
| CPU | AMD EPYC 7B12 | `/proc/cpuinfo` |
| Worker ID | 0 (primary) | metadata `agent-worker-number` |

### Multi-host JAX setup

v4-32 is a **multi-host** TPU slice. JAX requires coordinated initialization across all 4 workers. Running `jax.devices()` on a single worker fails with:

```
TPU initialization failed: Failed to establish SliceBuilder grpc channel to <worker-0-ip>:8471
```

This means training scripts must be launched on **all workers simultaneously** using `JAX_NUM_PROCESSES`, coordinator address, etc. The existing single-host training scripts (`run_exp_jax_*.sh`) set `TPU_PROCESS_BOUNDS=1,1,1` which collapses to single-host mode and won't work here.

---

## Memory Problem: GLM-5-FP8 doesn't fit

| Model | Format | Size | Available HBM |
|-------|--------|------|---------------|
| GLM-5-FP8 | FP8 (e4m3) | **~744 GB** | 512 GB |
| GLM-5-FP8 | bf16 (dequantized) | **~1.5 TB** | 512 GB |
| GLM-4.7-Flash | bf16 | ~18 GB | 512 GB (fits easily) |

**GLM-5-FP8 does not fit on v4-32 in any format.** Even keeping weights in FP8 (1 byte per param), 744B params = 744 GB > 512 GB HBM.

### Minimum hardware requirements

| Strategy | Min TPU | Total HBM | Notes |
|----------|---------|-----------|-------|
| FP8 on-device | v4-64 (32 chips) | 1024 GB | ~23 GB/chip for weights, leaves 9 GB for activations |
| bf16 dequantized | v4-128 (64 chips) | 2048 GB | Comfortable headroom |
| FP8 + host offload | v4-32 (current) | 512 GB HBM + 400 GB host | Possible but slow, complex code |

### Host offloading option (not implemented)

Could keep weights in host RAM and stream layer-by-layer to TPU during forward pass:
- Each of 78 layers: ~9.5B params (MoE experts dominate) = ~9.5 GB in FP8
- Stream one layer's weights to TPU, dequantize, compute, discard
- Pro: Works on current hardware
- Con: Massive PCIe bandwidth bottleneck, requires significant code changes to `glm5_jax.py`

---

## What was accomplished

### Code (ready for when hardware is available)

All Eagle3 training code for GLM-5-FP8 has been written and syntax-checked:

1. **`models/glm5_jax.py`** — Full GLM-5-FP8 target model in pure JAX:
   - `GLM5Config` with all GLM-5 architecture constants
   - FP8 block-wise dequantization (`_dequant_fp8_block`)
   - Weight loading with FP8 → bf16 conversion, DSA/MTP weight skipping
   - Full forward pass (MLA, 256-expert MoE, sigmoid router, 78 decoder layers)
   - Aux layer indices {1, 38, 74} for Eagle3 multi-layer fusion

2. **`models/eagle3_jax.py`** — Added `eagle3_config_for_glm5()` factory:
   - hidden_size=6144, 32 heads, 8 KV heads, head_dim=192, intermediate=16384
   - `load_eagle3_params` now accepts `default_config` for from-scratch init

3. **`models/sharding.py`** — `make_mesh(tp_size=None)` now accepts explicit TP size

4. **`train_eagle3_jax.py`** — Added `--target-model-type glm5` and `--tp-size` flags

5. **Shell scripts**: `download_models_glm5.sh`, `setup_glm5_env.sh`, `preflight_glm5.sh`, `run_exp_glm5_a.sh`

### Model download (in progress)

GLM-5-FP8 download was started and is at ~406 GB (80/142 safetensors shards) as of writing. Download target: `/path/to/models/GLM-5-FP8/`.

### Environment

- Venv at `/path/to/workspace/.venv-train-jax` is set up with `ml_dtypes` for FP8 support
- JAX cannot initialize TPU on this worker alone (multi-host coordination needed)

---

## Design decisions documented

1. **DSA skipped**: GLM-5's Dynamic Sparse Attention selects top-2048 positions. With `max_length <= 1024`, all positions are selected → equivalent to full causal attention. Indexer weights are ignored.

2. **MTP skipped**: `num_nextn_predict_layers: 1` is not needed for Eagle3 training.

3. **Sigmoid router**: GLM-5 uses sigmoid scoring (same as GLM-Flash — already implemented).

4. **Same vocab**: GLM-5 and GLM-Flash share vocab_size=154,880 — no tokenizer changes needed.

5. **Eagle3 scaling**: Draft head dimensions scale proportionally with target hidden_size (6144 vs 2048). GQA ratio kept at 4:1.

---

## GLM-4.7-FP8: Fits on v4-32

After discovering GLM-5-FP8 doesn't fit, we pivoted to `zai-org/GLM-4.7-FP8`:

| Model | Params | FP8 Size | Fits v4-32? |
|-------|--------|----------|-------------|
| GLM-5-FP8 | ~744B | ~744 GB | No (512 GB available) |
| **GLM-4.7-FP8** | **~367B** | **~367 GB** | **Yes (+145 GB headroom)** |

### Architecture differences from GLM-5-FP8

- **Attention**: Standard GQA (96h/8kv, head_dim=128) — NOT MLA
- **QK norm**: RMSNorm per-head before RoPE (new)
- **Partial RoPE**: Standard layout, first 50% of head_dim (64/128 dims)
- **FP8 format**: Channel-wise (compressed-tensors) — NOT block-wise
- **Experts**: 160 (not 256)
- **Vocab**: 151,552 (not 154,880)

### Code created for GLM-4.7-FP8

1. **`models/glm47_fp8_jax.py`** — Target model with GQA attention, channel-wise FP8 dequant
2. **`eagle3_config_for_glm47_fp8()`** — Draft head config (H=5120, 40h/10kv/128d)
3. **`worker_start_glm47_fp8.sh`** — Per-worker multi-host launcher
4. **`run_exp_glm47_fp8_a.sh`** — Multi-host orchestrator
5. **`download_models_glm47_fp8.sh`** — Model download script
6. **`preflight_glm47_fp8.sh`** — Pre-flight checks

### Multi-host v4-32 setup

- 4 workers (<worker-ips>), worker 0 = coordinator at <coordinator-ip>
- `JAX_COORDINATOR_ADDRESS=<coordinator-ip>:2222`, `JAX_NUM_PROCESSES=4`
- TP=16 (all chips), no DP — model too large for per-worker replication
- Existing `worker_start_training.sh` pattern adapted

## GLM-4.7-FP8 Training Progress

### Issues solved during bring-up

| # | Issue | Root cause | Fix |
|---|-------|-----------|-----|
| 1 | Host RAM OOM at shard 50/93 | `load_params` accumulated all weights in host RAM (676 GB bf16) before sharding to TPU | Streaming loader: `jax.make_array_from_callback` places each tensor on TPU immediately, host only holds one shard at a time (~4 GB) |
| 2 | TPU HBM OOM at shard 48/93 | `jax.device_put` triggers `broadcast_one_to_all` which temporarily allocates full unsharded array on each chip | Replaced with `jax.make_array_from_callback` (no broadcast) |
| 3 | TPU HBM OOM at shard 88/93 | MoE expert weights in bf16 = 42 GB/chip, exceeds 32 GB HBM budget | Store MoE experts as uint8 (raw FP8 bytes) on TPU with f32 channel-wise scales. JIT dequant via `lax.bitcast_convert_type(uint8→float8_e4m3fn)→f32 * scale→bf16`. Reduces MoE to 21 GB/chip. |
| 4 | Multi-host batch split `IndexError` | `dp_index = jax.process_index()` returns 0-3 in pure-TP mode, but `dp_size=1` → out-of-bounds | Set `dp_index=0` for all processes when no "dp" mesh axis |
| 5 | Checkpoint save `PermissionError` | `/path/to/workspace/checkpoints/` owned by different user | `sudo chown` to fix permissions |
| 6 | Checkpoint save `RuntimeError` (non-addressable devices) | `np.array(v)` on multi-host sharded JAX array fails — can only access local shards | Use `v.addressable_shards[0].data` for checkpoint save (Eagle3 draft params are small/replicated) |
| 7 | Checkpoint save TPU core halt | `process_allgather` crashes TPU during large array gather | Reverted to addressable shards approach (issue #6 fix) |

### HBM budget (final working config)

| Component | Per chip (TP=16) | Notes |
|-----------|-----------------|-------|
| MoE experts (uint8) | 21.0 GB | 89 layers × 10 experts/chip × (gate_up + down) |
| MoE scales (f32) | 0.03 GB | Channel-wise, negligible |
| Attention (bf16) | 1.5 GB | 92 layers × GQA (96h/8kv) |
| Shared experts (bf16) | 0.3 GB | 89 layers × 1 shared expert |
| Embed + lm_head (bf16) | 0.2 GB | 151552 × 5120, sharded |
| **Total weights** | **~23.1 GB** | |
| **Headroom** | **~8.9 GB** | For activations, Eagle3, optimizer |

### Training results (TTT=1 single-step mode)

Config: B=2, grad_accum=16, max_length=512, LR=1e-4, cosine decay, 3 epochs

| Step | Loss | Acc | s/step | Note |
|------|------|-----|--------|------|
| 10 | 9.34 | 1.5% | 42.8 | XLA warmup |
| 50 | 4.28 | 37.0% | 13.8 | |
| 100 | 2.97 | 52.3% | 10.5 | |
| 200 | 2.04 | 66.7% | 8.8 | Checkpoint save |

W&B: [glm4-large-eagle3-experiments](https://wandb.ai/your-wandb-entity/glm4-large-eagle3-experiments)

### TTT=7 does not fit

TTT=7 (7-step unrolled forward pass through 92 MoE layers) OOMs even with B=1, max_length=128. The XLA program compilation requires ~312 MB more than available (~79 MB free). Options:
- Gradient checkpointing (recompute activations during backward)
- Larger TPU slice (v4-64+)
- Host offloading of some layers

### Key environment findings

- **Shared NFS filesystem** — NOT local NVMe. All 4 workers share the same filesystem. No per-worker model downloads needed.
- **GCS is not mounted** on this instance. All paths use `/path/to/specjax`.
- **Worker-to-worker SSH** requires distributing SSH keys via shared NFS (`/path/to/workspace/w0_ssh_pubkey.tmp`), then w-0 can launch on all workers.
- **Log buffering**: `tee` has large buffers; use `stdbuf -oL tee` for line-buffered output.
