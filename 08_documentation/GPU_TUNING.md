# GPU Tuning Guide

This guide explains the training-time knobs in `config.py` and how to adapt them for common NVIDIA GPUs (A30, A100, 3090, 4090, T4). It also documents the rationale behind the defaults.

### Glossary: knobs and their impact (plain-English)

- **enable_tf32**
  - What it is: Uses TensorFloat-32 for FP32 matmul on Ampere/Ada GPUs.
  - Impact: Faster training with near-FP32 quality; no extra code changes.
  - Trade-offs: Tiny numerical differences vs exact FP32 (negligible here).
  - When to change: Keep True on A30/3090/4090/A100; set False on older GPUs.

- **enable_amp**
  - What it is: Mixed precision (autocast) during forward/backward.
  - Impact: Large speed + memory gains (bf16/fp16); common best practice.
  - Trade-offs: Rare instability on some models if precision too low.
  - When to change: Keep True; turn off only to debug odd NaNs/divergence.

- **amp_dtype ("bf16" | "fp16")**
  - What it is: Which mixed-precision format to use.
  - Impact: bf16 is more numerically stable on Ampere/Ada; fp16 is more widely supported.
  - Trade-offs: fp16 can be slightly faster on some kernels but more NaN-prone.
  - When to change: Prefer bf16 on A30/3090/4090/A100. If you see NaNs, try fp16.

- **enable_compile**
  - What it is: `torch.compile` JIT to fuse/optimize graphs.
  - Impact: Higher throughput after warmup; initial compile time + extra memory.
  - Trade-offs: More RAM/VRAM during compile; harder to debug stacktraces.
  - When to change: Disable if memory is tight or while diagnosing crashes.

- **max_length**
  - What it is: Training sequence length (context window). In this repo, `block_size = max_length`.
  - Impact: Longer sequences improve long-context learning and tokens/step; increases VRAM roughly quadratically for attention.
  - Trade-offs: Higher OOM risk; slower per-iteration.
  - When to change: If OOM, reduce (e.g., 1024 → 768) after lowering batch_size.

- **batch_size**
  - What it is: Per-process batch size (global batch = batch_size × world_size).
  - Impact: Biggest VRAM knob and strongest effect on throughput.
  - Trade-offs: Too large → OOM; too small → underutilized GPU, noisy gradients.
  - When to change: First thing to lower if you hit OOM; raise if VRAM < 80%.

- **eval_steps**
  - What it is: How often to run evaluation.
  - Impact: Larger = fewer eval pauses and lower memory spikes.
  - Trade-offs: Metrics update less frequently.
  - When to change: Increase if eval causes pauses/OOM; decrease if you need tighter monitoring.

- **logging_steps**
  - What it is: How often to log scalars.
  - Impact: Larger reduces overhead in busy loops.
  - Trade-offs: Coarser logs.
  - When to change: Increase for production runs; decrease when debugging.

- **eval_iters**
  - What it is: Batches used to compute each eval loss.
  - Impact: Larger gives steadier metrics, but more time/memory.
  - Trade-offs: Slower eval and higher VRAM.
  - When to change: Reduce if eval is slow/memory-heavy (e.g., 100 → 50).

- **dataloader_num_workers**
  - What it is: CPU processes to load data.
  - Impact: More workers can feed the GPU faster; too many can thrash CPU.
  - Trade-offs: CPU contention if set too high.
  - When to change: Start at 2–6 and tune based on CPU usage and GPU idle time.

- **dataloader_pin_memory**
  - What it is: Locks host memory to speed CPU→GPU copies.
  - Impact: Faster transfers; standard for CUDA training.
  - Trade-offs: Pins RAM, slightly reducing system flexibility.
  - When to change: Keep True on CUDA; disable only if you suspect pinning issues.

- **dataloader_persistent_workers**
  - What it is: Keeps loader workers alive across epochs.
  - Impact: Reduces epoch boundary overhead.
  - Trade-offs: Slightly longer-lived processes.
  - When to change: Keep True for long runs; disable if frequent config changes.

- **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** (environment variable)
  - What it is: Alternative allocator that reduces fragmentation.
  - Impact: Helps long runs avoid OOM due to fragmented memory.
  - Trade-offs: Different allocation behavior; generally safe.
  - When to change: Set when you see OOM with lots of reserved-but-unallocated memory in the error.

## Runtime knobs (in `config.py -> slm_config`)

- enable_tf32: Enable TF32 matmul on Ampere+/Ada. Good speedups with minimal quality loss.
- enable_amp: Mixed precision (bf16/fp16) autocast during forward/backward.
- amp_dtype: Preferred AMP dtype. bf16 on Ampere/Ada is stable; use fp16 if bf16 unsupported.
- enable_compile: Toggle torch.compile JIT. Faster throughput; increases compile time and memory. Disable when memory is tight or debugging.
- max_length: Sequence length (context window). Higher → more tokens/step but more VRAM. In this trainer `block_size = max_length`.
- batch_size: Per-process batch size. Global batch = batch_size * world_size.
- eval_steps, logging_steps: Frequency of eval/logging; larger values reduce overhead.
- eval_iters: Number of mini-batches used to compute eval loss. Higher = steadier metric; more time/memory.

DataLoader-related knobs (also in `slm_config`):
- dataloader_num_workers: CPU worker processes for loading. 2–6 is typical; tune to your CPU.
- dataloader_pin_memory: Pin host memory for faster H2D copies. Keep True on CUDA.
- dataloader_persistent_workers: Keep workers alive between epochs; reduces startup overhead.

At runtime, the trainer will:
- Set TF32 flags when enable_tf32=True.
- Choose bf16 if amp_dtype=bf16 and the GPU supports it; else fp16 if enable_amp=True; else float32.
- Use max_length directly as block_size to maximize tokens/sec.
- Respect enable_compile to decide whether to JIT-compile the model.

## Recommended presets

Start with these; watch `nvidia-smi` and keep VRAM below ~90% by adjusting batch_size then max_length.

### A30 24GB (x2)
- enable_tf32: true
- enable_amp: true
- amp_dtype: bf16
- max_length: 1024
- batch_size: 24 (per GPU; tune 20-28)
- eval_steps: 2000
- logging_steps: 100

### RTX 3090 24GB (single)
- enable_tf32: true
- enable_amp: true
- amp_dtype: bf16 (if unstable, use fp16)
- max_length: 1024 (drop to 768 if OOM)
- batch_size: 12-20 (start at 16)
- eval_steps: 2000
- logging_steps: 100-200

Quick recipe (3090):
```python
"enable_tf32": True,
"enable_amp": True,
"amp_dtype": "bf16",
"enable_compile": False,   # set True if memory allows
"max_length": 1024,        # drop to 768 if OOM persists
"batch_size": 16,          # try 12 if OOM; raise if VRAM < 80%
"eval_steps": 2000,
"logging_steps": 100,
"eval_iters": 50,
```

### A100 40/80GB
- enable_tf32: true
- enable_amp: true
- amp_dtype: bf16
- max_length: 2048
- batch_size: 32-64 (per GPU)

### RTX 4090 24GB
- enable_tf32: true
- enable_amp: true
- amp_dtype: bf16
- max_length: 1024
- batch_size: 16-24

### T4 16GB
- enable_tf32: false
- enable_amp: true
- amp_dtype: fp16 (bf16 unsupported)
- max_length: 768
- batch_size: 8-12

## Conservative settings (commented example)

Use these when publishing for broad hardware compatibility:

```python
# Conservative baseline for wide compatibility
# "enable_tf32": False,
# "enable_amp": True,
# "amp_dtype": "fp16",
# "max_length": 768,
# "batch_size": 8,
# "eval_steps": 1000,
# "logging_steps": 50,
```

## Model FLOPs Utilization (MFU) Optimization

**What is MFU?**
- MFU measures how efficiently your GPUs are being used
- Higher MFU = better GPU utilization = faster training
- Target: 15-25% for most setups, 30%+ for optimized setups

**GPU-Specific MFU Targets:**

| GPU Type | Memory | Good MFU | Excellent MFU | Recommended Batch Size |
|----------|--------|----------|---------------|----------------------|
| **A30** | 24GB | 15-20% | 25-30% | 12-18 per GPU |
| **A40** | 48GB | 20-25% | 30-35% | 20-30 per GPU |
| **A100** | 40GB/80GB | 25-30% | 35-40% | 16-24 per GPU |
| **V100** | 16GB/32GB | 15-20% | 25-30% | 8-12 per GPU |
| **RTX 4090** | 24GB | 20-25% | 30-35% | 12-18 per GPU |
| **RTX 4080** | 16GB | 15-20% | 25-30% | 8-12 per GPU |
| **RTX 3090** | 24GB | 18-22% | 28-32% | 10-16 per GPU |

**MFU Optimization Steps:**

1. **Check Current MFU:**
   ```bash
   # Look for MFU in training output
   tail -f 04_training/model_training.log | grep mfu
   ```

2. **Increase Batch Size (Most Important):**
   ```python
   # In config.py - adjust based on your GPU
   "batch_size": 12,  # For A30 (24GB)
   "batch_size": 20,  # For A40 (48GB)
   "batch_size": 8,   # For RTX 4080 (16GB)
   ```

3. **Optimize Sequence Length:**
   ```python
   # Longer sequences = more compute per batch
   "max_length": 1024,  # For high-memory GPUs
   "max_length": 512,   # For lower-memory GPUs
   ```

4. **Enable Optimizations:**
   ```python
   "enable_amp": True,        # Mixed precision
   "enable_compile": True,    # Torch compile
   "enable_tf32": True,       # TensorFloat-32
   ```

5. **Monitor Memory Usage:**
   ```bash
   # Check GPU memory usage
   nvidia-smi -l 1
   
   # Should use 70-85% of GPU memory for good MFU
   ```

**Quick MFU Check:**
```bash
# Run this during training to monitor MFU
watch -n 1 'nvidia-smi && echo "---" && tail -5 04_training/model_training.log | grep mfu'
```

**Expected Training Output:**
```
# Good MFU (15-20%)
iter 100: loss 8.5174, time 1011.21ms, mfu 15.2%

# Poor MFU (1-5%)
iter 100: loss 8.5174, time 1011.21ms, mfu 1.6%
```

## How to tune

1. Start with the preset for your GPU.
2. Monitor `nvidia-smi` during warmup:
   - If VRAM < 80%, increase batch_size first, then max_length.
   - If OOM occurs, reduce batch_size first, then max_length.
3. Watch throughput (tokens/sec) and MFU in logs; higher is better if loss is stable.
4. If NaNs/instability:
   - Switch amp_dtype to fp16, reduce LR by 10-20%.
   - Temporarily disable compile to debug.

## Troubleshooting playbooks

### Out of memory (OOM)
- Lower `batch_size` first (e.g., 24 → 16 → 12).
- Then lower `max_length` (e.g., 1024 → 768).
- Set `enable_compile=False` to reduce peak memory.
- Reduce `eval_iters` (e.g., 100 → 50) and increase `eval_steps`.
- Consider enabling segmented allocator to reduce fragmentation:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Low throughput / under-utilized GPU
- Increase `batch_size` until VRAM ~85–90%.
- Keep `enable_tf32=True`, `enable_amp=True`.
- Enable `enable_compile=True` once stable.
- Increase `eval_steps`/`logging_steps` to reduce overhead.
- Tune `dataloader_num_workers` (2–6) and keep `pin_memory=True`.

### Eval spikes memory/time
- Increase `eval_steps` (run eval less often).
- Reduce `eval_iters`.
- Ensure eval runs after warmup, not at step 0 (the trainer already skips step 0).

### NaNs / instability
- Switch to `amp_dtype=fp16`.
- Reduce learning rate by 10–20%.
- Disable `enable_compile` while diagnosing.

## Notes

- Mixed precision and TF32 speed up training with negligible quality impact at this SLM scale.
- Full-context training (block_size = max_length) improves utilization; ensure clean tokenization.
- For multi-GPU, launch with torchrun/NCCL; set dataloader worker settings if the input pipeline is the bottleneck.
- The trainer frees CUDA cache after evaluations to limit memory spikes.

## GPU Troubleshooting

For comprehensive GPU troubleshooting, see the dedicated [GPU Troubleshooting Guide](GPU_TROUBLESHOOTING.md).

**Common GPU Issues:**
- **"ProcessGroupNCCL is only supported with GPUs, no GPUs found!"** → Driver installation required
- **"CUDA out of memory"** → Reduce batch size or max_length
- **"NVIDIA driver version is insufficient"** → Update drivers
- **Multi-GPU not working** → Check GPU detection and NCCL installation
- **Low GPU utilization** → Optimize batch size and data loading

**Quick Verification:**
```bash
# Check GPU status
nvidia-smi

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU count
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```


