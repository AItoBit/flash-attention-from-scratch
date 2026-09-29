# Benchmark methodology

## What we measure

| Metric | How |
|---|---|
| Latency | CUDA-synchronized `perf_counter`, warmup 5–10, iters 15–30 |
| TFLOP/s | `4 B H N² D / time` |
| tokens/s | `B N / time` |
| Memory | `torch.cuda.max_memory_allocated` after `reset_peak_memory_stats` |
| Occupancy, DRAM, Tensor pipe | Nsight Compute (`profiling/scripts`) |

Causal vs non-causal is reported separately. Causal does less work
(skipped tiles) so TFLOP/s using the full `4 N² D` formula is a lower
bound on hardware efficiency; we still use that formula so rows in a
table are comparable.

## Shapes

Default sweep:

```
B ∈ {1, 2, 4, 8}
H ∈ {8, 16, 32}
N ∈ {128, 256, 512, 1024, 2048, 4096, 8192}
D ∈ {32, 64, 128}
```

Naive / tiled kernels skip `N > 2048` so they do not allocate multi-GB
score tensors in CI.

## Baselines

- `reference` — PyTorch fp32 oracle (correctness, not speed)
- `sdpa` — `torch.nn.functional.scaled_dot_product_attention`
- Triton — optional, Linux-only (`benchmarks/benchmark_triton.py`)

PyTorch SDPA may dispatch to FlashAttention, memory-efficient attention,
or a math kernel depending on dtype, head dim, and GPU. Treat it as a
**system** baseline, not a single algorithm.

## Fairness rules

1. Same layout `[B, H, N, D]`, contiguous.
2. Same dtype for the kernel under test; the oracle always accumulates
   in fp32.
3. Include host-side tensor allocation in memory measurements, not in
   latency (tensors are allocated once, then warmup, then timed).
4. Do not compare a fused fp16 kernel to a naive fp32 kernel and call
   it "FlashAttention speedup". Sweep dtype as a separate axis.

## Roofline

`python/flash_attn_scratch/roofline.py` estimates arithmetic intensity.
Replace `GPU_PEAKS` with datasheet numbers for your card. The qualitative
claim:

```
v1  memory-bound  (N² HBM)
v4  much higher AI, often still memory-bound on DRAM for KV rereads
v7  closer to the compute ridge if MMA actually fires
```

## Reproducing

```bash
python benchmarks/benchmark_latency.py --seq 128 256 512 1024 --dtype fp16
python benchmarks/benchmark_memory.py
python benchmarks/benchmark_pytorch.py --N 1024
python benchmarks/sweep.py --quick
```

Windows: the same commands, from the repo root, with a CUDA-enabled
PyTorch. The CUDA extension JIT-compiles on first `attention(..., impl="flash")`.
