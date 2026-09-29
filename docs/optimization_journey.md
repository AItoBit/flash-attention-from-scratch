# Optimization journey

Each stage is a complete, tested implementation. The point of the
repository is the **delta** between stages, not the final kernel alone.

## v1 naive — DRAM bound by construction

Three kernels, `S` and `P` in HBM. Expected Nsight picture:

- high DRAM throughput relative to peak
- low SM utilization on the QK kernel (one thread, one `S_ij`, inner
  loop over `D` with no reuse)
- softmax is a row reduction and looks latency-bound

This is the baseline the rest of the repo exists to beat.

## v2 tiled QK — reuse on chip, still writes N×N

Shared-memory GEMM for `QKᵀ`. `Q` and `K` tiles are reused `BN` / `BM`
times. DRAM traffic for QK drops; **the N×N write remains**. Softmax and
PV are unchanged. Expected: lower `dram__bytes` on the QK kernel,
similar end-to-end memory footprint.

## v3 online softmax — the algorithm, not yet the kernel

One query row per block, stream KV. No `N×N` in HBM. Occupancy is poor
(`N` blocks with modest work). Numerically this is already FlashAttention.
Performance is a demonstration that tiling queries matters.

## v4 fused — first real FlashAttention

Query tiles × KV stream, one kernel, no `N×N`. This is the milestone:

> scores never hit global memory

End-to-end memory becomes `O(N)` in sequence length. Latency should
pull away from v1/v2 as `N` grows; at `N=128` launch overhead and tile
padding can hide the win.

## v5 shared-memory — the data-movement pass

Same math as v4. Changes: `float4`/`half2`-style vectorized loads,
double-buffered KV tiles, compile-time `BM/BN` for the autotuner,
bank-aware layouts. Expected: higher `gld_efficiency`, fewer
`stall_sync` from the extra prefetch (sometimes more smem → less
occupancy; measure, don't guess).

## v6 warp — fewer block-wide reductions

Each warp owns a group of query rows. Row max/sum use
`__shfl_xor_sync` instead of a shared-memory reduction tree. Expected:
lower `stall_group` / `stall_sync`, more eligible warps.

## v7 Tensor Core — shift the bound

WMMA for `QKᵀ` and `PV`. Softmax still SIMT. Expected: `pipe_tensor`
becomes nonzero; the kernel may still not be compute-bound because of
the fragment round-trip. That gap is documented in `tensor_cores.md`.

## v8 FA2 scheduling — work partition

Warps own disjoint Q rows and only synchronize around KV loads. Larger
`BM`. The experiment: **does changing who owns which rows improve
utilization without changing the math?** Compare Nsight `sm__throughput`
and `eligible_warps` against v6/v4 at the same tile size.

## How to read a profile

```
v1  DRAM bound, N² writes
v2  DRAM still N², QK more efficient
v3  IO-aware algorithm, under-occupied
v4  fused, O(N) HBM, first speedup cliff
v5  load efficiency / prefetch
v6  sync reduction
v7  math pipe (Tensor Cores)
v8  scheduling / occupancy
```

Fill the table in the README with `python benchmarks/benchmark_latency.py`
and `profiling/scripts/ncu_attention.sh` on your GPU. Numbers on a
laptop and an H100 will disagree; the **shape of the story** should not.
