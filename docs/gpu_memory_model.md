# GPU memory model

A CUDA kernel sees a hierarchy. FlashAttention is an algorithm for walking
that hierarchy on purpose.

```
registers     ~ 256 KB / SM, ~1 TB/s / SM locally
     ↑
shared memory ~ 64–228 KB / SM, on-chip, programmer-managed
     ↑
L2 cache      ~ 4–50 MB, device-wide
     ↑
HBM           ~ 0.5–3 TB/s, 10–80 GB, the bottleneck
```

## What "IO-aware" means

The naive attention kernel:

1. Reads `Q[i, :]` and `K[j, :]` from HBM, writes `S[i, j]` to HBM.
2. Reads `S[i, :]` from HBM, writes `P[i, :]` to HBM.
3. Reads `P[i, :]` and `V[:, :]` from HBM, writes `O[i, :]`.

`S` and `P` are each `N²` and used once. That is the definition of
wasted HBM bandwidth.

FlashAttention keeps `S` and `P` in **shared memory + registers** for the
lifetime of a tile, and only `Q, K, V, O` (plus a length-`N` log-sum-exp
vector) touch HBM.

## Coalescing

Threads in a warp should read consecutive addresses. Layout `[B, H, N, D]`
with `D` contiguous is friendly when a warp walks `D` for a fixed `(b,h,n)`.
It is hostile when a warp walks `N` of the score matrix with `D` as the
inner reduction — that is why tiling exists.

## Bank conflicts

Shared memory is 32 banks. `Qs[row][col]` with `col` consecutive across
a warp is conflict-free. `Qs[row][col+1]` padding (`BK+1`) is the usual
fix when a warp strides the other index. Stage 2 pads tiles; later stages
keep the same discipline.

## Occupancy vs tile size

A `BM=64, BN=64, D=128` fused kernel wants ~150 KB of shared memory.
That drops occupancy to 1 block/SM on many GPUs. Smaller tiles raise
occupancy and lower arithmetic intensity. The autotuner in
`python/flash_attn_scratch/autotune.py` exists because this trade is
**shape-dependent**.

## Kernel launch overhead

Stage 1 launches three kernels. That is visible at `N ≤ 256`. Fusion
(Stage 4) removes two launches and, more importantly, two full HBM
round-trips of `N²`.
