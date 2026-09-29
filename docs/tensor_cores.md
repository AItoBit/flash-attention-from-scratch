# Tensor Cores

NVIDIA Tensor Cores execute mixed-precision GEMM fragments, not scalar
FMAs. The programming models, oldest to newest:

| API | Granularity | Notes |
|---|---|---|
| WMMA (`nvcuda::wmma`) | 16×16×16 fragments, warp-wide | What Stage 7 uses |
| MMA PTX (`mma.sync`) | explicit shape / dtype / layout | next step after WMMA |
| `wgmma` (Hopper) | warp-group, async | Stage 9 |
| CUTLASS / CuTe | full pipeline | production |

This repository stops at WMMA on purpose. The point is the **impedance
mismatch** with softmax, not a CUTLASS clone.

## The mismatch

```
Q fragment ── MMA ──► score fragment (fp32 accumulator)
                         │
                         ▼
                   store to SRAM
                         │
                   online softmax     ← not a matmul
                         │
                   P as fp16 SRAM
                         │
P fragment ── MMA ──► O fragment
```

You cannot softmax inside an accumulator fragment in WMMA. You must
`store_matrix_sync` to a regular `[BM, BN]` layout, run the recurrence
the SIMT kernels already use, `float2half` the probabilities, then
`load_matrix_sync` for the PV MMA.

That round-trip is the whole lesson. A faster kernel would keep `S` in
registers with `ldmatrix` / MMA PTX and do softmax in registers per
lane; the numerical algorithm does not change.

## Layout convention used here

`C = A @ B` with WMMA 16×16×16:

- QKᵀ: `A = Q` row-major `[BM, D]`, `B = Kᵀ` which is `K` row-major
  `[BN, D]` viewed as col-major `[D, BN]`.
- PV: `A = P` row-major `[BM, BN]`, `B = V` row-major `[BN, D]`.

Head dim is padded to a multiple of 16. fp32 inputs are converted to
fp16 for the MMA (Tensor Cores do not run fp32×fp32 on Turing/Ampere in
the same way; Ampere TF32 is a possible extension).

## Utilization

Nsight Compute metric to watch: `sm__pipe_tensor_cycles_active`. If it
is low after Stage 7, the kernel is still bound on the softmax round-trip
or on shared-memory bandwidth, not on math. That is expected for this
WMMA teaching kernel; saturating the pipe wants a deeper MMA pipeline
(Stage 9).
