# FlashAttention forward derivation

Start from one query row `q_i ∈ R^D` and the full `K, V ∈ R^{N×D}`.

\[
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d}}, \qquad
p_{ij} = \frac{e^{s_{ij}}}{\sum_t e^{s_{it}}}, \qquad
o_i = \sum_j p_{ij} v_j
\]

Partition the key/value axis into blocks of length `B_c`:

\[
K = [K_1 \mid \cdots \mid K_{T_c}], \qquad
V = [V_1 \mid \cdots \mid V_{T_c}]
\]

and (Stage 4+) the query axis into blocks of length `B_r`.

## Algorithm (one query tile)

```
load Q_i                                     # HBM → SRAM
m_i = -∞, ℓ_i = 0, O_i = 0
for j in 1..T_c:
    load K_j, V_j                            # HBM → SRAM
    S_ij = Q_i K_jᵀ · scale                  # SRAM / Tensor Cores
    apply causal mask on S_ij
    m_new = max(m_i, rowmax(S_ij))
    P_ij  = exp(S_ij - m_new)
    ℓ_i   = exp(m_i - m_new) ℓ_i + rowsum(P_ij)
    O_i   = diag(exp(m_i - m_new)) O_i + P_ij V_j
    m_i   = m_new
O_i ← diag(ℓ_i)^{-1} O_i                     # HBM ← SRAM
L_i ← m_i + log ℓ_i                          # saved for backward
```

Nothing of size `N×N` is ever stored. The working set is

```
Q tile   [B_r, D]
K tile   [B_c, D]
V tile   [B_c, D]
S tile   [B_r, B_c]
O acc    [B_r, D]
```

which fits in shared memory for the tile sizes used here (`32×32` or
`64×32`, `D ≤ 128`).

## Where the FLOPs go

Inside the `j` loop there are two GEMMs (`QKᵀ` and `PV`) and a handful
of pointwise ops (max, exp, rescale). On Tensor Cores the GEMMs dominate;
on a pure SIMT kernel the softmax is a real fraction of runtime, which
is why Stage 6 moves the max/sum onto warp shuffles.

## Causal skip

If every key index in the tile is `> ` every query index in the tile,
the whole iteration is a no-op and we `break` (causal sequences are
monotonic in `j`). Partial tiles still evaluate the per-element predicate.

## Implementation map

| File | What it adds |
|---|---|
| `src/v3_online_softmax/online_softmax.cu` | recurrence, `B_r = 1` |
| `src/v4_flash/flash_fwd.cu` | query tiling, first fused kernel |
| `src/v5_shared_memory/flash_shared.cu` | vectorized loads, double buffer, autotunable `BM/BN` |
| `src/v6_warp/flash_warp.cu` | warp-owned rows, `__shfl_xor_sync` |
| `src/v7_tensorcore/flash_mma.cu` | WMMA for both GEMMs |
| `src/v8_flash2/flash2.cu` | FA2-style warp partitioning of the Q tile |
