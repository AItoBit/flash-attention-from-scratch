# Backward derivation

Let `S = scale · Q Kᵀ`, `P = softmax(S)`, `O = P V`.
Given `dO = ∂L/∂O` we want `dQ, dK, dV`.

## Unfused identities

\[
dV = P^\top dO
\]

\[
dP = dO V^\top
\]

Softmax Jacobian along a row (`⊙` is elementwise, `1` is the all-ones
vector of length `N`):

\[
dS = P \odot \bigl(dP - (dP \odot P)\,1 \bigr)
\]

The term `(dP ⊙ P) 1` equals `rowsum(dO ⊙ O)` because `O = P V` and
`dP = dO Vᵀ`, so

\[
D_i = \sum_d dO_{id}\, O_{id}, \qquad
dS_{ij} = P_{ij}\,(dP_{ij} - D_i)
\]

Then

\[
dQ = dS\, K \cdot \mathrm{scale}, \qquad
dK = dS^\top Q \cdot \mathrm{scale}
\]

`D ∈ R^{B,H,N}` is computed once from `O` and `dO`. It does not depend
on the KV tile.

## Fused recomputation

We do **not** save `P`. Forward saves `O` and the log-sum-exp
`L_i = m_i + log ℓ_i`. For each tile:

```
S_ij = scale · Q_i K_jᵀ
P_ij = exp(S_ij - L_i)          # equals the forward P, stably
dV_j += P_ijᵀ dO_i
dP_ij = dO_i V_jᵀ
dS_ij = P_ij ⊙ (dP_ij - D_i)
dQ_i += dS_ij K_j · scale
dK_j += dS_ijᵀ Q_i · scale
```

Causal tiles that were skipped in the forward are skipped here too;
`P` is zero there, so they contribute nothing.

## Atomics

Each Q-tile block owns `dQ_i` (no atomics). Many Q tiles touch the same
`K_j` / `V_j`, so `dK` and `dV` use `atomicAdd`. A production FA2
backward would partition the K axis across thread blocks to avoid
atomics; that is left as a Stage-9 exercise.

## Implementation

`src/backward/flash_bwd.cu`

1. `delta_kernel` writes `D`.
2. `flash_bwd_kernel` recomputes `P` tile by tile and accumulates
   `dQ, dK, dV`.

Python autograd: `FlashAttentionFn` in `python/flash_attn_scratch/attention.py`.
