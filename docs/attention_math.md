# Attention math

Scaled dot-product attention (Vaswani et al., 2017) is

\[
O = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

with tensors laid out as

```
Q, K, V, O : [B, H, N, D]
S = Q Kᵀ     : [B, H, N, N]
P = softmax(S)
O = P V      : [B, H, N, D]
```

`B` is batch, `H` is heads, `N` is sequence length, `D` is head dimension.
The conventional algorithm materializes `S` and `P`. That is an `N × N`
matrix per batch element and head.

## FLOPs vs bytes

QKᵀ is a GEMM of shape `(N, D) × (D, N)`: `2 N² D` FLOPs.
PV is `(N, N) × (N, D)`: another `2 N² D` FLOPs.

```
FLOPs ≈ 4 B H N² D
```

Bytes for the naive algorithm are dominated by **writing and rereading**
`S` and `P`, each `N²` elements:

```
bytes_naive ≈ 2 B H N² · sizeof(acc)   +   O(B H N D)
```

For `N = 4096`, `H = 32`, `B = 1`, fp32 scores:

```
S + P ≈ 2 · 32 · 4096² · 4 bytes ≈ 4.3 GB
```

just for intermediates. Arithmetic is cheap relative to that HBM traffic
on every datacenter GPU of the last five years. FlashAttention exists
because attention is **IO-bound**, not because softmax is slow.

## Softmax, stably

For a row `x ∈ R^N`:

\[
m = \max_j x_j, \qquad
\ell = \sum_j e^{x_j - m}, \qquad
p_i = e^{x_i - m}/\ell
\]

Subtracting `m` is required. `exp(x)` overflows in fp32 around `x ≈ 89`.
Attention logits with `1/√d` scaling are usually safe in fp32, but fp16
accumulators are not. Every kernel in this repo accumulates softmax in
fp32, including the fp16/bf16 Tensor Core path.

## Causal masking

Decoder attention requires `P_{ij} = 0` for `j > i`. Equivalently

```
S_ij = -∞    if j > i
```

We never build a mask tensor. Inside a tile, a thread writes `-inf` when
`key_position > query_position`. Entire KV tiles strictly above the
diagonal are skipped:

```
         K blocks
       0    1    2    3
Q 0   [X]
  1   [X]  [X]
  2   [X]  [X]  [X]
  3   [X]  [X]  [X]  [X]
```

## Scale

Default scale is `1/√D`. Pass `scale=` to override (useful for reproducing
papers that fold extra constants into the score).
