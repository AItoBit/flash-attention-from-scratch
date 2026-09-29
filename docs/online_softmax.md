# Online softmax

Ordinary softmax over a row `x` needs two passes: a max, then a sum of
exponentials. FlashAttention cannot afford two passes over `N` because
it never stores the row.

Milakov & Gimelshein (2018) and the FlashAttention paper use a
single-pass recurrence. For blocks `x^{(1)}, x^{(2)}, …` of the row:

\[
m_j = \max(m_{j-1}, \max x^{(j)})
\]

\[
\ell_j = e^{m_{j-1}-m_j}\,\ell_{j-1} + \sum \exp(x^{(j)} - m_j)
\]

The output accumulator is rescaled the same way. If `O` holds
`∑ p·V` computed under the old max,

\[
O \leftarrow e^{m_{j-1}-m_j}\,O + \exp(x^{(j)}-m_j)\,V^{(j)}
\]

After the last block, `O ← O / ℓ`. Equivalently one can store
log-sum-exp `L = m + log ℓ` and divide at the end; we write `L` out for
the backward pass.

## State per query row

```
float m_i;       // running max
float l_i;       // running sum of exp(s - m)
float O_i[D];    // unnormalized output
```

That is `O(D)` extra state instead of `O(N)`.

## Why this is numerically the same

By induction, `m_j` is the max of everything seen so far, and `ℓ_j`
equals `∑ exp(x - m_j)` over that prefix. The final `p` matches the
two-pass softmax up to fp32 rounding. Tests in `tests/test_precision.py`
multiply `Q` and `K` by 20 to force the naive `exp` into the overflow
regime; the online form stays finite.

## Stage 3 vs Stage 4

Stage 3 (`src/v3_online_softmax`) runs **one query position per block**
so the recurrence is obvious: there is a single `(m, ℓ, O)`.

Stage 4 tiles queries as well (`BM` rows per block). Each row has its
own `(m, ℓ, O)`; the recurrence is identical, just vectorized across the
tile. That is the only conceptual jump from "online softmax" to
"FlashAttention".
