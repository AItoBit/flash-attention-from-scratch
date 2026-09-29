from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Exact scaled dot-product attention. The correctness oracle.

    Args:
        q, k, v: [B, H, N, D] in fp32 / fp16 / bf16
        causal: if True, position j > i is masked to -inf before softmax
        scale: defaults to 1/sqrt(D)

    Compute is promoted to fp32 so the oracle stays numerically stable even
    when the inputs are half precision.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must all have shape [B, H, N, D]")

    d = q.size(-1)
    scale_f = (1.0 / math.sqrt(d)) if scale is None else scale
    qf = q.float()
    kf = k.float()
    vf = v.float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale_f
    if causal:
        n = q.size(-2)
        mask = torch.ones(n, n, device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, vf).to(dtype=q.dtype)


def attention_pytorch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """PyTorch scaled_dot_product_attention baseline (math / mem-efficient / flash)."""
    d = q.size(-1)
    scale_f = (1.0 / math.sqrt(d)) if scale is None else scale
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                          is_causal=causal, scale=scale_f)


def online_softmax_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
    block_n: int = 64,
) -> torch.Tensor:
    """PyTorch implementation of the online-softmax recurrence (Stage 3 math).

    Streams K/V in blocks of `block_n` and never materializes the full N×N
    matrix. Used as an independent algorithmic check of the CUDA kernels.
    """
    b, h, n, d = q.shape
    scale_f = (1.0 / math.sqrt(d)) if scale is None else scale
    qf, kf, vf = q.float(), k.float(), v.float()
    m = q.new_full((b, h, n), float("-inf"), dtype=torch.float32)
    l = torch.zeros(b, h, n, device=q.device, dtype=torch.float32)
    o = torch.zeros(b, h, n, d, device=q.device, dtype=torch.float32)

    for j0 in range(0, n, block_n):
        j1 = min(n, j0 + block_n)
        kj = kf[:, :, j0:j1, :]
        vj = vf[:, :, j0:j1, :]
        s = torch.matmul(qf, kj.transpose(-2, -1)) * scale_f
        if causal:
            q_idx = torch.arange(n, device=q.device)[:, None]
            k_idx = torch.arange(j0, j1, device=q.device)[None, :]
            s = s.masked_fill(k_idx > q_idx, float("-inf"))
        m_tile = s.amax(dim=-1)
        m_new = torch.maximum(m, m_tile)
        alpha = torch.exp(m - m_new)
        p = torch.exp(s - m_new.unsqueeze(-1))
        p = torch.nan_to_num(p, nan=0.0)
        l = alpha * l + p.sum(dim=-1)
        o = alpha.unsqueeze(-1) * o + torch.matmul(p, vj)
        m = m_new

    return (o / l.clamp_min(1e-30).unsqueeze(-1)).to(dtype=q.dtype)
