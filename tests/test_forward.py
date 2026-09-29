from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from flash_attn_scratch.reference import attention_reference, online_softmax_torch

CUDA = torch.cuda.is_available()
CUDA_IMPLS = ("naive", "tiled", "online", "flash", "shared", "warp", "mma", "flash2")


def _rand_qkv(B, H, N, D, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, N, D, generator=g, dtype=torch.float32)
    k = torch.randn(B, H, N, D, generator=g, dtype=torch.float32)
    v = torch.randn(B, H, N, D, generator=g, dtype=torch.float32)
    return q.to(device=device, dtype=dtype), k.to(device=device, dtype=dtype), v.to(
        device=device, dtype=dtype
    )


def _max_err(a, b):
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    rel = diff / bf.abs().clamp_min(1e-6)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def _tol(dtype):
    if dtype == torch.float32:
        return 2e-3, 5e-4
    return 2.5e-2, 5e-3


def test_softmax_rows_sum_to_one():
    q, k, v = _rand_qkv(2, 4, 17, 32)
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    p = torch.softmax(scores, dim=-1)
    s = p.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 8, 16), (2, 4, 33, 32), (1, 8, 64, 64), (2, 2, 7, 128)])
def test_reference_shapes(causal, shape):
    q, k, v = _rand_qkv(*shape)
    o = attention_reference(q, k, v, causal=causal)
    assert o.shape == q.shape
    assert torch.isfinite(o).all()


def test_causal_upper_triangle_ignored():
    q, k, v = _rand_qkv(1, 1, 8, 16)
    o = attention_reference(q, k, v, causal=True)
    k2 = k.clone()
    k2[:, :, 5, :] += 50
    o2 = attention_reference(q, k2, v, causal=True)
    # Query positions 0..4 must not see key 5.
    assert torch.allclose(o[:, :, :5], o2[:, :, :5], atol=1e-5)
    assert not torch.allclose(o[:, :, 5:], o2[:, :, 5:], atol=1e-3)


def test_online_softmax_matches_reference():
    q, k, v = _rand_qkv(2, 3, 37, 32)
    ref = attention_reference(q, k, v, causal=True)
    on = online_softmax_torch(q, k, v, causal=True, block_n=16)
    max_abs, mean_abs, _ = _max_err(ref, on)
    assert max_abs < 2e-5 and mean_abs < 1e-6


def test_numerical_stability_large_values():
    q, k, v = _rand_qkv(1, 2, 64, 32)
    q = q * 20
    k = k * 20
    o = attention_reference(q, k, v, causal=False)
    assert torch.isfinite(o).all()
    on = online_softmax_torch(q, k, v, causal=False, block_n=8)
    max_abs, _, _ = _max_err(o, on)
    assert max_abs < 2e-4


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("impl", CUDA_IMPLS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cuda_matches_reference(impl, causal, dtype):
    from flash_attn_scratch.attention import attention

    q, k, v = _rand_qkv(2, 4, 48, 32, dtype=dtype, device="cuda")
    ref = attention_reference(q, k, v, causal=causal)
    out = attention(q, k, v, causal=causal, impl=impl)
    atol, rtol_mean = _tol(dtype)
    max_abs, mean_abs, _ = _max_err(ref, out)
    assert max_abs < atol, f"{impl} max_abs={max_abs}"
    assert mean_abs < rtol_mean, f"{impl} mean_abs={mean_abs}"
