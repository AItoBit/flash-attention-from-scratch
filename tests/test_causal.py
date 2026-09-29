from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from flash_attn_scratch.reference import attention_reference

CUDA = torch.cuda.is_available()
CUDA_IMPLS = ("naive", "tiled", "online", "flash", "shared", "warp", "mma", "flash2")


@pytest.mark.parametrize("n", [1, 2, 5, 17, 31, 65])
def test_causal_odd_lengths_reference(n):
    q = torch.randn(1, 2, n, 32)
    k = torch.randn(1, 2, n, 32)
    v = torch.randn(1, 2, n, 32)
    o = attention_reference(q, k, v, causal=True)
    assert o.shape[-2] == n
    # First row of causal attention can only see position 0.
    o_only = attention_reference(q[:, :, :1], k[:, :, :1], v[:, :, :1], causal=True)
    assert torch.allclose(o[:, :, :1], o_only, atol=1e-5)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("impl", CUDA_IMPLS)
@pytest.mark.parametrize("n", [1, 7, 33, 96])
def test_causal_cuda(impl, n):
    from flash_attn_scratch.attention import attention

    q = torch.randn(1, 2, n, 32, device="cuda")
    k = torch.randn(1, 2, n, 32, device="cuda")
    v = torch.randn(1, 2, n, 32, device="cuda")
    ref = attention_reference(q, k, v, causal=True)
    out = attention(q, k, v, causal=True, impl=impl)
    err = (ref.float() - out.float()).abs().max().item()
    assert err < 3e-3, f"{impl} N={n} err={err}"
