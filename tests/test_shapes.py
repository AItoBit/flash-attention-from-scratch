from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from flash_attn_scratch.reference import attention_reference

CUDA = torch.cuda.is_available()
CUDA_IMPLS = ("naive", "online", "flash", "warp")


@pytest.mark.parametrize("B,H,N,D", [
    (1, 1, 1, 8),
    (1, 1, 3, 32),
    (2, 8, 16, 64),
    (1, 4, 128, 128),
    (4, 2, 15, 40),
])
def test_reference_arbitrary_shapes(B, H, N, D):
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    o = attention_reference(q, k, v, causal=True)
    assert o.shape == (B, H, N, D)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("impl", CUDA_IMPLS)
@pytest.mark.parametrize("D", [32, 64, 128])
def test_head_dims(impl, D):
    from flash_attn_scratch.attention import attention

    q = torch.randn(1, 2, 48, D, device="cuda")
    k = torch.randn(1, 2, 48, D, device="cuda")
    v = torch.randn(1, 2, 48, D, device="cuda")
    ref = attention_reference(q, k, v)
    out = attention(q, k, v, impl=impl)
    assert (ref.float() - out.float()).abs().max().item() < 3e-3
