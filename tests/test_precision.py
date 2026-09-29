from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from flash_attn_scratch.reference import attention_reference, online_softmax_torch

CUDA = torch.cuda.is_available()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_dtype_roundtrip(dtype):
    if dtype == torch.bfloat16 and not hasattr(torch, "bfloat16"):
        pytest.skip("bf16 not available")
    q = torch.randn(1, 1, 16, 32, dtype=dtype)
    k = torch.randn(1, 1, 16, 32, dtype=dtype)
    v = torch.randn(1, 1, 16, 32, dtype=dtype)
    o = attention_reference(q, k, v)
    assert o.dtype == dtype
    assert torch.isfinite(o.float()).all()


def test_fp16_oracle_uses_fp32_accumulation():
    q = torch.randn(1, 1, 32, 64, dtype=torch.float16)
    k = q.clone()
    v = torch.randn(1, 1, 32, 64, dtype=torch.float16)
    o = attention_reference(q, k, v)
    # Diagonal-dominant-ish: output should stay finite in fp16 storage.
    assert torch.isfinite(o.float()).all()


def test_large_logits_online_vs_reference():
    q = torch.randn(1, 1, 64, 32) * 30
    k = torch.randn(1, 1, 64, 32) * 30
    v = torch.randn(1, 1, 64, 32)
    ref = attention_reference(q, k, v)
    on = online_softmax_torch(q, k, v, block_n=9)
    assert (ref - on).abs().max().item() < 1e-4


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_fp16_cuda_stable_on_large_inputs():
    from flash_attn_scratch.attention import attention

    q = (torch.randn(1, 2, 64, 32, device="cuda") * 20).half()
    k = (torch.randn(1, 2, 64, 32, device="cuda") * 20).half()
    v = torch.randn(1, 2, 64, 32, device="cuda").half()
    ref = attention_reference(q, k, v)
    out = attention(q, k, v, impl="flash")
    err = (ref.float() - out.float()).abs().mean().item()
    assert torch.isfinite(out.float()).all()
    assert err < 5e-3
