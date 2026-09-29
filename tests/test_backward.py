from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from flash_attn_scratch.reference import attention_reference

CUDA = torch.cuda.is_available()


def _rand(B, H, N, D, dtype=torch.float32, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(1)
    q = torch.randn(B, H, N, D, generator=g)
    k = torch.randn(B, H, N, D, generator=g)
    v = torch.randn(B, H, N, D, generator=g)
    return (q.to(device=device, dtype=dtype),
            k.to(device=device, dtype=dtype),
            v.to(device=device, dtype=dtype))


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_autograd(causal):
    from flash_attn_scratch.attention import attention
    from flash_attn_scratch.extension import load_extension

    q, k, v = _rand(2, 2, 32, 32, dtype=torch.float32)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    ref = attention_reference(q, k, v, causal=causal)
    dout = torch.randn_like(ref)
    ref.backward(dout)
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q.grad = k.grad = v.grad = None
    o, lse = attention(q.detach(), k.detach(), v.detach(), causal=causal,
                       impl="flash", return_lse=True)
    dq, dk, dv = load_extension().flash_bwd(
        q.detach(), k.detach(), v.detach(), o, dout.contiguous(), lse, causal, None
    )
    assert (dq - dq_ref).abs().max().item() < 2e-3
    assert (dk - dk_ref).abs().max().item() < 2e-3
    assert (dv - dv_ref).abs().max().item() < 2e-3
