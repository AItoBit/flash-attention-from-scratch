from __future__ import annotations

from typing import Optional

import torch

from .reference import attention_pytorch_sdpa, attention_reference, online_softmax_torch

IMPLS = (
    "reference",
    "sdpa",
    "online_torch",
    "naive",
    "tiled",
    "online",
    "flash",
    "shared",
    "warp",
    "mma",
    "flash2",
)

FUSED_IMPLS = ("online", "flash", "shared", "warp", "mma", "flash2")
MATERIALIZING_IMPLS = ("naive", "tiled")


class FlashAttentionFn(torch.autograd.Function):
    """Autograd glue for fused CUDA kernels that save LSE for the backward."""

    @staticmethod
    def forward(ctx, q, k, v, causal, scale, impl):  # noqa: ANN001
        from .extension import load_extension

        ext = load_extension()
        fn = {
            "online": ext.online_fwd,
            "flash": ext.flash_fwd,
            "warp": ext.flash_warp_fwd,
            "mma": ext.flash_mma_fwd,
            "flash2": ext.flash2_fwd,
        }.get(impl)
        if impl == "shared":
            o, lse = ext.flash_shared_fwd(q, k, v, causal, scale, 32, 32)
        elif fn is None:
            raise ValueError(f"no fused autograd path for impl={impl}")
        else:
            o, lse = fn(q, k, v, causal, scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):  # noqa: ANN001
        from .extension import load_extension

        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = load_extension().flash_bwd(q, k, v, o, do.contiguous(), lse,
                                                ctx.causal, ctx.scale)
        return dq, dk, dv, None, None, None


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
    impl: str = "flash",
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Unified attention entry point.

    `impl` selects which stage of the optimization laboratory to run:

        reference     Stage 0  PyTorch matmul + softmax oracle
        sdpa          PyTorch scaled_dot_product_attention
        online_torch  Stage 3  online softmax in PyTorch (no CUDA)
        naive         Stage 1  three CUDA kernels, N×N in HBM
        tiled         Stage 2  tiled QK^T, still materializes S
        online        Stage 3  CUDA online softmax, one query row / block
        flash         Stage 4  first fused FlashAttention kernel
        shared        Stage 5  vectorized loads + double buffering
        warp          Stage 6  warp-shuffle reductions
        mma           Stage 7  WMMA Tensor Cores
        flash2        Stage 8  FA2-style warp partitioning
    """
    if impl not in IMPLS:
        raise ValueError(f"unknown impl {impl!r}; choose from {IMPLS}")

    if impl == "reference":
        out = attention_reference(q, k, v, causal=causal, scale=scale)
        return (out, None) if return_lse else out
    if impl == "sdpa":
        out = attention_pytorch_sdpa(q, k, v, causal=causal, scale=scale)
        return (out, None) if return_lse else out
    if impl == "online_torch":
        out = online_softmax_torch(q, k, v, causal=causal, scale=scale)
        return (out, None) if return_lse else out

    from .extension import load_extension

    ext = load_extension()
    if impl == "naive":
        out, extra = ext.naive_fwd(q, k, v, causal, scale)
    elif impl == "tiled":
        out, extra = ext.tiled_fwd(q, k, v, causal, scale)
    elif impl == "online":
        out, extra = ext.online_fwd(q, k, v, causal, scale)
    elif impl == "flash":
        out, extra = ext.flash_fwd(q, k, v, causal, scale)
    elif impl == "shared":
        out, extra = ext.flash_shared_fwd(q, k, v, causal, scale, 32, 32)
    elif impl == "warp":
        out, extra = ext.flash_warp_fwd(q, k, v, causal, scale)
    elif impl == "mma":
        out, extra = ext.flash_mma_fwd(q, k, v, causal, scale)
    elif impl == "flash2":
        out, extra = ext.flash2_fwd(q, k, v, causal, scale)
    else:
        raise ValueError(impl)

    return (out, extra) if return_lse else out


def attention_with_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    scale: Optional[float] = None,
    impl: str = "flash",
) -> torch.Tensor:
    """Forward + autograd backward through the fused CUDA kernels."""
    if impl not in FUSED_IMPLS:
        raise ValueError(f"autograd is implemented for fused impls {FUSED_IMPLS}")
    return FlashAttentionFn.apply(q, k, v, causal, scale, impl)
