"""Roofline model for the attention kernels in this repository.

Arithmetic intensity:

    AI = FLOPs / bytes_moved

Standard attention (materialized S, P):

    FLOPs  ≈  4 B H N² D
    bytes  ≈  2 B H N² e_acc   (write S, write P)   +  4 B H N D e   (Q,K,V,O)
             + reads of Q,K for QK and P,V for PV

FlashAttention:

    FLOPs  ≈  4 B H N² D          (same arithmetic)
    bytes  ≈  4 B H N D e         (Q,K,V,O once, plus extra KV rereads
                                   proportional to N/Bc query-tile passes)

The reread factor is ceil(N / BM): each KV tile is loaded once per Q tile.
That is still O(N² / tile) loads of D-wide rows — far less than O(N²) HBM
traffic for the attention matrix itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RooflinePoint:
    name: str
    flops: float
    bytes: float
    ai: float
    bound: str


def attention_flops(B: int, H: int, N: int, D: int) -> float:
    return 4.0 * B * H * N * N * D


def naive_bytes(B: int, H: int, N: int, D: int, elem: int = 2) -> float:
    # Q,K,V,O + fp32 S + fp32 P
    return float(4 * B * H * N * D * elem + 2 * B * H * N * N * 4)


def flash_bytes(B: int, H: int, N: int, D: int, BM: int = 32, elem: int = 2) -> float:
    q_tiles = (N + BM - 1) // BM
    # Q,O once; K,V reread for every query tile
    return float(B * H * (2 * N * D + 2 * q_tiles * N * D) * elem)


def classify(ai: float, ridge: float) -> str:
    return "compute-bound" if ai > ridge else "memory-bound"


def points(B: int, H: int, N: int, D: int, peak_flops: float, peak_bw: float,
           elem: int = 2) -> list[RooflinePoint]:
    ridge = peak_flops / peak_bw
    flops = attention_flops(B, H, N, D)
    out = []
    for name, nbytes in (
        ("naive", naive_bytes(B, H, N, D, elem)),
        ("flash", flash_bytes(B, H, N, D, 32, elem)),
        ("flash BM=64", flash_bytes(B, H, N, D, 64, elem)),
    ):
        ai = flops / nbytes
        out.append(RooflinePoint(name, flops, nbytes, ai, classify(ai, ridge)))
    return out


# Peak numbers are placeholders — replace with `nvidia-smi` / datasheet values.
GPU_PEAKS = {
    "rtx_4090": {"flops_fp16_tc": 330e12, "hbm_bw": 1.0e12},
    "rtx_3090": {"flops_fp16_tc": 142e12, "hbm_bw": 936e9},
    "a100_80g": {"flops_fp16_tc": 312e12, "hbm_bw": 2.0e12},
    "h100_sxm": {"flops_fp16_tc": 989e12, "hbm_bw": 3.35e12},
}
