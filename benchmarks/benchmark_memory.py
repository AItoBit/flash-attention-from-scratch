"""Peak allocated CUDA memory for materializing vs fused kernels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from flash_attn_scratch.attention import attention


def peak_mb(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--D", type=int, default=64)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(f"{'N':>6} {'naive MB':>12} {'flash MB':>12} {'ratio':>8}  (S,P ~ N²)")
    for n in args.seq:
        q = torch.randn(args.B, args.H, n, args.D, device="cuda", dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        naive_n2 = args.B * args.H * n * n * 4 * 2 / (1024 ** 2)
        try:
            mb_naive = peak_mb(lambda: attention(q, k, v, impl="naive")) if n <= 2048 else float("nan")
        except Exception:
            mb_naive = float("nan")
        mb_flash = peak_mb(lambda: attention(q, k, v, impl="flash"))
        ratio = mb_naive / mb_flash if mb_flash else float("nan")
        print(f"{n:6d} {mb_naive:12.1f} {mb_flash:12.1f} {ratio:8.2f}   S+P≈{naive_n2:.1f}MB")


if __name__ == "__main__":
    main()
