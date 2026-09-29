"""Optional Triton comparison. Falls back with a clear message if Triton is absent."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from flash_attn_scratch.attention import attention


def bench(fn, warmup=10, iters=30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--D", type=int, default=64)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    q = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    print("ours/flash  ", f"{bench(lambda: attention(q, k, v, impl='flash')):8.3f} ms")
    print("ours/mma    ", f"{bench(lambda: attention(q, k, v, impl='mma')):8.3f} ms")

    try:
        import triton  # noqa: F401
    except Exception:
        print("Triton is not installed. `pip install triton` on Linux + a recent GPU.")
        print("On Windows this comparison is expected to be unavailable.")
        return

    try:
        from torch.nn.functional import scaled_dot_product_attention as sdpa

        print("torch sdpa  ", f"{bench(lambda: sdpa(q, k, v)):8.3f} ms")
    except Exception as exc:  # noqa: BLE001
        print("torch sdpa failed", exc)


if __name__ == "__main__":
    main()
