"""Compare against PyTorch SDPA (math / mem-efficient / flash backends)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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
    p.add_argument("--causal", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    q = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    backends = {
        "sdpa_default": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=args.causal),
    }

    for impl in ("naive", "flash", "warp", "mma", "flash2"):
        backends[f"ours_{impl}"] = lambda impl=impl: attention(
            q, k, v, causal=args.causal, impl=impl
        )

    print(f"B={args.B} H={args.H} N={args.N} D={args.D} causal={args.causal}")
    for name, fn in backends.items():
        try:
            ms = bench(fn)
            print(f"  {name:<22} {ms:8.3f} ms")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<22} FAILED ({exc})")


if __name__ == "__main__":
    main()
