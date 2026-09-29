"""Latency sweep across kernels, sequence lengths, and baselines."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from flash_attn_scratch.attention import IMPLS, attention
from flash_attn_scratch.reference import attention_pytorch_sdpa, attention_reference


def flops(B, H, N, D) -> float:
    return 4.0 * B * H * N * N * D


def bench(fn, warmup=10, iters=30) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def try_triton(q, k, v, causal):
    try:
        from torch.nn.functional import scaled_dot_product_attention as sdpa

        def run():
            return sdpa(q, k, v, is_causal=causal)

        return run
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--seq", type=int, nargs="+",
                   default=[128, 256, 512, 1024, 2048])
    p.add_argument("--impls", nargs="+",
                   default=["reference", "sdpa", "naive", "tiled", "online",
                            "flash", "shared", "warp", "mma", "flash2"])
    p.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--causal", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "latency.csv")
    args = p.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"device={device} dtype={args.dtype} B={args.B} H={args.H} D={args.D} causal={args.causal}")
    print(f"{'impl':<14} {'N':>6} {'ms':>10} {'TFLOP/s':>10} {'tok/s':>12}")
    for n in args.seq:
        q = torch.randn(args.B, args.H, n, args.D, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for impl in args.impls:
            if impl in ("naive", "tiled") and n > 2048:
                continue
            if impl not in ("reference", "sdpa", "online_torch") and device != "cuda":
                continue
            try:
                ms = bench(lambda impl=impl: attention(q, k, v, causal=args.causal, impl=impl),
                           warmup=5, iters=15) * 1e3
            except Exception as exc:  # noqa: BLE001
                print(f"{impl:<14} {n:>6}  FAILED ({exc})")
                continue
            tflops = flops(args.B, args.H, n, args.D) / (ms * 1e-3) / 1e12
            toks = args.B * n / (ms * 1e-3)
            print(f"{impl:<14} {n:>6} {ms:10.3f} {tflops:10.2f} {toks:12.0f}")
            rows.append(dict(impl=impl, N=n, ms=ms, tflops=tflops, tokens_per_s=toks,
                             B=args.B, H=args.H, D=args.D, dtype=args.dtype,
                             causal=args.causal))

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["impl"])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
