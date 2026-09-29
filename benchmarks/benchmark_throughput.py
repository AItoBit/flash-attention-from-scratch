"""Throughput (tokens/s, TFLOP/s) vs batch and sequence length."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from flash_attn_scratch.attention import attention


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", default="flash")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "throughput.csv")
    args = p.parse_args()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for B in args.batches:
        for N in args.seq:
            q = torch.randn(B, args.heads, N, args.D, device="cuda", dtype=dtype)
            k, v = torch.randn_like(q), torch.randn_like(q)
            for _ in range(5):
                attention(q, k, v, impl=args.impl)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 20
            for _ in range(iters):
                attention(q, k, v, impl=args.impl)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / iters * 1e3
            toks = B * N / (ms * 1e-3)
            tflops = 4 * B * args.heads * N * N * args.D / (ms * 1e-3) / 1e12
            print(f"B={B:<3} N={N:<5} {ms:8.3f} ms  {tflops:6.2f} TFLOP/s  {toks:10.0f} tok/s")
            rows.append(dict(B=B, N=N, ms=ms, tflops=tflops, tokens_per_s=toks, impl=args.impl))
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
