"""Full laboratory sweep: kernels × sequence lengths × dtypes.

Writes CSV under benchmarks/results/ and a matplotlib figure under plots/.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="short sequences only")
    args = p.parse_args()
    seq = [128, 256, 512, 1024] if args.quick else [128, 256, 512, 1024, 2048, 4096]
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks" / "benchmark_latency.py"),
        "--seq",
        *[str(n) for n in seq],
        "--impls",
        "sdpa",
        "naive",
        "tiled",
        "online",
        "flash",
        "shared",
        "warp",
        "mma",
        "flash2",
        "--dtype",
        "fp16",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
