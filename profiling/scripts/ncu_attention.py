#!/usr/bin/env python3
"""Launch Nsight Compute against one impl if `ncu` is on PATH."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "profiling" / "ncu"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    impl = sys.argv[1] if len(sys.argv) > 1 else "flash"
    ncu = shutil.which("ncu")
    if not ncu:
        print("ncu not on PATH. Install Nsight Compute and re-run.")
        print("Metrics to capture are listed in profiling/scripts/README.md")
        return 0
    cmd = [
        ncu,
        "--set",
        "full",
        "-o",
        str(OUT / impl),
        sys.executable,
        str(ROOT / "benchmarks" / "benchmark_latency.py"),
        "--seq",
        "512",
        "--impls",
        impl,
        "--dtype",
        "fp16",
    ]
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
