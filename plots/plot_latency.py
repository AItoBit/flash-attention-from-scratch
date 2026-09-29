"""Plot latency vs sequence length from a latency.csv produced by the bench."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(csv_path: Path | None = None):
    csv_path = csv_path or ROOT / "benchmarks" / "results" / "latency.csv"
    if not csv_path.exists():
        print(f"no {csv_path}; run python benchmarks/benchmark_latency.py first")
        return 1
    series = defaultdict(list)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            series[row["impl"]].append((int(row["N"]), float(row["ms"])))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; printing series instead")
        for impl, pts in series.items():
            print(impl, sorted(pts))
        return 0
    for impl, pts in sorted(series.items()):
        pts = sorted(pts)
        plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=impl)
    plt.xlabel("sequence length N")
    plt.ylabel("latency (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = ROOT / "plots" / "latency.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
