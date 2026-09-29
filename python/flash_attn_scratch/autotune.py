from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import torch

from .attention import attention
from .extension import load_extension


@dataclass(frozen=True)
class Config:
    BM: int = 32
    BN: int = 32


DEFAULT_CONFIGS: tuple[Config, ...] = (
    Config(32, 32),
    Config(64, 32),
    Config(32, 64),
    Config(64, 64),
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_once(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool,
               cfg: Config, warmup: int = 5, iters: int = 20) -> float:
    ext = load_extension()
    for _ in range(warmup):
        ext.flash_shared_fwd(q, k, v, causal, None, cfg.BM, cfg.BN)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        ext.flash_shared_fwd(q, k, v, causal, None, cfg.BM, cfg.BN)
    _sync()
    return (time.perf_counter() - t0) / iters


def autotune(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    configs: Optional[Iterable[Config]] = None,
    warmup: int = 5,
    iters: int = 20,
) -> dict:
    """Exhaustive tile search: C*(shape) = argmin_C latency(C, shape).

    Returns the winning config plus the full table so the search itself is
    part of the laboratory, not a hidden heuristic.
    """
    configs = tuple(configs or DEFAULT_CONFIGS)
    rows = []
    best = None
    for cfg in configs:
        try:
            ms = bench_once(q, k, v, causal, cfg, warmup=warmup, iters=iters) * 1e3
            err = None
        except Exception as exc:  # noqa: BLE001
            ms = float("inf")
            err = str(exc)
        rec = {**asdict(cfg), "latency_ms": ms, "error": err}
        rows.append(rec)
        if best is None or ms < best["latency_ms"]:
            best = rec
    return {
        "shape": list(q.shape),
        "dtype": str(q.dtype),
        "causal": causal,
        "winner": best,
        "table": rows,
    }


def recommend(n: int, d: int) -> Config:
    """Cheap static heuristic used when you do not want to run the search."""
    if d >= 128 and n >= 2048:
        return Config(32, 64)
    if n >= 1024:
        return Config(64, 32)
    return Config(32, 32)


def demo_attention(q, k, v, causal=False):
    cfg = recommend(q.size(2), q.size(3))
    return attention(q, k, v, causal=causal, impl="shared")
