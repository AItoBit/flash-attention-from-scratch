from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

_EXT = None
_LOAD_ERROR: Optional[BaseException] = None


def _sources() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    files = [
        "python/bindings.cpp",
        "src/v1_naive/qk.cu",
        "src/v1_naive/softmax.cu",
        "src/v1_naive/pv.cu",
        "src/v2_tiled/attention_tiled.cu",
        "src/v3_online_softmax/online_softmax.cu",
        "src/v4_flash/flash_fwd.cu",
        "src/v5_shared_memory/flash_shared.cu",
        "src/v6_warp/flash_warp.cu",
        "src/v7_tensorcore/flash_mma.cu",
        "src/v8_flash2/flash2.cu",
        "src/backward/flash_bwd.cu",
    ]
    return [str(root / f) for f in files]


def load_extension(verbose: bool = False) -> Any:
    """Load the CUDA extension, JIT-compiling it on first use if needed."""
    global _EXT, _LOAD_ERROR
    if _EXT is not None:
        return _EXT
    if _LOAD_ERROR is not None:
        raise RuntimeError(f"CUDA extension previously failed to load: {_LOAD_ERROR}") from _LOAD_ERROR

    try:
        from flash_attn_scratch import _C  # type: ignore

        _EXT = _C
        return _EXT
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        _LOAD_ERROR = exc
        raise RuntimeError("PyTorch is required to build the CUDA extension.") from exc

    root = Path(__file__).resolve().parents[2]
    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "-allow-unsupported-compiler",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    ]
    cxx_flags = ["/O2", "/utf-8"] if sys.platform == "win32" else ["-O3"]
    build_dir = root / "build_ext"
    build_dir.mkdir(exist_ok=True)
    try:
        _EXT = load(
            name="flash_attn_scratch_ext",
            sources=_sources(),
            extra_cflags=cxx_flags,
            extra_cuda_cflags=nvcc_flags,
            extra_include_paths=[str(root / "include")],
            build_directory=str(build_dir),
            verbose=verbose or os.environ.get("FLASH_ATTN_VERBOSE") == "1",
        )
    except Exception as exc:  # noqa: BLE001
        _LOAD_ERROR = exc
        raise RuntimeError(
            "Failed to JIT-compile CUDA kernels. Need nvcc + a C++ compiler "
            "(MSVC on Windows, g++/clang on Linux). "
            f"Original error: {exc}"
        ) from exc
    return _EXT


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
