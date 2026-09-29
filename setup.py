from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
CSRC = [
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


def cuda_extension():
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build the CUDA extension.") from exc

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
    cxx_flags = ["-O3"]
    if sys.platform == "win32":
        cxx_flags = ["/O2", "/utf-8"]

    return CUDAExtension(
        name="flash_attn_scratch._C",
        sources=CSRC,
        include_dirs=[str(ROOT / "include")],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    )


ext_modules = []
cmdclass = {}
if os.environ.get("FLASH_ATTN_SKIP_CUDA", "0") != "1":
    from torch.utils.cpp_extension import BuildExtension

    ext_modules = [cuda_extension()]
    cmdclass = {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}


setup(
    name="flash-attention-from-scratch",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
