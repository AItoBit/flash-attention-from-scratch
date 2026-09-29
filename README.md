# FlashAttention From Scratch

CUDA implementation and **optimization study** of exact scaled dot-product attention.

```text
Reference Attention
      ↓
Naive CUDA              (materializes N×N)
      ↓
Tiled CUDA              (shared-memory GEMM, still N×N)
      ↓
Online Softmax          (the recurrence)
      ↓
Fused FlashAttention    (no N×N in HBM)
      ↓
Shared-memory optimized
      ↓
Warp-optimized
      ↓
Tensor Core / MMA
      ↓
FlashAttention-2-style scheduling
      ↓
Production benchmark suite
```

This repository is not “I implemented FlashAttention.” It is a
**progressive laboratory**: every version is correct, benchmarked, and
compared to the previous one, so the reason FlashAttention is fast is
visible in measurements rather than in comments.

```text
kernel-forge
      ↓
flash-attention-from-scratch     ← you are here
      ↓
mini-triton
      ↓
tiny-vllm
```

---

## Why each stage exists

| Kernel | What changed | Why it should be faster | Typical bound |
|---|---|---|---|
| PyTorch naive | oracle | — | CPU / eager GPU |
| CUDA naive | three kernels, `S,P` in HBM | — | **DRAM**, `O(N²)` traffic |
| CUDA tiled | tiled `QKᵀ` | Q/K reuse on-chip | still DRAM (`N²` write) |
| Online softmax | running `(m, ℓ, O)` | no `N×N` | under-occupied |
| Flash v1 (fused) | Q tiles + fused QK/softmax/PV | HBM = `Q,K,V,O` | SM / DRAM mix |
| Shared-memory | vectorized loads, double buffer | load efficiency | SM / smem |
| Warp | shuffle reductions | less `__syncthreads` | latency hiding |
| Tensor Core | WMMA GEMMs | math throughput | softmax round-trip |
| FA2-style | warp-owned Q rows | scheduling / occupancy | SM |

Run numbers on your GPU — the table below is the **schema**, not a fake
leaderboard. Fill it with `python benchmarks/benchmark_latency.py`.

| Kernel | Latency | Speedup | Memory | TFLOP/s |
|---|---:|---:|---:|---:|
| PyTorch naive / SDPA | | 1.0× | | |
| CUDA naive | | | `O(N²)` | |
| CUDA tiled | | | `O(N²)` | |
| Flash v1 | | | `O(N)` | |
| Flash Tensor Core | | | `O(N)` | |

Illustrative shape of the result (not a substitute for a local run):

```text
Latency

        naive
          │
          │
          │       PyTorch SDPA
          │      /
          │    /
          │  /
Flash ────┼────────────
          │
          └────────────────  sequence length
```

```text
Standard Attention     memory ∝ N²
Flash Attention        memory ∝ N
```

---

## Quick start

```bash
pip install -r requirements.txt
# first CUDA call JIT-compiles the kernels (needs nvcc + MSVC/g++)
python -c "from flash_attn_scratch import attention; print('ok')"
```

From the repo root, with `PYTHONPATH=python`:

```python
import torch
from flash_attn_scratch import attention, attention_reference

q = k = v = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.float16)
ref = attention_reference(q, k, v, causal=True)
out = attention(q, k, v, causal=True, impl="flash")
print((ref.float() - out.float()).abs().max())
```

`impl` selects the stage:

```text
reference | sdpa | online_torch | naive | tiled | online | flash | shared | warp | mma | flash2
```

### Build the standalone CUDA library

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release
./build/flash_check
./build/flash_bench 1 8 512 64
```

Windows (Visual Studio 2022/2026 + CUDA 13):

```powershell
.\scripts\build.ps1
.\scripts\test.ps1
```

CUDA 13.x does not officially support MSVC 14.50. The CMake / setup
flags already pass `-allow-unsupported-compiler`. You still need a
working NVIDIA driver to **run** the kernels; compiling only needs `nvcc`.

### Tests / benchmarks / profiles

```bash
pytest tests -q
python benchmarks/benchmark_latency.py --seq 128 256 512 1024 --dtype fp16
python benchmarks/benchmark_memory.py
python benchmarks/benchmark_pytorch.py --N 1024
python profiling/scripts/ncu_attention.py
```

---

## Architecture (five layers, none hidden)

```text
┌──────────────────────────────┐
│ Python / PyTorch interface   │  python/flash_attn_scratch/
├──────────────────────────────┤
│ Attention algorithms         │  reference, online softmax, fused
├──────────────────────────────┤
│ Kernel scheduling            │  v4 tiles, v8 warp partition, autotune
├──────────────────────────────┤
│ CUDA memory + warp primitives│  v2/v5 smem, v6 shuffles
├──────────────────────────────┤
│ Tensor Core / MMA hardware   │  v7 WMMA
└──────────────────────────────┘
```

```text
flash-attention-from-scratch/
├── docs/                      # math, memory, derivations, methodology
├── include/                   # C++ API
├── src/v1_naive … v8_flash2   # one directory per stage
├── src/backward/              # dQ, dK, dV
├── python/flash_attn_scratch/ # dispatcher, autotuner, roofline
├── tests/
├── benchmarks/
└── profiling/
```

---

## Correctness

Every CUDA kernel is checked against the Stage-0 PyTorch oracle
(fp32 accumulation):

- forward output, causal masking, odd `N`, `D ∈ {32,64,128}`
- fp16 / bf16 / fp32
- large logits (`Q,K *= 20`) — naive `exp` overflows, online softmax must not
- backward vs `torch.autograd` on the reference graph

```bash
pytest tests/test_forward.py tests/test_causal.py tests/test_backward.py -q
```

---

## Autotuner

`python/flash_attn_scratch/autotune.py` searches

```text
(BM, BN) ∈ {(32,32), (64,32), (32,64), (64,64)}
```

and records `{architecture, dtype, N, D, config, latency}`. Winner:

\[
C^*(shape) = \arg\min_C \mathrm{latency}(C, shape)
\]

That is the start of the compiler/runtime story this project is meant
to lead into (`mini-triton`).

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/attention_math.md](docs/attention_math.md) | SDPA, FLOPs vs bytes, causal mask |
| [docs/gpu_memory_model.md](docs/gpu_memory_model.md) | HBM / smem / coalescing / occupancy |
| [docs/online_softmax.md](docs/online_softmax.md) | the recurrence |
| [docs/flash_attention_derivation.md](docs/flash_attention_derivation.md) | fused forward |
| [docs/backward_derivation.md](docs/backward_derivation.md) | `dQ, dK, dV` |
| [docs/tensor_cores.md](docs/tensor_cores.md) | WMMA and the softmax gap |
| [docs/optimization_journey.md](docs/optimization_journey.md) | what to expect in Nsight |
| [docs/benchmark_methodology.md](docs/benchmark_methodology.md) | how to measure |

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
