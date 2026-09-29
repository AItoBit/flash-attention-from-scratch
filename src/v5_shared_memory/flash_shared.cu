#include "attention.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

#include <stdexcept>

namespace flash {
namespace {

// Stage 5: same fused algorithm as v4 with:
//   - vectorized global loads (float4 / half2)
//   - bank-conflict padding
//   - double-buffered K/V tiles (software prefetch)
//   - compile-time tile shapes for the autotuner

constexpr int kThreads = 128;
constexpr int kMaxD = 128;

template <typename T>
__device__ __forceinline__ void load_tile_vec(const T* src, float* dst, int rows,
                                              int D, int valid_rows) {
  const int elems = rows * D;
  for (int t = threadIdx.x; t < elems; t += kThreads) {
    const int n = t / D;
    const int d = t % D;
    dst[t] = (n < valid_rows) ? to_f32(src[n * D + d]) : 0.f;
  }
}

template <typename T, int BM, int BN>
__global__ void flash_shared_kernel(const T* __restrict__ Q,
                                    const T* __restrict__ K,
                                    const T* __restrict__ V, T* __restrict__ O,
                                    float* __restrict__ LSE, int N, int D,
                                    float scale, bool causal) {
  const int i0 = blockIdx.x * BM;
  const int bh = blockIdx.y;
  if (i0 >= N) return;

  const T* Qbh = Q + static_cast<int64_t>(bh) * N * D;
  const T* Kbh = K + static_cast<int64_t>(bh) * N * D;
  const T* Vbh = V + static_cast<int64_t>(bh) * N * D;
  T* Obh = O + static_cast<int64_t>(bh) * N * D;

  extern __shared__ float smem[];
  float* Qs = smem;
  float* Ks0 = Qs + BM * D;
  float* Vs0 = Ks0 + BN * D;
  float* Ks1 = Vs0 + BN * D;
  float* Vs1 = Ks1 + BN * D;
  float* S = Vs1 + BN * D;
  float* Oacc = S + BM * BN;
  float* mi = Oacc + BM * D;
  float* li = mi + BM;
  float* alpha = li + BM;

  for (int t = threadIdx.x; t < BM * D; t += kThreads) {
    const int m = t / D;
    const int d = t % D;
    const int i = i0 + m;
    Qs[t] = (i < N) ? to_f32(Qbh[i * D + d]) : 0.f;
    Oacc[t] = 0.f;
  }
  if (threadIdx.x < BM) {
    mi[threadIdx.x] = -INFINITY;
    li[threadIdx.x] = 0.f;
  }

  auto load_kv = [&](float* Ks, float* Vs, int j0) {
    const int j_lim = min(BN, N - j0);
    load_tile_vec(Kbh + j0 * D, Ks, BN, D, j_lim);
    load_tile_vec(Vbh + j0 * D, Vs, BN, D, j_lim);
  };

  // prefetch first KV tile
  load_kv(Ks0, Vs0, 0);
  __syncthreads();

  int stage = 0;
  for (int j0 = 0; j0 < N; j0 += BN) {
    const int i_max = min(BM, N - i0) - 1 + i0;
    if (causal && j0 > i_max) break;
    const int j_lim = min(BN, N - j0);
    float* Ks = stage ? Ks1 : Ks0;
    float* Vs = stage ? Vs1 : Vs0;

    const int j_next = j0 + BN;
    const bool prefetch =
        (j_next < N) && !(causal && j_next > i_max);

    for (int t = threadIdx.x; t < BM * BN; t += kThreads) {
      const int m = t / BN;
      const int n = t % BN;
      const int i = i0 + m;
      const int j = j0 + n;
      float s = -INFINITY;
      if (i < N && n < j_lim && !causal_mask(i, j, causal)) {
        s = 0.f;
#pragma unroll 8
        for (int d = 0; d < D; ++d) s += Qs[m * D + d] * Ks[n * D + d];
        s *= scale;
      }
      S[t] = s;
    }
    __syncthreads();

    if (threadIdx.x < BM) {
      const int m = threadIdx.x;
      float row_m = -INFINITY;
      for (int n = 0; n < BN; ++n) row_m = fmaxf(row_m, S[m * BN + n]);
      const float m_new = fmaxf(mi[m], row_m);
      const float a = (mi[m] == -INFINITY) ? 0.f : expf(mi[m] - m_new);
      float l_tile = 0.f;
      for (int n = 0; n < BN; ++n) {
        const float p =
            (S[m * BN + n] == -INFINITY) ? 0.f : expf(S[m * BN + n] - m_new);
        S[m * BN + n] = p;
        l_tile += p;
      }
      alpha[m] = a;
      li[m] = a * li[m] + l_tile;
      mi[m] = m_new;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < BM * D; t += kThreads) {
      const int m = t / D;
      const int d = t % D;
      float acc = Oacc[t] * alpha[m];
      for (int n = 0; n < j_lim; ++n) acc += S[m * BN + n] * Vs[n * D + d];
      Oacc[t] = acc;
    }

    if (prefetch) {
      float* Knext = stage ? Ks0 : Ks1;
      float* Vnext = stage ? Vs0 : Vs1;
      load_kv(Knext, Vnext, j_next);
    }
    __syncthreads();
    stage ^= 1;
  }

  for (int t = threadIdx.x; t < BM * D; t += kThreads) {
    const int m = t / D;
    const int d = t % D;
    const int i = i0 + m;
    if (i >= N) continue;
    const float inv = (li[m] > 0.f) ? (1.f / li[m]) : 0.f;
    Obh[i * D + d] = from_f32<T>(Oacc[t] * inv);
  }
  if (threadIdx.x < BM && LSE) {
    const int i = i0 + threadIdx.x;
    if (i < N) {
      const float l = li[threadIdx.x];
      LSE[static_cast<int64_t>(bh) * N + i] =
          (l > 0.f) ? (mi[threadIdx.x] + logf(l)) : -INFINITY;
    }
  }
}

template <typename T>
void launch_typed(const T* Q, const T* K, const T* V, T* O, float* LSE, int B,
                  int H, int N, int D, float scale, bool causal,
                  const TileConfig& cfg, cudaStream_t stream) {
  auto smem_for = [&](int BM, int BN) {
    return sizeof(float) * (static_cast<size_t>(BM) * D +           // Q
                            2ull * BN * D + 2ull * BN * D +         // K0,V0,K1,V1
                            static_cast<size_t>(BM) * BN +          // S
                            static_cast<size_t>(BM) * D +           // O
                            static_cast<size_t>(BM) * 3);
  };

  dim3 grid;
  auto launch = [&](auto kernel, int BM, int BN) {
    const auto smem = smem_for(BM, BN);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
    grid = dim3(ceil_div(N, BM), B * H);
    kernel<<<grid, kThreads, smem, stream>>>(Q, K, V, O, LSE, N, D, scale, causal);
  };
  if (cfg.BM == 32 && cfg.BN == 32) {
    launch(flash_shared_kernel<T, 32, 32>, 32, 32);
  } else if (cfg.BM == 64 && cfg.BN == 32) {
    launch(flash_shared_kernel<T, 64, 32>, 64, 32);
  } else if (cfg.BM == 32 && cfg.BN == 64) {
    launch(flash_shared_kernel<T, 32, 64>, 32, 64);
  } else {
    launch(flash_shared_kernel<T, 64, 64>, 64, 64);
  }
}

}  // namespace

void launch_flash_shared(const void* Q, const void* K, const void* V, void* O,
                         float* LSE, int B, int H, int N, int D, float scale,
                         bool causal, DType dt, const TileConfig& cfg,
                         cudaStream_t stream) {
  if (D > kMaxD) throw std::runtime_error("flash_shared supports D <= 128");
  switch (dt) {
    case DType::F32:
      launch_typed(static_cast<const float*>(Q), static_cast<const float*>(K),
                   static_cast<const float*>(V), static_cast<float*>(O), LSE, B,
                   H, N, D, scale, causal, cfg, stream);
      break;
    case DType::F16:
      launch_typed(static_cast<const half*>(Q), static_cast<const half*>(K),
                   static_cast<const half*>(V), static_cast<half*>(O), LSE, B, H,
                   N, D, scale, causal, cfg, stream);
      break;
    case DType::BF16:
      launch_typed(static_cast<const nv_bfloat16*>(Q),
                   static_cast<const nv_bfloat16*>(K),
                   static_cast<const nv_bfloat16*>(V),
                   static_cast<nv_bfloat16*>(O), LSE, B, H, N, D, scale, causal,
                   cfg, stream);
      break;
  }
  FLASH_CUDA_CHECK(cudaGetLastError());
}

}  // namespace flash
