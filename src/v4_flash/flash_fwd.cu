#include "attention.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

#include <stdexcept>

namespace flash {
namespace {

// First fused FlashAttention kernel.
// Grid: (ceil(N / BM), B*H)
// Each block owns a query tile Q_i and streams K/V tiles.
// Scores live only in shared memory — HBM never sees N×N.

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int kThreads = 128;
constexpr int kMaxD = 128;

template <typename T>
__global__ void flash_fwd_kernel(const T* __restrict__ Q, const T* __restrict__ K,
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
  float* Qs = smem;                         // [BM * D]
  float* Ks = Qs + BM * D;                  // [BN * D]
  float* Vs = Ks + BN * D;                  // [BN * D]
  float* S = Vs + BN * D;                   // [BM * BN]
  float* Oacc = S + BM * BN;                // [BM * D]
  float* mi = Oacc + BM * D;                // [BM]
  float* li = mi + BM;                      // [BM]
  float* alpha = li + BM;                   // [BM]

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
  __syncthreads();

  for (int j0 = 0; j0 < N; j0 += BN) {
    const int i_max = min(BM, N - i0) - 1 + i0;
    if (causal && j0 > i_max) break;

    const int j_lim = min(BN, N - j0);
    for (int t = threadIdx.x; t < BN * D; t += kThreads) {
      const int n = t / D;
      const int d = t % D;
      if (n < j_lim) {
        Ks[t] = to_f32(Kbh[(j0 + n) * D + d]);
        Vs[t] = to_f32(Vbh[(j0 + n) * D + d]);
      } else {
        Ks[t] = 0.f;
        Vs[t] = 0.f;
      }
    }
    __syncthreads();

    for (int t = threadIdx.x; t < BM * BN; t += kThreads) {
      const int m = t / BN;
      const int n = t % BN;
      const int i = i0 + m;
      const int j = j0 + n;
      float s = -INFINITY;
      if (i < N && n < j_lim && !causal_mask(i, j, causal)) {
        s = 0.f;
        const float* q = Qs + m * D;
        const float* k = Ks + n * D;
        for (int d = 0; d < D; ++d) s += q[d] * k[d];
        s *= scale;
      }
      S[t] = s;
    }
    __syncthreads();

    if (threadIdx.x < BM) {
      const int m = threadIdx.x;
      float row_m = -INFINITY;
#pragma unroll
      for (int n = 0; n < BN; ++n) row_m = fmaxf(row_m, S[m * BN + n]);
      const float m_new = fmaxf(mi[m], row_m);
      const float a = (mi[m] == -INFINITY) ? 0.f : expf(mi[m] - m_new);
      float l_tile = 0.f;
#pragma unroll
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
      const float* prow = S + m * BN;
      for (int n = 0; n < j_lim; ++n) acc += prow[n] * Vs[n * D + d];
      Oacc[t] = acc;
    }
    __syncthreads();
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
      const float m = mi[threadIdx.x];
      LSE[static_cast<int64_t>(bh) * N + i] =
          (l > 0.f) ? (m + logf(l)) : -INFINITY;
    }
  }
}

}  // namespace

void launch_flash_fwd(const void* Q, const void* K, const void* V, void* O,
                      float* LSE, int B, int H, int N, int D, float scale,
                      bool causal, DType dt, cudaStream_t stream) {
  if (D > kMaxD) {
    throw std::runtime_error("flash_fwd supports D <= 128");
  }
  dim3 grid(ceil_div(N, BM), B * H);
  const size_t smem = sizeof(float) * (static_cast<size_t>(BM) * D * 2 +  // Q, O
                                       static_cast<size_t>(BN) * D * 2 +  // K, V
                                       static_cast<size_t>(BM) * BN +     // S
                                       static_cast<size_t>(BM) * 3);      // m,l,a
  auto go = [&](auto dummy) {
    using T = decltype(dummy);
    flash_fwd_kernel<T><<<grid, kThreads, smem, stream>>>(
        static_cast<const T*>(Q), static_cast<const T*>(K),
        static_cast<const T*>(V), static_cast<T*>(O), LSE, N, D, scale, causal);
  };
  switch (dt) {
    case DType::F32:
      go(float{});
      break;
    case DType::F16:
      go(half{});
      break;
    case DType::BF16:
      go(nv_bfloat16{});
      break;
  }
  FLASH_CUDA_CHECK(cudaGetLastError());
}

}  // namespace flash
