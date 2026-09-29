#include "attention.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

#include <stdexcept>

namespace flash {
namespace {

// Stage 6: warp-owned query rows, shuffle reductions, no block-wide softmax tree.
// 4 warps × 8 query rows = BM 32. Each lane in a warp owns one key column of BN=32.
// Row max / row sum use __shfl_xor_sync instead of shared-memory reductions.

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int kWarps = 4;
constexpr int kRowsPerWarp = BM / kWarps;  // 8
constexpr int kThreads = kWarps * 32;
constexpr int kMaxD = 128;

template <typename T>
__global__ void flash_warp_kernel(const T* __restrict__ Q, const T* __restrict__ K,
                                  const T* __restrict__ V, T* __restrict__ O,
                                  float* __restrict__ LSE, int N, int D,
                                  float scale, bool causal) {
  const int i0 = blockIdx.x * BM;
  const int bh = blockIdx.y;
  if (i0 >= N) return;

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const T* Qbh = Q + static_cast<int64_t>(bh) * N * D;
  const T* Kbh = K + static_cast<int64_t>(bh) * N * D;
  const T* Vbh = V + static_cast<int64_t>(bh) * N * D;
  T* Obh = O + static_cast<int64_t>(bh) * N * D;

  extern __shared__ float smem[];
  float* Qs = smem;                  // [BM * D]
  float* Ks = Qs + BM * D;           // [BN * D]
  float* Vs = Ks + BN * D;           // [BN * D]
  float* Oacc = Vs + BN * D;         // [BM * D]
  float* li = Oacc + BM * D;         // [BM]
  float* mi = li + BM;               // [BM]

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

    // Each warp owns kRowsPerWarp query rows. Lane n computes score vs key n.
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int m = warp_id * kRowsPerWarp + r;
      const int i = i0 + m;
      const int n = lane;
      const int j = j0 + n;

      float s = -INFINITY;
      if (i < N && n < j_lim && !causal_mask(i, j, causal)) {
        s = 0.f;
        const float* q = Qs + m * D;
        const float* k = Ks + n * D;
        for (int d = 0; d < D; ++d) s += q[d] * k[d];
        s *= scale;
      }

      const float row_m = warp_all_reduce_max(s);
      const float m_old = mi[m];
      const float m_new = fmaxf(m_old, row_m);
      const float a = (m_old == -INFINITY) ? 0.f : expf(m_old - m_new);
      const float p = (s == -INFINITY) ? 0.f : expf(s - m_new);
      const float l_tile = warp_all_reduce_sum(p);

      if (lane == 0) {
        li[m] = a * li[m] + l_tile;
        mi[m] = m_new;
      }

      // Rescale previous O and accumulate p * V[n]
      for (int d = lane; d < D; d += 32) {
        float acc = Oacc[m * D + d] * a;
        // Each lane holds p for its key; we must sum p*V over the warp.
        float pv = p * Vs[n * D + d];
        pv = warp_all_reduce_sum(pv);
        Oacc[m * D + d] = acc + pv;
      }
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
      LSE[static_cast<int64_t>(bh) * N + i] =
          (l > 0.f) ? (mi[threadIdx.x] + logf(l)) : -INFINITY;
    }
  }
}

}  // namespace

void launch_flash_warp(const void* Q, const void* K, const void* V, void* O,
                       float* LSE, int B, int H, int N, int D, float scale,
                       bool causal, DType dt, cudaStream_t stream) {
  if (D > kMaxD) throw std::runtime_error("flash_warp supports D <= 128");
  dim3 grid(ceil_div(N, BM), B * H);
  const size_t smem = sizeof(float) * (static_cast<size_t>(BM) * D * 2 +
                                       static_cast<size_t>(BN) * D * 2 +
                                       static_cast<size_t>(BM) * 2);
  auto go = [&](auto dummy) {
    using T = decltype(dummy);
    flash_warp_kernel<T><<<grid, kThreads, smem, stream>>>(
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
