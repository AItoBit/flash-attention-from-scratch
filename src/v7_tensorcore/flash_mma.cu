#include "attention.hpp"
#include "cuda_utils.cuh"
#include "mma_utils.cuh"
#include "utils.hpp"

#include <stdexcept>

namespace flash {
namespace {

// Tensor Core FlashAttention (WMMA 16×16×16).
// QK^T and PV run as MMA; softmax stays in fp32 registers/shared memory.
// This is the pedagogical point: softmax is not a matmul, so fragments must
// round-trip through a regular layout between the two GEMMs.

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int kThreads = 128;  // 4 warps: 2 along M × 2 along N
constexpr int kMaxD = 128;

template <typename T>
__device__ __forceinline__ half to_half_val(T x);

template <>
__device__ __forceinline__ half to_half_val<half>(half x) {
  return x;
}
template <>
__device__ __forceinline__ half to_half_val<float>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ half to_half_val<nv_bfloat16>(nv_bfloat16 x) {
  return __float2half(__bfloat162float(x));
}

template <typename T>
__global__ void flash_mma_kernel(const T* __restrict__ Q, const T* __restrict__ K,
                                 const T* __restrict__ V, T* __restrict__ O,
                                 float* __restrict__ LSE, int N, int D,
                                 float scale, bool causal) {
  const int i0 = blockIdx.x * BM;
  const int bh = blockIdx.y;
  if (i0 >= N) return;

  const int warp_id = threadIdx.x >> 5;
  const int warp_m = warp_id / 2;
  const int warp_n = warp_id % 2;

  const T* Qbh = Q + static_cast<int64_t>(bh) * N * D;
  const T* Kbh = K + static_cast<int64_t>(bh) * N * D;
  const T* Vbh = V + static_cast<int64_t>(bh) * N * D;
  T* Obh = O + static_cast<int64_t>(bh) * N * D;

  extern __shared__ unsigned char raw[];
  half* Qh = reinterpret_cast<half*>(raw);          // BM * D_pad
  const int Dp = (D + 15) / 16 * 16;
  half* Kh = Qh + BM * Dp;
  half* Vh = Kh + BN * Dp;
  half* Ph = Vh + BN * Dp;
  float* S = reinterpret_cast<float*>(Ph + BM * BN);
  float* Oacc = S + BM * BN;
  float* mi = Oacc + BM * Dp;
  float* li = mi + BM;
  float* alpha = li + BM;

  for (int t = threadIdx.x; t < BM * Dp; t += kThreads) {
    const int m = t / Dp;
    const int d = t % Dp;
    const int i = i0 + m;
    Qh[t] = (i < N && d < D) ? to_half_val(Qbh[i * D + d]) : __float2half(0.f);
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

    for (int t = threadIdx.x; t < BN * Dp; t += kThreads) {
      const int n = t / Dp;
      const int d = t % Dp;
      if (n < j_lim && d < D) {
        Kh[t] = to_half_val(Kbh[(j0 + n) * D + d]);
        Vh[t] = to_half_val(Vbh[(j0 + n) * D + d]);
      } else {
        Kh[t] = __float2half(0.f);
        Vh[t] = __float2half(0.f);
      }
    }
    __syncthreads();

    mma::mma_qk_tile(Qh, Kh, S, BM, BN, Dp, Dp, Dp, BN, warp_m, warp_n);
    __syncthreads();

    for (int t = threadIdx.x; t < BM * BN; t += kThreads) {
      const int m = t / BN;
      const int n = t % BN;
      const int i = i0 + m;
      const int j = j0 + n;
      float s = S[t] * scale;
      if (i >= N || n >= j_lim || causal_mask(i, j, causal)) s = -INFINITY;
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
        Ph[m * BN + n] = __float2half(p);
        l_tile += p;
      }
      alpha[m] = a;
      li[m] = a * li[m] + l_tile;
      mi[m] = m_new;
    }
    __syncthreads();

    for (int t = threadIdx.x; t < BM * Dp; t += kThreads) {
      const int m = t / Dp;
      Oacc[t] *= alpha[m];
    }
    __syncthreads();

    mma::mma_pv_tile(Ph, Vh, Oacc, BM, BN, Dp, BN, Dp, Dp, warp_m, warp_n);
    __syncthreads();
  }

  for (int t = threadIdx.x; t < BM * D; t += kThreads) {
    const int m = t / D;
    const int d = t % D;
    const int i = i0 + m;
    if (i >= N) continue;
    const float inv = (li[m] > 0.f) ? (1.f / li[m]) : 0.f;
    Obh[i * D + d] = from_f32<T>(Oacc[m * Dp + d] * inv);
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

void launch_flash_mma(const void* Q, const void* K, const void* V, void* O,
                      float* LSE, int B, int H, int N, int D, float scale,
                      bool causal, DType dt, cudaStream_t stream) {
  if (D > kMaxD) throw std::runtime_error("flash_mma supports D <= 128");
  const int Dp = (D + 15) / 16 * 16;
  dim3 grid(ceil_div(N, BM), B * H);
  const size_t smem =
      sizeof(half) * (static_cast<size_t>(BM) * Dp + 2ull * BN * Dp +
                      static_cast<size_t>(BM) * BN) +
      sizeof(float) * (static_cast<size_t>(BM) * BN + static_cast<size_t>(BM) * Dp +
                       static_cast<size_t>(BM) * 3);

  auto go = [&](auto dummy) {
    using T = decltype(dummy);
    flash_mma_kernel<T><<<grid, kThreads, smem, stream>>>(
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
