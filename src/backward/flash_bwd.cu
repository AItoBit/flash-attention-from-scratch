#include "attention.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

#include <stdexcept>
#include <type_traits>

namespace flash {
namespace {

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int kThreads = 128;
constexpr int kMaxD = 128;

template <typename T>
__global__ void delta_kernel(const T* __restrict__ O, const T* __restrict__ dO,
                             float* __restrict__ Delta, int N, int D) {
  const int i = blockIdx.x;
  const int bh = blockIdx.y;
  if (i >= N) return;
  const T* o = O + (static_cast<int64_t>(bh) * N + i) * D;
  const T* go = dO + (static_cast<int64_t>(bh) * N + i) * D;
  __shared__ float red[kThreads / 32];
  float sum = 0.f;
  for (int d = threadIdx.x; d < D; d += kThreads) {
    sum += to_f32(o[d]) * to_f32(go[d]);
  }
  sum = block_reduce_sum<kThreads>(sum, red);
  if (threadIdx.x == 0) Delta[static_cast<int64_t>(bh) * N + i] = sum;
}

template <typename T>
__device__ __forceinline__ void atomic_add_t(T* addr, float val) {
  if constexpr (std::is_same<T, float>::value) {
    atomicAdd(addr, val);
  } else {
    atomicAdd(addr, from_f32<T>(val));
  }
}

template <typename T>
__global__ void flash_bwd_kernel(const T* __restrict__ Q, const T* __restrict__ K,
                                 const T* __restrict__ V, const T* __restrict__ dO,
                                 const float* __restrict__ LSE,
                                 const float* __restrict__ Delta, T* __restrict__ dQ,
                                 T* __restrict__ dK, T* __restrict__ dV, int N,
                                 int D, float scale, bool causal) {
  const int i0 = blockIdx.x * BM;
  const int bh = blockIdx.y;
  if (i0 >= N) return;

  const T* Qbh = Q + static_cast<int64_t>(bh) * N * D;
  const T* Kbh = K + static_cast<int64_t>(bh) * N * D;
  const T* Vbh = V + static_cast<int64_t>(bh) * N * D;
  const T* dObh = dO + static_cast<int64_t>(bh) * N * D;
  T* dQbh = dQ + static_cast<int64_t>(bh) * N * D;
  T* dKbh = dK + static_cast<int64_t>(bh) * N * D;
  T* dVbh = dV + static_cast<int64_t>(bh) * N * D;
  const float* LSEbh = LSE + static_cast<int64_t>(bh) * N;
  const float* Dbh = Delta + static_cast<int64_t>(bh) * N;

  extern __shared__ float smem[];
  float* Qs = smem;
  float* dOs = Qs + BM * D;
  float* Ks = dOs + BM * D;
  float* Vs = Ks + BN * D;
  float* S = Vs + BN * D;
  float* dP = S + BM * BN;
  float* dQs = dP + BM * BN;
  float* Li = dQs + BM * D;
  float* Di = Li + BM;

  for (int t = threadIdx.x; t < BM * D; t += kThreads) {
    const int m = t / D;
    const int d = t % D;
    const int i = i0 + m;
    Qs[t] = (i < N) ? to_f32(Qbh[i * D + d]) : 0.f;
    dOs[t] = (i < N) ? to_f32(dObh[i * D + d]) : 0.f;
    dQs[t] = 0.f;
  }
  if (threadIdx.x < BM) {
    const int i = i0 + threadIdx.x;
    Li[threadIdx.x] = (i < N) ? LSEbh[i] : 0.f;
    Di[threadIdx.x] = (i < N) ? Dbh[i] : 0.f;
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

    // Recompute P = exp(QK^T * scale - LSE)
    for (int t = threadIdx.x; t < BM * BN; t += kThreads) {
      const int m = t / BN;
      const int n = t % BN;
      const int i = i0 + m;
      const int j = j0 + n;
      float s = 0.f;
      if (i < N && n < j_lim && !causal_mask(i, j, causal)) {
        for (int d = 0; d < D; ++d) s += Qs[m * D + d] * Ks[n * D + d];
        s = expf(s * scale - Li[m]);
      } else {
        s = 0.f;
      }
      S[t] = s;
    }
    __syncthreads();

    // dV += P^T @ dO   and   dP = dO @ V^T
    for (int t = threadIdx.x; t < BN * D; t += kThreads) {
      const int n = t / D;
      const int d = t % D;
      if (n >= j_lim) continue;
      float acc = 0.f;
      for (int m = 0; m < BM; ++m) {
        if (i0 + m < N) acc += S[m * BN + n] * dOs[m * D + d];
      }
      atomic_add_t(&dVbh[(j0 + n) * D + d], acc);
    }

    for (int t = threadIdx.x; t < BM * BN; t += kThreads) {
      const int m = t / BN;
      const int n = t % BN;
      float acc = 0.f;
      if (i0 + m < N && n < j_lim) {
        for (int d = 0; d < D; ++d) acc += dOs[m * D + d] * Vs[n * D + d];
      }
      dP[t] = acc;
    }
    __syncthreads();

    // dS = P ⊙ (dP - D_i)
    for (int t = threadIdx.x; t < BM * BN; t += kThreads) {
      const int m = t / BN;
      dP[t] = S[t] * (dP[t] - Di[m]);
    }
    __syncthreads();

    // dQ += dS @ K
    for (int t = threadIdx.x; t < BM * D; t += kThreads) {
      const int m = t / D;
      const int d = t % D;
      float acc = 0.f;
      for (int n = 0; n < j_lim; ++n) acc += dP[m * BN + n] * Ks[n * D + d];
      dQs[t] += acc * scale;
    }

    // dK += dS^T @ Q
    for (int t = threadIdx.x; t < BN * D; t += kThreads) {
      const int n = t / D;
      const int d = t % D;
      if (n >= j_lim) continue;
      float acc = 0.f;
      for (int m = 0; m < BM; ++m) acc += dP[m * BN + n] * Qs[m * D + d];
      atomic_add_t(&dKbh[(j0 + n) * D + d], acc * scale);
    }
    __syncthreads();
  }

  for (int t = threadIdx.x; t < BM * D; t += kThreads) {
    const int m = t / D;
    const int d = t % D;
    const int i = i0 + m;
    if (i < N) dQbh[i * D + d] = from_f32<T>(dQs[t]);
  }
}

}  // namespace

void launch_flash_bwd(const void* Q, const void* K, const void* V, const void* O,
                      const void* dO, const float* LSE, void* dQ, void* dK,
                      void* dV, int B, int H, int N, int D, float scale,
                      bool causal, DType dt, cudaStream_t stream) {
  if (D > kMaxD) throw std::runtime_error("flash_bwd supports D <= 128");

  float* Delta = nullptr;
  FLASH_CUDA_CHECK(cudaMalloc(&Delta, sizeof(float) * static_cast<size_t>(B) * H * N));

  dim3 grid_delta(N, B * H);
  dim3 grid(ceil_div(N, BM), B * H);
  const size_t smem =
      sizeof(float) * (static_cast<size_t>(BM) * D * 3 +  // Q, dO, dQ
                       static_cast<size_t>(BN) * D * 2 +  // K, V
                       static_cast<size_t>(BM) * BN * 2 + // P, dS
                       static_cast<size_t>(BM) * 2);      // L, D

  auto go = [&](auto dummy) {
    using T = decltype(dummy);
    delta_kernel<T><<<grid_delta, kThreads, 0, stream>>>(
        static_cast<const T*>(O), static_cast<const T*>(dO), Delta, N, D);
    flash_bwd_kernel<T><<<grid, kThreads, smem, stream>>>(
        static_cast<const T*>(Q), static_cast<const T*>(K), static_cast<const T*>(V),
        static_cast<const T*>(dO), LSE, Delta, static_cast<T*>(dQ),
        static_cast<T*>(dK), static_cast<T*>(dV), N, D, scale, causal);
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
  if (stream) {
    FLASH_CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    FLASH_CUDA_CHECK(cudaDeviceSynchronize());
  }
  FLASH_CUDA_CHECK(cudaFree(Delta));
}

}  // namespace flash
