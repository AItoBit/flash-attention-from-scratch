#include "attention.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

namespace flash {
namespace {

// Tiled QK^T: each thread block owns a BM x BN tile of S and walks D in BK
// chunks through shared memory. S still lands in HBM — this stage isolates
// tiling, not IO-awareness of the full attention algorithm.
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;

template <typename T>
__global__ void qk_tiled_kernel(const T* __restrict__ Q, const T* __restrict__ K,
                                float* __restrict__ S, int N, int D,
                                float scale, bool causal) {
  const int i0 = blockIdx.y * BM;
  const int j0 = blockIdx.x * BN;
  const int bh = blockIdx.z;

  const T* Qbh = Q + static_cast<int64_t>(bh) * N * D;
  const T* Kbh = K + static_cast<int64_t>(bh) * N * D;
  float* Sbh = S + static_cast<int64_t>(bh) * N * N;

  __shared__ float Qs[BM][BK + 1];  // +1 pads shared-memory banks
  __shared__ float Ks[BN][BK + 1];

  const int tx = threadIdx.x;  // 0..BN-1
  const int ty = threadIdx.y;  // 0..BM-1

  float acc = 0.f;
  const int i = i0 + ty;
  const int j = j0 + tx;
  const bool valid = (i < N) && (j < N);

  for (int k0 = 0; k0 < D; k0 += BK) {
    const int d_q = k0 + tx;
    if (ty < BM && tx < BK) {
      Qs[ty][tx] =
          (i < N && d_q < D) ? to_f32(Qbh[i * D + d_q]) : 0.f;
    }
    const int d_k = k0 + ty;
    if (tx < BN && ty < BK) {
      // Ks[key_row][k_inner]
      Ks[tx][ty] =
          (j < N && d_k < D) ? to_f32(Kbh[j * D + d_k]) : 0.f;
    }
    __syncthreads();

    const int k_lim = min(BK, D - k0);
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      if (kk < k_lim) acc += Qs[ty][kk] * Ks[tx][kk];
    }
    __syncthreads();
  }

  if (!valid) return;
  if (causal && j > i) {
    Sbh[i * N + j] = -INFINITY;
  } else {
    Sbh[i * N + j] = acc * scale;
  }
}

}  // namespace

void launch_qk_tiled(const void* Q, const void* K, float* S, int B, int H,
                     int N, int D, float scale, bool causal, DType dt,
                     cudaStream_t stream) {
  dim3 block(BN, BM);
  dim3 grid(ceil_div(N, BN), ceil_div(N, BM), B * H);
  switch (dt) {
    case DType::F32:
      qk_tiled_kernel<float><<<grid, block, 0, stream>>>(
          static_cast<const float*>(Q), static_cast<const float*>(K), S, N, D,
          scale, causal);
      break;
    case DType::F16:
      qk_tiled_kernel<half><<<grid, block, 0, stream>>>(
          static_cast<const half*>(Q), static_cast<const half*>(K), S, N, D,
          scale, causal);
      break;
    case DType::BF16:
      qk_tiled_kernel<nv_bfloat16><<<grid, block, 0, stream>>>(
          static_cast<const nv_bfloat16*>(Q), static_cast<const nv_bfloat16*>(K),
          S, N, D, scale, causal);
      break;
  }
  FLASH_CUDA_CHECK(cudaGetLastError());
}

}  // namespace flash
