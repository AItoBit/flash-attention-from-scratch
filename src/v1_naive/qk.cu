#include "attention.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

namespace flash {
namespace {

template <typename T>
__global__ void qk_naive_kernel(const T* __restrict__ Q, const T* __restrict__ K,
                                float* __restrict__ S, int B, int H, int N,
                                int D, float scale, bool causal) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = B * H * N * N;
  if (idx >= total) return;

  const int j = idx % N;
  const int i = (idx / N) % N;
  const int h = (idx / (N * N)) % H;
  const int b = idx / (N * N * H);

  if (causal && j > i) {
    S[idx] = -INFINITY;
    return;
  }

  const T* q = Q + ((b * H + h) * N + i) * D;
  const T* k = K + ((b * H + h) * N + j) * D;
  float acc = 0.f;
  for (int d = 0; d < D; ++d) {
    acc += to_f32(q[d]) * to_f32(k[d]);
  }
  S[idx] = acc * scale;
}

}  // namespace

void launch_qk_naive(const void* Q, const void* K, float* S, int B, int H,
                     int N, int D, float scale, bool causal, DType dt,
                     cudaStream_t stream) {
  const int total = B * H * N * N;
  const int threads = 256;
  const int blocks = ceil_div(total, threads);
  switch (dt) {
    case DType::F32:
      qk_naive_kernel<float><<<blocks, threads, 0, stream>>>(
          static_cast<const float*>(Q), static_cast<const float*>(K), S, B, H, N,
          D, scale, causal);
      break;
    case DType::F16:
      qk_naive_kernel<half><<<blocks, threads, 0, stream>>>(
          static_cast<const half*>(Q), static_cast<const half*>(K), S, B, H, N, D,
          scale, causal);
      break;
    case DType::BF16:
      qk_naive_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
          static_cast<const nv_bfloat16*>(Q), static_cast<const nv_bfloat16*>(K),
          S, B, H, N, D, scale, causal);
      break;
  }
  FLASH_CUDA_CHECK(cudaGetLastError());
}

}  // namespace flash
