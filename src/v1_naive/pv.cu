#include "attention.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

namespace flash {
namespace {

template <typename T>
__global__ void pv_naive_kernel(const float* __restrict__ P,
                                const T* __restrict__ V, T* __restrict__ O,
                                int B, int H, int N, int D) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = B * H * N * D;
  if (idx >= total) return;

  const int d = idx % D;
  const int i = (idx / D) % N;
  const int h = (idx / (D * N)) % H;
  const int b = idx / (D * N * H);

  const float* prow = P + ((b * H + h) * N + i) * N;
  float acc = 0.f;
  for (int j = 0; j < N; ++j) {
    acc += prow[j] * to_f32(V[((b * H + h) * N + j) * D + d]);
  }
  O[idx] = from_f32<T>(acc);
}

}  // namespace

void launch_pv_naive(const float* P, const void* V, void* O, int B, int H,
                     int N, int D, DType dt, cudaStream_t stream) {
  const int total = B * H * N * D;
  const int threads = 256;
  const int blocks = ceil_div(total, threads);
  switch (dt) {
    case DType::F32:
      pv_naive_kernel<float><<<blocks, threads, 0, stream>>>(
          P, static_cast<const float*>(V), static_cast<float*>(O), B, H, N, D);
      break;
    case DType::F16:
      pv_naive_kernel<half><<<blocks, threads, 0, stream>>>(
          P, static_cast<const half*>(V), static_cast<half*>(O), B, H, N, D);
      break;
    case DType::BF16:
      pv_naive_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
          P, static_cast<const nv_bfloat16*>(V), static_cast<nv_bfloat16*>(O), B,
          H, N, D);
      break;
  }
  FLASH_CUDA_CHECK(cudaGetLastError());
}

}  // namespace flash
