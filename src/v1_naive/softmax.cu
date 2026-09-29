#include "attention.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

#include <cstdint>

namespace flash {
namespace {

constexpr int kThreads = 128;

__global__ void softmax_naive_kernel(const float* __restrict__ S,
                                     float* __restrict__ P, int N) {
  // grid: (N, B*H)  — one block per attention row
  const int row = blockIdx.x;
  const int bh = blockIdx.y;
  if (row >= N) return;

  const float* srow = S + (static_cast<int64_t>(bh) * N + row) * N;
  float* prow = P + (static_cast<int64_t>(bh) * N + row) * N;

  __shared__ float red[kThreads / 32];

  float m = -INFINITY;
  for (int j = threadIdx.x; j < N; j += kThreads) {
    m = fmaxf(m, srow[j]);
  }
  m = block_reduce_max<kThreads>(m, red);

  float l = 0.f;
  for (int j = threadIdx.x; j < N; j += kThreads) {
    const float e = (srow[j] == -INFINITY) ? 0.f : expf(srow[j] - m);
    prow[j] = e;
    l += e;
  }
  l = block_reduce_sum<kThreads>(l, red);
  const float inv = (l > 0.f) ? (1.f / l) : 0.f;

  for (int j = threadIdx.x; j < N; j += kThreads) {
    prow[j] *= inv;
  }
}

}  // namespace

void launch_softmax_naive(const float* S, float* P, int B, int H, int N,
                          cudaStream_t stream) {
  dim3 grid(N, B * H);
  softmax_naive_kernel<<<grid, kThreads, 0, stream>>>(S, P, N);
  FLASH_CUDA_CHECK(cudaGetLastError());
}

}  // namespace flash
