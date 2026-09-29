#include "attention.hpp"
#include "cuda_utils.cuh"
#include "utils.hpp"

namespace flash {
namespace {

// Stage 3 isolates the online-softmax recurrence.
// Grid: (N, B*H) — one block per query position.
// The block streams K/V tiles, keeping only:
//   m_i  running row max
//   l_i  running row sum of exp(s - m)
//   O_i  unnormalized output accumulator in D
// No N×N matrix is written.

constexpr int BN = 32;
constexpr int kThreads = 128;
constexpr int kMaxD = 128;

template <typename T>
__global__ void online_softmax_kernel(const T* __restrict__ Q,
                                      const T* __restrict__ K,
                                      const T* __restrict__ V, T* __restrict__ O,
                                      float* __restrict__ LSE, int N, int D,
                                      float scale, bool causal) {
  const int i = blockIdx.x;
  const int bh = blockIdx.y;
  if (i >= N) return;

  const T* Qbh = Q + static_cast<int64_t>(bh) * N * D;
  const T* Kbh = K + static_cast<int64_t>(bh) * N * D;
  const T* Vbh = V + static_cast<int64_t>(bh) * N * D;
  T* Obh = O + static_cast<int64_t>(bh) * N * D;

  extern __shared__ float smem[];
  float* Qs = smem;                  // [D]
  float* Ks = Qs + D;                // [BN * D]
  float* Vs = Ks + BN * D;           // [BN * D]
  float* scores = Vs + BN * D;       // [BN]
  float* Oacc = scores + BN;         // [D]
  float* red = Oacc + D;             // [32]

  for (int d = threadIdx.x; d < D; d += kThreads) {
    Qs[d] = to_f32(Qbh[i * D + d]);
    Oacc[d] = 0.f;
  }
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.f;

  for (int j0 = 0; j0 < N; j0 += BN) {
    const int j_lim = min(BN, N - j0);

    if (causal && j0 > i) break;  // remaining tiles are strictly upper-triangular

    for (int t = threadIdx.x; t < j_lim * D; t += kThreads) {
      const int j = t / D;
      const int d = t % D;
      Ks[j * D + d] = to_f32(Kbh[(j0 + j) * D + d]);
      Vs[j * D + d] = to_f32(Vbh[(j0 + j) * D + d]);
    }
    __syncthreads();

    // Each thread computes one or more scores of this KV tile.
    for (int j = threadIdx.x; j < BN; j += kThreads) {
      float s = -INFINITY;
      if (j < j_lim) {
        const int gj = j0 + j;
        if (!(causal && gj > i)) {
          s = 0.f;
          for (int d = 0; d < D; ++d) s += Qs[d] * Ks[j * D + d];
          s *= scale;
        }
      }
      scores[j] = s;
    }
    __syncthreads();

    float m_tile = -INFINITY;
    for (int j = threadIdx.x; j < j_lim; j += kThreads) {
      m_tile = fmaxf(m_tile, scores[j]);
    }
    m_tile = block_reduce_max<kThreads>(m_tile, red);
    const float m_new = fmaxf(m_i, m_tile);
    const float alpha = (m_i == -INFINITY) ? 0.f : expf(m_i - m_new);

    if (threadIdx.x == 0) {
      // stash alpha in red[1] so all threads can use it after sync
      red[1] = alpha;
      red[2] = m_new;
    }
    __syncthreads();
    const float a = red[1];
    const float mn = red[2];

    for (int d = threadIdx.x; d < D; d += kThreads) {
      Oacc[d] *= a;
    }

    float l_tile = 0.f;
    for (int j = threadIdx.x; j < j_lim; j += kThreads) {
      const float p = (scores[j] == -INFINITY) ? 0.f : expf(scores[j] - mn);
      scores[j] = p;  // reuse as P
      l_tile += p;
    }
    l_tile = block_reduce_sum<kThreads>(l_tile, red);
    l_i = a * l_i + l_tile;

    __syncthreads();
    for (int d = threadIdx.x; d < D; d += kThreads) {
      float acc = 0.f;
      for (int j = 0; j < j_lim; ++j) acc += scores[j] * Vs[j * D + d];
      Oacc[d] += acc;
    }
    __syncthreads();
    m_i = mn;
  }

  const float inv = (l_i > 0.f) ? (1.f / l_i) : 0.f;
  for (int d = threadIdx.x; d < D; d += kThreads) {
    Obh[i * D + d] = from_f32<T>(Oacc[d] * inv);
  }
  if (threadIdx.x == 0 && LSE) {
    LSE[static_cast<int64_t>(bh) * N + i] =
        (l_i > 0.f) ? (m_i + logf(l_i)) : -INFINITY;
  }
}

}  // namespace

void launch_online_softmax(const void* Q, const void* K, const void* V, void* O,
                           float* LSE, int B, int H, int N, int D, float scale,
                           bool causal, DType dt, cudaStream_t stream) {
  if (D > kMaxD) {
    throw std::runtime_error("online softmax kernel supports D <= 128");
  }
  dim3 grid(N, B * H);
  const size_t smem =
      sizeof(float) * (static_cast<size_t>(D) + 2ull * BN * D + BN + D + 32);
  auto launch = [&](auto dummy) {
    using T = decltype(dummy);
    online_softmax_kernel<T><<<grid, kThreads, smem, stream>>>(
        static_cast<const T*>(Q), static_cast<const T*>(K),
        static_cast<const T*>(V), static_cast<T*>(O), LSE, N, D, scale, causal);
  };
  switch (dt) {
    case DType::F32:
      launch(float{});
      break;
    case DType::F16:
      launch(half{});
      break;
    case DType::BF16:
      launch(nv_bfloat16{});
      break;
  }
  FLASH_CUDA_CHECK(cudaGetLastError());
}

}  // namespace flash
