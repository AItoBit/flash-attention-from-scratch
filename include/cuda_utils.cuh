#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#include "config.hpp"

#ifdef INFINITY
#undef INFINITY
#endif
#define INFINITY CUDART_INF_F

namespace flash {

// ---------------------------------------------------------------------------
// dtype conversion
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float to_f32(T x);

template <>
__device__ __forceinline__ float to_f32<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ float to_f32<half>(half x) {
  return __half2float(x);
}
template <>
__device__ __forceinline__ float to_f32<nv_bfloat16>(nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_f32(float x);

template <>
__device__ __forceinline__ float from_f32<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ half from_f32<half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ nv_bfloat16 from_f32<nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

__host__ __device__ __forceinline__ int idx4(int b, int h, int n, int d, int H,
                                             int N, int D) {
  return ((b * H + h) * N + n) * D + d;
}

__host__ __device__ __forceinline__ int bh_index(int b, int h, int H) {
  return b * H + h;
}

// ---------------------------------------------------------------------------
// warp / block reductions
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_all_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  }
  return v;
}

__device__ __forceinline__ float warp_all_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}

template <int BLOCK>
__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  val = warp_reduce_max(val);
  if (lane == 0) smem[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (BLOCK / 32)) ? smem[lane] : -INFINITY;
  if (wid == 0) val = warp_reduce_max(val);
  if (threadIdx.x == 0) smem[0] = val;
  __syncthreads();
  return smem[0];
}

template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  val = warp_reduce_sum(val);
  if (lane == 0) smem[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (BLOCK / 32)) ? smem[lane] : 0.f;
  if (wid == 0) val = warp_reduce_sum(val);
  if (threadIdx.x == 0) smem[0] = val;
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ bool causal_mask(int q_pos, int k_pos, bool causal) {
  return causal && (k_pos > q_pos);
}

}  // namespace flash
