#pragma once

#include "config.hpp"
#include "tensor.hpp"

#include <cuda_runtime.h>

namespace flash {

// ---------------------------------------------------------------------------
// Stage 0 — CPU reference (fp32 accumulation)
// ---------------------------------------------------------------------------
void attention_cpu_fp32(const float* Q, const float* K, const float* V, float* O,
                        int B, int H, int N, int D, float scale, bool causal);

// ---------------------------------------------------------------------------
// Stage 1 — Naive CUDA: materializes S and P in HBM
// ---------------------------------------------------------------------------
void launch_qk_naive(const void* Q, const void* K, float* S, int B, int H,
                     int N, int D, float scale, bool causal, DType dt,
                     cudaStream_t stream);

void launch_softmax_naive(const float* S, float* P, int B, int H, int N,
                          cudaStream_t stream);

void launch_pv_naive(const float* P, const void* V, void* O, int B, int H,
                     int N, int D, DType dt, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 2 — Tiled QK^T into HBM scores
// ---------------------------------------------------------------------------
void launch_qk_tiled(const void* Q, const void* K, float* S, int B, int H,
                     int N, int D, float scale, bool causal, DType dt,
                     cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stage 3 — Online softmax (one query row per block, stream KV tiles)
// ---------------------------------------------------------------------------
void launch_online_softmax(const void* Q, const void* K, const void* V, void* O,
                           float* LSE, int B, int H, int N, int D, float scale,
                           bool causal, DType dt, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stages 4–8 — fused FlashAttention variants
// ---------------------------------------------------------------------------
void launch_flash_fwd(const void* Q, const void* K, const void* V, void* O,
                      float* LSE, int B, int H, int N, int D, float scale,
                      bool causal, DType dt, cudaStream_t stream);

void launch_flash_shared(const void* Q, const void* K, const void* V, void* O,
                         float* LSE, int B, int H, int N, int D, float scale,
                         bool causal, DType dt, const TileConfig& cfg,
                         cudaStream_t stream);

void launch_flash_warp(const void* Q, const void* K, const void* V, void* O,
                       float* LSE, int B, int H, int N, int D, float scale,
                       bool causal, DType dt, cudaStream_t stream);

void launch_flash_mma(const void* Q, const void* K, const void* V, void* O,
                      float* LSE, int B, int H, int N, int D, float scale,
                      bool causal, DType dt, cudaStream_t stream);

void launch_flash2(const void* Q, const void* K, const void* V, void* O,
                   float* LSE, int B, int H, int N, int D, float scale,
                   bool causal, DType dt, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Backward (fused, recomputes P from Q/K and saved LSE)
// ---------------------------------------------------------------------------
void launch_flash_bwd(const void* Q, const void* K, const void* V, const void* O,
                      const void* dO, const float* LSE, void* dQ, void* dK,
                      void* dV, int B, int H, int N, int D, float scale,
                      bool causal, DType dt, cudaStream_t stream);

}  // namespace flash
