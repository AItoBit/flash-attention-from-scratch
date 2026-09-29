#pragma once

#include <mma.h>
#include <cuda_fp16.h>

namespace flash {
namespace mma {

using namespace nvcuda;

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// QK^T: C[M,N] = A[M,K] @ B[K,N] where B is K^T.
// Q is [BM, D] row-major, K is [BN, D] row-major.
// Viewing K as B with col-major [D, BN] is equivalent.

__device__ __forceinline__ void mma_qk_tile(
    const half* Qs, const half* Ks, float* S, int BM, int BN, int D, int ldq,
    int ldk, int lds, int warp_m, int warp_n) {
  wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.f);

  const half* a = Qs + warp_m * WM * ldq;
  const half* b = Ks + warp_n * WN * ldk;  // row-major [BN, D] → col-major [D, BN]
  for (int k = 0; k < D; k += WK) {
    wmma::load_matrix_sync(a_frag, a + k, ldq);
    wmma::load_matrix_sync(b_frag, b + k, ldk);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(S + warp_m * WM * lds + warp_n * WN, c_frag, lds,
                          wmma::mem_row_major);
}

// O[BM, D] += P[BM, BN] @ V[BN, D]
__device__ __forceinline__ void mma_pv_tile(
    const half* P, const half* Vs, float* Oacc, int BM, int BN, int D, int ldp,
    int ldv, int ldo, int warp_m, int warp_d) {
  wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;

  for (int d0 = warp_d * WN; d0 < D; d0 += WN * 2) {  // 2 warps along D when warp_d in {0,1}
    (void)BM;
    wmma::load_matrix_sync(c_frag, Oacc + warp_m * WM * ldo + d0, ldo,
                           wmma::mem_row_major);
    const half* a = P + warp_m * WM * ldp;
    const half* b = Vs + d0;
    for (int k = 0; k < BN; k += WK) {
      wmma::load_matrix_sync(a_frag, a + k, ldp);
      wmma::load_matrix_sync(b_frag, b + k * ldv, ldv);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(Oacc + warp_m * WM * ldo + d0, c_frag, ldo,
                            wmma::mem_row_major);
  }
}

}  // namespace mma
}  // namespace flash
