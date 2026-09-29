#pragma once

#include <cmath>
#include <cstdint>

namespace flash {

enum class DType : int { F32 = 0, F16 = 1, BF16 = 2 };

enum class Impl : int {
  Naive = 1,
  Tiled = 2,
  Online = 3,
  Flash = 4,
  Shared = 5,
  Warp = 6,
  MMA = 7,
  Flash2 = 8
};

struct TileConfig {
  int BM = 64;
  int BN = 64;
  int BK = 32;
};

struct AttentionShape {
  int B = 1;
  int H = 1;
  int N = 0;
  int D = 0;
};

inline float default_scale(int D) {
  return 1.0f / sqrtf(static_cast<float>(D));
}

// Standard attention FLOP count: 2*N*N*D for QK^T plus 2*N*N*D for PV.
inline double attention_flops(int B, int H, int N, int D, bool /*causal*/) {
  return 4.0 * static_cast<double>(B) * H * N * N * D;
}

}  // namespace flash
