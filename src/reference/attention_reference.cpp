#include "attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace flash {

void attention_cpu_fp32(const float* Q, const float* K, const float* V, float* O,
                        int B, int H, int N, int D, float scale, bool causal) {
  std::vector<float> scores(static_cast<size_t>(N));
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < N; ++i) {
        const float* q = Q + ((b * H + h) * N + i) * D;
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) {
          if (causal && j > i) {
            scores[j] = -std::numeric_limits<float>::infinity();
            continue;
          }
          const float* k = K + ((b * H + h) * N + j) * D;
          float acc = 0.f;
          for (int d = 0; d < D; ++d) acc += q[d] * k[d];
          scores[j] = acc * scale;
          m = std::max(m, scores[j]);
        }
        float l = 0.f;
        for (int j = 0; j < N; ++j) {
          const float e = (scores[j] == -std::numeric_limits<float>::infinity())
                              ? 0.f
                              : std::exp(scores[j] - m);
          scores[j] = e;
          l += e;
        }
        const float inv = (l > 0.f) ? (1.f / l) : 0.f;
        float* o = O + ((b * H + h) * N + i) * D;
        for (int d = 0; d < D; ++d) o[d] = 0.f;
        for (int j = 0; j < N; ++j) {
          const float p = scores[j] * inv;
          const float* v = V + ((b * H + h) * N + j) * D;
          for (int d = 0; d < D; ++d) o[d] += p * v[d];
        }
      }
    }
  }
}

}  // namespace flash
