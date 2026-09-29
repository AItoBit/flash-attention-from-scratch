#pragma once

#include <cstddef>
#include <cstdint>

namespace flash {

// Row-major [B, H, N, D] view. Strides are in elements, not bytes.
struct Tensor4D {
  void* data = nullptr;
  int B = 0;
  int H = 0;
  int N = 0;
  int D = 0;
  int stride_b = 0;
  int stride_h = 0;
  int stride_n = 0;
  int stride_d = 1;

  template <typename T>
  T* ptr() const {
    return reinterpret_cast<T*>(data);
  }

  int64_t numel() const {
    return static_cast<int64_t>(B) * H * N * D;
  }

  int64_t offset(int b, int h, int n, int d) const {
    return static_cast<int64_t>(b) * stride_b +
           static_cast<int64_t>(h) * stride_h +
           static_cast<int64_t>(n) * stride_n +
           static_cast<int64_t>(d) * stride_d;
  }
};

inline Tensor4D make_contiguous_ncd(void* data, int B, int H, int N, int D) {
  Tensor4D t;
  t.data = data;
  t.B = B;
  t.H = H;
  t.N = N;
  t.D = D;
  t.stride_d = 1;
  t.stride_n = D;
  t.stride_h = N * D;
  t.stride_b = H * N * D;
  return t;
}

}  // namespace flash
