#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace flash {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline void cuda_check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "CUDA error %s:%d: %s", file, line,
                  cudaGetErrorString(err));
    throw std::runtime_error(buf);
  }
}

#define FLASH_CUDA_CHECK(expr) ::flash::cuda_check((expr), __FILE__, __LINE__)

inline void* device_alloc_bytes(size_t nbytes) {
  void* p = nullptr;
  FLASH_CUDA_CHECK(cudaMalloc(&p, nbytes));
  return p;
}

inline void device_free(void* p) {
  if (p) {
    FLASH_CUDA_CHECK(cudaFree(p));
  }
}

}  // namespace flash
