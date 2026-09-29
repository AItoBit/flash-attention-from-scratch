#include "attention.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
    std::printf("No CUDA device visible. Compile succeeded; run on a machine with a GPU.\n");
    return 2;
  }
  const int B = 1, H = 2, N = 32, D = 32;
  const int elems = B * H * N * D;
  std::vector<float> Q(elems), K(elems), V(elems), Ocpu(elems), Ogpu(elems);
  for (int i = 0; i < elems; ++i) {
    Q[i] = std::sin(0.01f * i);
    K[i] = std::cos(0.02f * i);
    V[i] = std::sin(0.03f * i + 1.f);
  }
  const float scale = flash::default_scale(D);
  flash::attention_cpu_fp32(Q.data(), K.data(), V.data(), Ocpu.data(), B, H, N, D,
                            scale, true);

  float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr, *dLSE = nullptr;
  FLASH_CUDA_CHECK(cudaMalloc(&dQ, elems * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMalloc(&dK, elems * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMalloc(&dV, elems * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMalloc(&dO, elems * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMalloc(&dLSE, B * H * N * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMemcpy(dQ, Q.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  FLASH_CUDA_CHECK(cudaMemcpy(dK, K.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  FLASH_CUDA_CHECK(cudaMemcpy(dV, V.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

  flash::launch_flash_fwd(dQ, dK, dV, dO, dLSE, B, H, N, D, scale, true,
                          flash::DType::F32, nullptr);
  FLASH_CUDA_CHECK(cudaDeviceSynchronize());
  FLASH_CUDA_CHECK(cudaMemcpy(Ogpu.data(), dO, elems * sizeof(float),
                              cudaMemcpyDeviceToHost));

  float max_abs = 0.f, mean_abs = 0.f;
  for (int i = 0; i < elems; ++i) {
    const float e = std::fabs(Ocpu[i] - Ogpu[i]);
    max_abs = std::max(max_abs, e);
    mean_abs += e;
  }
  mean_abs /= elems;
  std::printf("CPU vs flash_fwd (causal)  max_abs=%.6f  mean_abs=%.6f\n", max_abs,
              mean_abs);

  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);
  cudaFree(dLSE);
  return max_abs < 2e-3f ? 0 : 1;
}
