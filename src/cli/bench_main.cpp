#include "attention.hpp"
#include "config.hpp"
#include "utils.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void fill_randn(float* p, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.f, 1.f);
  for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

int main(int argc, char** argv) {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
    std::printf("No CUDA device visible. Compile succeeded; run on a machine with a GPU.\n");
    return 2;
  }
  int B = 1, H = 8, N = 512, D = 64;
  if (argc >= 5) {
    B = std::atoi(argv[1]);
    H = std::atoi(argv[2]);
    N = std::atoi(argv[3]);
    D = std::atoi(argv[4]);
  }
  const size_t elems = static_cast<size_t>(B) * H * N * D;
  std::vector<float> hQ(elems), hK(elems), hV(elems);
  fill_randn(hQ.data(), elems, 1);
  fill_randn(hK.data(), elems, 2);
  fill_randn(hV.data(), elems, 3);

  float *Q = nullptr, *K = nullptr, *V = nullptr, *O = nullptr, *LSE = nullptr;
  FLASH_CUDA_CHECK(cudaMalloc(&Q, elems * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMalloc(&K, elems * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMalloc(&V, elems * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMalloc(&O, elems * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMalloc(&LSE, static_cast<size_t>(B) * H * N * sizeof(float)));
  FLASH_CUDA_CHECK(cudaMemcpy(Q, hQ.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  FLASH_CUDA_CHECK(cudaMemcpy(K, hK.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  FLASH_CUDA_CHECK(cudaMemcpy(V, hV.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

  const float scale = flash::default_scale(D);
  const double flops = flash::attention_flops(B, H, N, D, false);
  std::printf("shape B=%d H=%d N=%d D=%d  FLOPs=%.3e\n", B, H, N, D, flops);

  auto bench = [&](const char* name, auto fn) {
    for (int i = 0; i < 5; ++i) fn();
    FLASH_CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    const int iters = 20;
    for (int i = 0; i < iters; ++i) fn();
    FLASH_CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double tflops = flops / (ms * 1e-3) / 1e12;
    std::printf("  %-12s  %8.3f ms  %6.2f TFLOP/s\n", name, ms, tflops);
  };

  bench("online", [&] {
    flash::launch_online_softmax(Q, K, V, O, LSE, B, H, N, D, scale, false,
                                 flash::DType::F32, nullptr);
  });
  bench("flash", [&] {
    flash::launch_flash_fwd(Q, K, V, O, LSE, B, H, N, D, scale, false,
                            flash::DType::F32, nullptr);
  });
  bench("warp", [&] {
    flash::launch_flash_warp(Q, K, V, O, LSE, B, H, N, D, scale, false,
                             flash::DType::F32, nullptr);
  });

  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  cudaFree(O);
  cudaFree(LSE);
  return 0;
}
