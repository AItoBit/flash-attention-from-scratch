#include "attention.hpp"
#include "config.hpp"
#include "utils.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace {

flash::DType dtype_from_tensor(const torch::Tensor& t) {
  if (t.scalar_type() == torch::kFloat32) return flash::DType::F32;
  if (t.scalar_type() == torch::kFloat16) return flash::DType::F16;
  if (t.scalar_type() == torch::kBFloat16) return flash::DType::BF16;
  TORCH_CHECK(false, "unsupported dtype (expected fp32/fp16/bf16)");
  return flash::DType::F32;
}

void check_qkvo(const torch::Tensor& q, const torch::Tensor& k,
                const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() && v.sizes() == q.sizes(),
              "Q, K, V must be [B, H, N, D] with identical shape");
  TORCH_CHECK(q.size(3) <= 128, "head dim D must be <= 128");
}

float scale_or_default(const torch::Tensor& q, const c10::optional<double>& scale) {
  if (scale.has_value()) return static_cast<float>(scale.value());
  return flash::default_scale(static_cast<int>(q.size(3)));
}

struct BHND {
  int B, H, N, D;
};

BHND shape_of(const torch::Tensor& q) {
  return {static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
          static_cast<int>(q.size(2)), static_cast<int>(q.size(3))};
}

torch::Tensor empty_lse(const torch::Tensor& q) {
  return torch::empty({q.size(0), q.size(1), q.size(2)},
                      q.options().dtype(torch::kFloat32));
}

std::tuple<torch::Tensor, torch::Tensor> naive_fwd(torch::Tensor q, torch::Tensor k,
                                                   torch::Tensor v, bool causal,
                                                   c10::optional<double> scale) {
  check_qkvo(q, k, v);
  auto s = shape_of(q);
  auto scores = torch::empty({s.B, s.H, s.N, s.N}, q.options().dtype(torch::kFloat32));
  auto probs = torch::empty_like(scores);
  auto out = torch::empty_like(q);
  const float sc = scale_or_default(q, scale);
  const auto dt = dtype_from_tensor(q);
  auto stream = at::cuda::getCurrentCUDAStream();
  flash::launch_qk_naive(q.data_ptr(), k.data_ptr(), scores.data_ptr<float>(), s.B,
                         s.H, s.N, s.D, sc, causal, dt, stream);
  flash::launch_softmax_naive(scores.data_ptr<float>(), probs.data_ptr<float>(), s.B,
                              s.H, s.N, stream);
  flash::launch_pv_naive(probs.data_ptr<float>(), v.data_ptr(), out.data_ptr(), s.B,
                         s.H, s.N, s.D, dt, stream);
  return {out, scores};
}

std::tuple<torch::Tensor, torch::Tensor> tiled_fwd(torch::Tensor q, torch::Tensor k,
                                                   torch::Tensor v, bool causal,
                                                   c10::optional<double> scale) {
  check_qkvo(q, k, v);
  auto s = shape_of(q);
  auto scores = torch::empty({s.B, s.H, s.N, s.N}, q.options().dtype(torch::kFloat32));
  auto probs = torch::empty_like(scores);
  auto out = torch::empty_like(q);
  const float sc = scale_or_default(q, scale);
  const auto dt = dtype_from_tensor(q);
  auto stream = at::cuda::getCurrentCUDAStream();
  flash::launch_qk_tiled(q.data_ptr(), k.data_ptr(), scores.data_ptr<float>(), s.B,
                         s.H, s.N, s.D, sc, causal, dt, stream);
  flash::launch_softmax_naive(scores.data_ptr<float>(), probs.data_ptr<float>(), s.B,
                              s.H, s.N, stream);
  flash::launch_pv_naive(probs.data_ptr<float>(), v.data_ptr(), out.data_ptr(), s.B,
                         s.H, s.N, s.D, dt, stream);
  return {out, scores};
}

using FusedLauncher = void (*)(const void*, const void*, const void*, void*, float*,
                               int, int, int, int, float, bool, flash::DType,
                               cudaStream_t);

std::tuple<torch::Tensor, torch::Tensor> fused_fwd(FusedLauncher launch,
                                                   torch::Tensor q, torch::Tensor k,
                                                   torch::Tensor v, bool causal,
                                                   c10::optional<double> scale) {
  check_qkvo(q, k, v);
  auto s = shape_of(q);
  auto out = torch::empty_like(q);
  auto lse = empty_lse(q);
  const float sc = scale_or_default(q, scale);
  launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
         lse.data_ptr<float>(), s.B, s.H, s.N, s.D, sc, causal,
         dtype_from_tensor(q), at::cuda::getCurrentCUDAStream());
  return {out, lse};
}

std::tuple<torch::Tensor, torch::Tensor> shared_fwd(torch::Tensor q, torch::Tensor k,
                                                    torch::Tensor v, bool causal,
                                                    c10::optional<double> scale,
                                                    int BM, int BN) {
  check_qkvo(q, k, v);
  auto s = shape_of(q);
  auto out = torch::empty_like(q);
  auto lse = empty_lse(q);
  flash::TileConfig cfg;
  cfg.BM = BM;
  cfg.BN = BN;
  flash::launch_flash_shared(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                             lse.data_ptr<float>(), s.B, s.H, s.N, s.D,
                             scale_or_default(q, scale), causal, dtype_from_tensor(q),
                             cfg, at::cuda::getCurrentCUDAStream());
  return {out, lse};
}

std::vector<torch::Tensor> flash_bwd(torch::Tensor q, torch::Tensor k,
                                     torch::Tensor v, torch::Tensor o,
                                     torch::Tensor do_, torch::Tensor lse,
                                     bool causal, c10::optional<double> scale) {
  check_qkvo(q, k, v);
  TORCH_CHECK(o.sizes() == q.sizes() && do_.sizes() == q.sizes());
  TORCH_CHECK(lse.is_contiguous() && lse.scalar_type() == torch::kFloat32);
  auto dq = torch::zeros_like(q);
  auto dk = torch::zeros_like(k);
  auto dv = torch::zeros_like(v);
  auto s = shape_of(q);
  flash::launch_flash_bwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
                          do_.data_ptr(), lse.data_ptr<float>(), dq.data_ptr(),
                          dk.data_ptr(), dv.data_ptr(), s.B, s.H, s.N, s.D,
                          scale_or_default(q, scale), causal, dtype_from_tensor(q),
                          at::cuda::getCurrentCUDAStream());
  return {dq, dk, dv};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "flash-attention-from-scratch CUDA kernels";
  m.def("naive_fwd", &naive_fwd, "Naive QK + softmax + PV (materializes N×N)");
  m.def("tiled_fwd", &tiled_fwd, "Tiled QK^T then softmax + PV");
  m.def(
      "online_fwd",
      [](torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
         c10::optional<double> scale) {
        return fused_fwd(&flash::launch_online_softmax, q, k, v, causal, scale);
      },
      "Online-softmax attention (one query row per block)");
  m.def(
      "flash_fwd",
      [](torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
         c10::optional<double> scale) {
        return fused_fwd(&flash::launch_flash_fwd, q, k, v, causal, scale);
      },
      "Fused FlashAttention-1-style kernel");
  m.def("flash_shared_fwd", &shared_fwd, "Shared-memory optimized fused kernel",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("causal") = false, pybind11::arg("scale") = pybind11::none(),
        pybind11::arg("BM") = 32, pybind11::arg("BN") = 32);
  m.def(
      "flash_warp_fwd",
      [](torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
         c10::optional<double> scale) {
        return fused_fwd(&flash::launch_flash_warp, q, k, v, causal, scale);
      },
      "Warp-shuffle fused kernel");
  m.def(
      "flash_mma_fwd",
      [](torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
         c10::optional<double> scale) {
        return fused_fwd(&flash::launch_flash_mma, q, k, v, causal, scale);
      },
      "WMMA Tensor Core fused kernel");
  m.def(
      "flash2_fwd",
      [](torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
         c10::optional<double> scale) {
        return fused_fwd(&flash::launch_flash2, q, k, v, causal, scale);
      },
      "FA2-style warp-partitioned fused kernel");
  m.def("flash_bwd", &flash_bwd, "Fused FlashAttention backward");
  m.def("attention_flops", [](int B, int H, int N, int D) {
    return flash::attention_flops(B, H, N, D, false);
  });
}
