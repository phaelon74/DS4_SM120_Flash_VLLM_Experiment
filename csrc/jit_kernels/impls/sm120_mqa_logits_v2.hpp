#pragma once

#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace deep_gemm {
namespace sm120_mla_v2 {

// SM120 MQA logits v2 (FP8 path).
//
// Replacement kernel for the dominant prefill bottleneck identified by live
// PyTorch + DeepGEMM profile traces: ``sm120_fallback::mqa_logits_kernel``
// (non-paged FP8) accounts for ~44% of TTFT at 4k, ~62% at 8k, and ~76% at
// 16k uncached prompt tokens (961 / 3647 / 13963 ms respectively, growing
// near-quadratically in prompt length).
//
// C1 (this commit): scaffold with scalar inner, correctness-equivalent to
// ``sm120_fp8_mqa_logits_fallback``. Provides the entry point and Python
// binding that subsequent commits will upgrade to a tensor-core BF16
// m16n8k16 mma.sync implementation (target: ~200x faster on the kernel
// itself, ~4x TTFT improvement at 16k).
//
// Scope is non-paged FP8 only:
//  - Paged FP8 already routes through the FAST kernel
//    (``paged_fp8_mqa_logits_fast_kernel``, ~11-23 ms / request) and is not
//    a meaningful bottleneck.
//  - FP4 paths are not exercised by the live DeepSeek V4 Flash dispatch.
void sm120_fp8_mqa_logits_v2(
    const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim);

void register_mqa_logits_v2_apis(pybind11::module& m);

} // namespace sm120_mla_v2
} // namespace deep_gemm
