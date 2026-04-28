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
// C1:  scaffold with scalar inner, correctness-equivalent to
//      ``sm120_fp8_mqa_logits_fallback``. Provides the entry point and
//      Python binding.
//
// C2a: adds a BF16 m16n8k16 mma.sync tensor-core path. Synthetic correctness
//      validated against torch ref + apis fallback (FP32 max_diff < 5e-3,
//      BF16 max_rel < 8e-3); 33x faster than the scalar inner at S=16k
//      (10.97 ms vs 365 ms; 706 us at S=4k).
//
// C3 (current): wired into the SM120 dispatch in ``csrc/apis/attention.hpp``
//      in place of ``sm120_fp8_mqa_logits_fallback``. The MMA inner is
//      default ON in this entry point. Toggles:
//        DG_SM120_MQA_LOGITS_V2_MMA=0     -> force the scalar inner.
//        DG_SM120_MQA_LOGITS_V2_STRICT=1  -> hard-fail on shapes the MMA
//                                            path cannot handle (default OFF
//                                            silently falls back to scalar).
//
// Math factorization that makes the MMA path clean:
//   logits[m,n] = sum_h max(0, sum_d q[m,h,d]*kv[n,d]*kv_sf[n]) * weights[m,h]
//               = kv_sf[n] * sum_h max(0, [Q[m,h,:] @ K[n,:]^T]) * weights[m,h]
// (valid because kv_sf[n] >= 0).
// The bracketed term is a pure unscaled BF16 matmul; ``kv_sf[n]`` becomes a
// single per-N scalar multiply applied in the final epilogue.
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

// C2a/C4 MMA launch entry. Implemented in ``csrc/sm120_mqa_logits_v2_mma.cu``.
// Returns ``true`` if the MMA fast path was launched, ``false`` if the
// kernel detected an unsupported shape combination and the caller must fall
// back to the scalar path. Currently supported:
//   * ``logits_dtype`` in {kFloat32, kBFloat16}
//   * ``head_dim`` in {64, 128}
//       - 64  is the C2a synthetic test/regression shape.
//       - 128 is the live DeepSeek V4 sparse indexer prefill+decode shape
//             (confirmed via DG_SM120_MQA_LOGITS_V2_TRACE on a real
//             4096-prompt request: seq_len=4096, seq_len_kv=1024,
//             num_heads=64, head_dim=128, dtype=fp32). The C2a draft only
//             supported head_dim=64 and silently fell back to the scalar
//             inner on every live call; C4 added the templated-on-kHeadDim
//             instantiation so the MMA path actually runs in production.
//   * ``num_heads <= 64`` (the SMEM weights buffer cap)
// Any other combination returns ``false`` without writing to ``logits``.
bool sm120_fp8_mqa_logits_v2_mma_try_launch(
    const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim);

void register_mqa_logits_v2_apis(pybind11::module& m);

} // namespace sm120_mla_v2
} // namespace deep_gemm
