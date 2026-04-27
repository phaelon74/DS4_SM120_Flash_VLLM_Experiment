#pragma once

// SM120 Fused sparse MLA decode kernel (v2).
//
// Provides the same numerical interface as
// ``sparse_mla_decode_from_bf16_workspace`` but reads directly from the
// physical FP8 KV cache (``fp8_ds_mla`` layout) using sparse indices and
// performs Q*K^T / online softmax / P*V in a single CTA, with no BF16
// workspace materialization.
//
// Implementation lives in ``csrc/sm120_sparse_mla_decode_v2.cu``.
//
// Performance characteristics (vs the workspace + BMM bridge):
//   * One kernel launch instead of (gather + bmm + softmax + bmm)
//   * No BF16 workspace tensor allocated
//   * Q held in registers / SMEM, not re-read per chunk
//   * Online softmax in registers, no probs/scores intermediate tensor
//   * Inner Q*K^T and P*V loops are scalar today; the file marks
//     ``// TODO(SM120-MMA):`` blocks where ``mma.sync.aligned.kind::f8f6f4``
//     instructions should be slotted in once initial integration is validated
//     on hardware.
//
// Functional gating: enabled by ``DG_SM120_FUSED_DECODE_V2=1`` from the
// patcher; default-off until live validation.

#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace deep_gemm {
namespace sm120_mla_v2 {

// Fused sparse MLA decode operating directly on the FP8 ds_mla KV cache.
//
// Arguments mirror ``sparse_mla_decode``:
//   q                   : [B, H, head_dim_q]     bf16 / fp16
//   k_cache             : [num_blocks, block_size, 1, token_bytes+scale_bytes]
//                         uint8 (fp8_ds_mla)
//   indices             : [B, 1, K]              int32/int64, -1 = padded
//   topk_length         : [B] int32/int64 OR None.  When provided it gives the
//                         number of valid entries per row in ``indices``.
//   attn_sink           : [H] fp32 OR None
//   head_dim_v          : 512 (only supported value today)
//   softmax_scale       : float
//   block_size          : KV-cache block size (256 in the live image)
//   out                 : optional pre-allocated output [B, H, head_dim_v] bf16
//
// Returns (out, lse) where lse has shape [B, H] in fp32.
//
// On unsupported dtypes / shapes we throw — callers should keep the existing
// workspace path as fallback.
std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_v2(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& indices,
    const pybind11::object& topk_length,
    const pybind11::object& attn_sink,
    int head_dim_v,
    double softmax_scale,
    int block_size,
    const pybind11::object& out);

void register_decode_v2_apis(pybind11::module& m);

} // namespace sm120_mla_v2
} // namespace deep_gemm
