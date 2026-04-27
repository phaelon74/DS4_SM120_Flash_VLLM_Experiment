#pragma once

// SM120 Fused sparse MLA prefill kernel (v2).
//
// Replaces the BF16 workspace + ``torch.bmm`` bridge currently used for
// DeepSeek V4 sparse MLA prefill on SM120 with a single fused kernel that
// reads directly from the FP8 ds_mla KV cache via a workspace_map (one int32
// per workspace row giving the physical KV slot).
//
// Compared to the existing
// ``sparse_mla_prefill_from_fp8_workspace_map`` scalar kernel, v2:
//   * Holds Q in registers / SMEM
//   * Does FA2-style online softmax in registers (no scores / probs tensors)
//   * Emits BF16 output and FP32 LSE in a single launch
//   * Has explicit hooks (``// TODO(SM120-MMA):``) for slotting in
//     ``mma.sync.aligned.kind::f8f6f4.block_scale`` warp-level instructions
//
// Functional gating: enabled by ``DG_SM120_FUSED_PREFILL_V2=1`` from the
// patcher; default-off until live validation.

#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace deep_gemm {
namespace sm120_mla_v2 {

// Fused sparse MLA prefill that reads directly from the FP8 KV cache via a
// per-row workspace_map (analogous to ``build_prefill_workspace_map`` output).
//
// Arguments mirror ``sparse_mla_prefill_from_fp8_workspace_map``:
//   q              : [S, H, head_dim] bf16/fp16
//   k_cache        : [num_blocks, block_size, 1, token_bytes+scale_bytes] uint8
//   workspace_map  : [N_workspace] int32 — physical KV slot index, or -1
//                    when slotted into a different cache (we only handle the
//                    single-cache path; callers with extra caches should use
//                    the existing two-cache helper).
//   indices        : [S, 1, K]  int32/int64; entries reference workspace_map
//                    rows.
//   topk_length    : optional [S] int32/int64
//   attn_sink      : optional [H] fp32
//   block_size     : KV cache block size
//   head_dim_v     : 512 (only supported value today)
//   softmax_scale  : float
//   out            : optional pre-allocated output [S, H, head_dim_v]
//
// Returns (out, max_logits, lse). max_logits is filled with NaN to signal it
// is "row_max" tracked internally; we emit lse in fp32.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_v2(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& workspace_map,
    const torch::Tensor& indices,
    const pybind11::object& topk_length,
    const pybind11::object& attn_sink,
    int block_size,
    int head_dim_v,
    double softmax_scale,
    const pybind11::object& out);

void register_prefill_v2_apis(pybind11::module& m);

// Combined entry point used by ``python_api.cpp``. This is implemented in
// ``csrc/sm120_sparse_mla_prefill_v2.cu`` and calls both
// ``register_decode_v2_apis`` (declared in
// ``sm120_sparse_mla_decode_v2.hpp``) and ``register_prefill_v2_apis``.
void register_apis(pybind11::module& m);

} // namespace sm120_mla_v2
} // namespace deep_gemm
