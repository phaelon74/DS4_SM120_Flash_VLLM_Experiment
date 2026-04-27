#pragma once

// SM120 HyperConnection prenorm GEMM (production v2).
//
// Replaces the scalar SM120 fallback in
// ``csrc/sm120_hc_prenorm_fallback.cu`` with a tiled, warp-cooperative
// implementation that uses vectorized loads and a unified block-reduce
// kernel for any N. Output contract is identical:
//
//   * A  : [M, K] BF16, K-major (a.stride(1) == 1 expected)
//   * B  : [N, K] FP32, K-major (b.stride(1) == 1 expected)
//   * D  : [num_splits, M, N] FP32 when num_splits > 1; [M, N] otherwise
//   * S  : [num_splits, M] FP32 (sum_k a^2)  when num_splits > 1; [M] otherwise
//
// Implementation lives in ``csrc/sm120_tf32_hc_prenorm_gemm.cu``. The MMA-
// upgrade path is marked with ``// TODO(SM120-MMA):`` blocks; until that's in,
// the kernel performs FMA in fp32 with TF32 rounding of B (matching the
// scalar fallback's numerics).
//
// Selected via ``DG_SM120_HC_PRENORM_V2=1`` (default ON in compose).

#include <torch/python.h>

namespace deep_gemm {

void sm120_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                const torch::Tensor& b,
                                const torch::Tensor& d,
                                const torch::Tensor& sqr_sum,
                                int m, int n, int k,
                                int num_splits);

} // namespace deep_gemm
