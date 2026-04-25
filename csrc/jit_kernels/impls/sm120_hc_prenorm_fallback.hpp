#pragma once

#include <torch/python.h>

namespace deep_gemm {

void sm120_tf32_hc_prenorm_gemm_fallback(const torch::Tensor& a,
                                         const torch::Tensor& b,
                                         const torch::Tensor& d,
                                         const torch::Tensor& sqr_sum,
                                         int m, int n, int k,
                                         int num_splits);

} // namespace deep_gemm
