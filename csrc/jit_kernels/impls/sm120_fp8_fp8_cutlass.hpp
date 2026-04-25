#pragma once

#include <cute/arch/mma_sm100_umma.hpp>
#include <torch/python.h>

namespace deep_gemm {

bool sm120_fp8_fp8_gemm_nt_cutlass(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, int m, int n, int k,
    int gran_mn_a, int gran_k_a, int gran_mn_b, int gran_k_b,
    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
    bool accumulate);

} // namespace deep_gemm
