#pragma once

#include <optional>

#include <cute/arch/mma_sm100_umma.hpp>
#include <torch/python.h>

namespace deep_gemm {

void sm120_fp8_gemm_nt_fallback(const torch::Tensor& a, const torch::Tensor& sfa,
                                const torch::Tensor& b, const torch::Tensor& sfb,
                                const std::optional<torch::Tensor>& c,
                                const torch::Tensor& d,
                                int m, int n, int k,
                                int gran_mn_a, int gran_k_a,
                                int gran_mn_b, int gran_k_b,
                                const cute::UMMA::Major& major_a,
                                const cute::UMMA::Major& major_b);

void sm120_fp8_bhr_hdr_bhd(const torch::Tensor& a, const torch::Tensor& a_scale,
                           const torch::Tensor& b, const torch::Tensor& b_scale,
                           const torch::Tensor& out);

void sm120_m_grouped_fp8_fp4_gemm_nt_contiguous_fallback(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& grouped_layout,
    int num_groups, int m, int n, int k, int gran_k_a, int gran_k_b,
    const cute::UMMA::Major& major_b, bool use_psum_layout);

} // namespace deep_gemm
