#pragma once

#include <cute/arch/mma_sm100_umma.hpp>
#include <torch/python.h>

namespace deep_gemm {

bool sm120_m_grouped_fp8_fp4_gemm_nt_contiguous_cutlass(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& grouped_layout,
    int num_groups, int m, int n, int k, int gran_k_a, int gran_k_b,
    const cute::UMMA::Major& major_b, bool use_psum_layout);

bool sm120_m_grouped_fp8_fp4_gemm_nt_contiguous_cutlass_with_starts(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& grouped_layout,
    const torch::Tensor& expert_starts, const torch::Tensor& expert_counts,
    int num_groups, int m, int n, int k, int gran_k_a, int gran_k_b,
    const cute::UMMA::Major& major_b, bool use_psum_layout);

torch::Tensor sm120_prepack_fp8_fp4_sfb(const torch::Tensor& sfb,
                                        int layout_m, int n, int k);

int64_t sm120_fp8_fp4_sfb_layout_numel(int layout_m, int n, int k);

} // namespace deep_gemm
