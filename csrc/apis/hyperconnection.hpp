#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp"
#include "../jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp"
#include "../jit_kernels/impls/sm120_hc_prenorm_fallback.hpp"
#include "../jit_kernels/impls/sm120_tf32_hc_prenorm_gemm.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#endif

namespace deep_gemm::hyperconnection {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 const torch::Tensor& d,
                                 const torch::Tensor& sqr_sum,
                                 const std::optional<int>& num_splits) {
    // A and B must be K-major, D must be N-major
    DG_HOST_ASSERT(get_major_type_ab(a) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(b) == cute::UMMA::Major::K);
    check_major_type_cd(d);

    // S must be contiguous
    DG_HOST_ASSERT(sqr_sum.is_contiguous());

    // Type and shape checks
    const auto [m, k ] = get_shape<2>(a);
    const auto [n, k_] = get_shape<2>(b);
    if (num_splits.has_value()) {
        const auto [num_splits_, m_, n_] = get_shape<3>(d);
        const auto [num_splits__, m__] = get_shape<2>(sqr_sum);
        DG_HOST_ASSERT(num_splits.value() == num_splits_ and num_splits.value() == num_splits__ and num_splits.value() >= 1);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    } else {
        const auto [m_, n_] = get_shape<2>(d);
        const auto [m__] = get_shape<1>(sqr_sum);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    }
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sqr_sum.scalar_type() == torch::kFloat);

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Dispatch into different implements
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else if (arch_major == 10) {
        sm100_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else if (arch_major == 12) {
        // Prefer the v2 unified kernel; fall back to the original scalar
        // implementation if env-disabled or if N exceeds the v2 cap (64).
        const char* env = std::getenv("DG_SM120_HC_PRENORM_V2");
        const bool use_v2 = (env == nullptr) || (env[0] != '0');
        const bool use_v2_path = (use_v2 && n <= 64);

        // dsl12x Phase 5 trace hook (DG_SM120_HC_PRENORM_TRACE=1).
        // One-time atomic-counter trace per process (cap 64). Captures the
        // live (M, N, K, num_splits, dtype, path) distribution so the
        // upcoming MMA-based mHC kernel (Phase 5b/c) can be tuned to the
        // real shape mix. Default OFF; on by default in
        // docker-compose.dsl12x.yml.
        static std::atomic<int> _dg_sm120_hc_prenorm_trace_count{0};
        const char* trace_env = std::getenv("DG_SM120_HC_PRENORM_TRACE");
        const bool trace_enabled = (trace_env != nullptr) && (trace_env[0] != '0');
        if (trace_enabled) {
            int count = _dg_sm120_hc_prenorm_trace_count.fetch_add(1, std::memory_order_relaxed);
            if (count < 64) {
                std::fprintf(
                    stderr,
                    "[sm120_hc_prenorm_trace #%d] M=%lld N=%lld K=%lld num_splits=%d "
                    "dtype_a=BF16 dtype_b=F32 dtype_d=F32 path=%s\n",
                    count, static_cast<long long>(m),
                    static_cast<long long>(n), static_cast<long long>(k),
                    num_splits.has_value() ? num_splits.value() : 1,
                    use_v2_path ? "v2" : "fallback");
                std::fflush(stderr);
            }
        }

        if (use_v2_path) {
            sm120_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k,
                                       num_splits.has_value() ? num_splits.value() : 1);
        } else {
            sm120_tf32_hc_prenorm_gemm_fallback(a, b, d, sqr_sum, m, n, k,
                                                num_splits.has_value() ? num_splits.value() : 1);
        }
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

#endif

static void register_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("tf32_hc_prenorm_gemm", &tf32_hc_prenorm_gemm,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("sqr_sum"),
          py::arg("num_splits") = std::nullopt);
#endif
}

} // namespace deep_gemm::hyperconnection
