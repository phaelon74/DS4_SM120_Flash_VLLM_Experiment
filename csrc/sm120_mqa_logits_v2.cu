// SM120 MQA logits v2 (FP8 non-paged).
//
// History:
//   C1:  scaffold with scalar inner, bit-exact to
//        ``deep_gemm::sm120_fallback::mqa_logits_kernel``.
//   C2a: BF16 m16n8k16 mma.sync tensor-core path
//        (``csrc/sm120_mqa_logits_v2_mma.cu``). Synthetic correctness PASSes
//        against both torch ref and the apis fallback (FP32 max_diff < 5e-3,
//        BF16 max_rel < 8e-3); 33x faster than the scalar inner at S=16k
//        (10.97 ms vs 365 ms; 706 us at S=4k).
//   C3:  apis dispatch wired to call this v2 entry point on SM120 instead
//        of ``sm120_fp8_mqa_logits_fallback`` (see ``csrc/apis/attention.hpp``).
//        The MMA inner is now ON by default in this entry point; the env
//        var ``DG_SM120_MQA_LOGITS_V2_MMA=0`` disables it (forces the scalar
//        inner) for debugging or A/B comparison.
//
// Env vars consumed:
//   DG_SM120_MQA_LOGITS_V2_MMA
//       Default ON. Set to 0/false/no to force the scalar inner. Set to any
//       other value (or leave unset) to use the MMA inner.
//   DG_SM120_MQA_LOGITS_V2_STRICT
//       Default OFF. When set to 1/true/yes, an unsupported shape returned
//       by the MMA path raises a hard error instead of silently falling
//       back to scalar. Use during development to catch silent shape
//       fallbacks; leave OFF for production serving so unsupported shapes
//       (e.g. ``head_dim != 64``) still produce correct results.
//
// Profile attribution (live torch trace, single request, rank 0; recorded
// before C3 wire-up):
//   prompt 4k:   961.557 ms /  42 calls  (44% of 2.18s TTFT)
//   prompt 8k:  3646.926 ms /  63 calls  (62% of 5.87s TTFT)
//   prompt 16k: 13962.641 ms / 105 calls (76% of 18.44s TTFT)
// Synthetic kernel-level cost after C2a (S=16k single call): 11 ms. Live
// numbers will be captured into AGENTS.md after the C3 restart.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_mqa_logits_v2.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace sm120_mla_v2 {
namespace {

template <typename out_t>
__device__ __forceinline__ void store_logit(out_t* out, int64_t offset,
                                            float value) {
    out[offset] = static_cast<out_t>(value);
}

template <>
__device__ __forceinline__ void store_logit<__nv_bfloat16>(
    __nv_bfloat16* out, int64_t offset, float value) {
    out[offset] = __float2bfloat16(value);
}

// Scalar inner identical to ``sm120_fallback::mqa_logits_kernel<out_t,false>``.
// Retained after C3 as the correctness fallback for shapes the MMA path
// (``csrc/sm120_mqa_logits_v2_mma.cu``) does not support, and as the inner
// selected when ``DG_SM120_MQA_LOGITS_V2_MMA=0``.
template <typename out_t>
__global__ void fp8_mqa_logits_v2_scalar_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ kv,
    const float* __restrict__ kv_sf,
    const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seq_len_k_start,
    const int32_t* __restrict__ cu_seq_len_k_end,
    out_t* __restrict__ logits,
    int seq_len, int seq_len_kv, int num_heads, int head_dim,
    int out_cols, int logits_stride, bool compressed_logits) {
    const int64_t total = static_cast<int64_t>(seq_len) * out_cols;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int m = static_cast<int>(linear / out_cols);
        const int out_col = static_cast<int>(linear -
                                             static_cast<int64_t>(m) * out_cols);
        const int start = max(0, min(cu_seq_len_k_start[m], seq_len_kv));
        const int end = max(0, min(cu_seq_len_k_end[m], seq_len_kv));
        const int n = compressed_logits ? start + out_col : out_col;

        float result = -INFINITY;
        if (n >= start && n < end) {
            float sum = 0.0f;
            const float k_scale = kv_sf[n];
            const int64_t kv_base = static_cast<int64_t>(n) * head_dim;
            for (int h = 0; h < num_heads; ++h) {
                const int64_t q_base =
                    (static_cast<int64_t>(m) * num_heads + h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += static_cast<float>(q[q_base + d]) *
                           (static_cast<float>(kv[kv_base + d]) * k_scale);
                }
                sum += fmaxf(dot, 0.0f) *
                       weights[static_cast<int64_t>(m) * num_heads + h];
            }
            result = sum;
        }
        store_logit(logits,
                    static_cast<int64_t>(m) * logits_stride + out_col,
                    result);
    }
}
int v2_grid(int64_t total) {
    constexpr int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    return static_cast<int>(std::min<int64_t>(blocks, 4096));
}

// Read a boolean env var. Treats unset / "0" / "false" / "no" / empty as
// false; everything else as true. Mirrors the convention used elsewhere in
// the SM120 dispatch (see e.g. patch_vllm_deepseekv4.py env handling).
bool env_flag_true(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    if (v[0] == '\0') return false;
    if (std::strcmp(v, "0") == 0) return false;
    if (std::strcmp(v, "false") == 0) return false;
    if (std::strcmp(v, "False") == 0) return false;
    if (std::strcmp(v, "FALSE") == 0) return false;
    if (std::strcmp(v, "no") == 0) return false;
    if (std::strcmp(v, "No") == 0) return false;
    return true;
}

// Returns true ONLY if the env var is explicitly set to a falsy value
// ("0" / "false" / "no" / "False" / "FALSE" / "No"). Unset env vars and
// non-falsy values both return false. Used for "default ON" toggles where
// we only want to disable on an explicit opt-out.
bool env_flag_explicitly_false(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    if (v[0] == '\0') return false;
    if (std::strcmp(v, "0") == 0) return true;
    if (std::strcmp(v, "false") == 0) return true;
    if (std::strcmp(v, "False") == 0) return true;
    if (std::strcmp(v, "FALSE") == 0) return true;
    if (std::strcmp(v, "no") == 0) return true;
    if (std::strcmp(v, "No") == 0) return true;
    return false;
}

} // namespace

void sm120_fp8_mqa_logits_v2(
    const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim) {
    DG_HOST_ASSERT(q.is_contiguous());
    DG_HOST_ASSERT(kv.is_contiguous());
    DG_HOST_ASSERT(kv_sf.is_contiguous());
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());
    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv_sf.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    // C3: dispatch to the BF16 m16n8k16 tensor-core path by default.
    //
    //   DG_SM120_MQA_LOGITS_V2_MMA=0      -> force scalar inner (escape).
    //   DG_SM120_MQA_LOGITS_V2_MMA unset
    //                       or anything   -> use MMA inner (default ON).
    //
    // The MMA launch returns ``false`` for unsupported shapes (head_dim != 64,
    // num_heads > 64, unsupported dtype). In that case:
    //   DG_SM120_MQA_LOGITS_V2_STRICT=1   -> hard-fail (dev mode, catches
    //                                        silent fallbacks).
    //   otherwise                         -> silently fall through to the
    //                                        scalar inner below (production
    //                                        mode, preserves correctness for
    //                                        any shape the v2 entry accepts).
    const bool prefer_mma =
        !env_flag_explicitly_false("DG_SM120_MQA_LOGITS_V2_MMA");
    bool mma_launched = false;
    if (prefer_mma) {
        mma_launched = sm120_fp8_mqa_logits_v2_mma_try_launch(
            q, kv, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end,
            logits, logits_dtype, seq_len, seq_len_kv, max_seqlen_k,
            logits_stride, num_heads, head_dim);
    }

    // One-time diagnostic for the first few calls per process: prints the
    // live shape and which inner was selected. This is the only way to
    // confirm the MMA fast path is actually running on the live sparse
    // indexer shape (the synthetic test uses head_dim=64 / num_heads=32,
    // but the live shape is set by the model config, not the test). Capped
    // at 4 calls so it never spams logs in production. Triggered when
    // ``DG_SM120_MQA_LOGITS_V2_TRACE=1``; default OFF so production logs
    // are not polluted.
    if (env_flag_true("DG_SM120_MQA_LOGITS_V2_TRACE")) {
        static std::atomic<int> trace_count{0};
        const int prev = trace_count.fetch_add(1, std::memory_order_relaxed);
        if (prev < 4) {
            const char* dtype_name = "?";
            if (logits_dtype == torch::kFloat32) dtype_name = "fp32";
            else if (logits_dtype == torch::kBFloat16) dtype_name = "bf16";
            const char* path_name = "scalar(forced)";
            if (prefer_mma) path_name = mma_launched ? "MMA" : "scalar(MMA-rejected)";
            std::fprintf(stderr,
                "[sm120_mqa_v2_trace #%d] seq_len=%d seq_len_kv=%d "
                "num_heads=%d head_dim=%d max_seqlen_k=%d dtype=%s path=%s\n",
                prev, seq_len, seq_len_kv, num_heads, head_dim, max_seqlen_k,
                dtype_name, path_name);
            std::fflush(stderr);
        }
    }

    if (mma_launched) return;
    if (prefer_mma && env_flag_true("DG_SM120_MQA_LOGITS_V2_STRICT")) {
        const char* dtype_name = "?";
        if (logits_dtype == torch::kFloat32) dtype_name = "fp32";
        else if (logits_dtype == torch::kBFloat16) dtype_name = "bf16";
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "DG_SM120_MQA_LOGITS_V2_STRICT=1: MMA path could not handle this "
            "shape: seq_len=%d seq_len_kv=%d num_heads=%d head_dim=%d "
            "dtype=%s. Supported window: head_dim==64, num_heads<=64, dtype "
            "in {fp32, bf16}. Either widen the MMA kernel's supported shapes "
            "to include this shape, or unset STRICT to silently fall back to "
            "the scalar inner.", seq_len, seq_len_kv, num_heads, head_dim,
            dtype_name);
        DG_HOST_UNREACHABLE(buf);
    }
    // Else fall through to scalar (silent fallback for unsupported shapes).

    const int out_cols = max_seqlen_k > 0 ? max_seqlen_k : seq_len_kv;
    const bool compressed_logits = max_seqlen_k > 0;
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t total = static_cast<int64_t>(seq_len) * out_cols;
    const int grid = v2_grid(total);

    if (logits_dtype == torch::kFloat32) {
        fp8_mqa_logits_v2_scalar_kernel<float>
            <<<grid, threads, 0, stream>>>(
                reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
                reinterpret_cast<const __nv_fp8_e4m3*>(kv.data_ptr()),
                kv_sf.data_ptr<float>(), weights.data_ptr<float>(),
                cu_seq_len_k_start.data_ptr<int32_t>(),
                cu_seq_len_k_end.data_ptr<int32_t>(),
                logits.data_ptr<float>(), seq_len, seq_len_kv, num_heads,
                head_dim, out_cols, logits_stride, compressed_logits);
    } else if (logits_dtype == torch::kBFloat16) {
        fp8_mqa_logits_v2_scalar_kernel<__nv_bfloat16>
            <<<grid, threads, 0, stream>>>(
                reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
                reinterpret_cast<const __nv_fp8_e4m3*>(kv.data_ptr()),
                kv_sf.data_ptr<float>(), weights.data_ptr<float>(),
                cu_seq_len_k_start.data_ptr<int32_t>(),
                cu_seq_len_k_end.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),
                seq_len, seq_len_kv, num_heads, head_dim, out_cols,
                logits_stride, compressed_logits);
    } else {
        DG_HOST_UNREACHABLE("Unsupported logits dtype for SM120 MQA logits v2");
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

void register_mqa_logits_v2_apis(pybind11::module& m) {
    m.def("sm120_fp8_mqa_logits_v2", &sm120_fp8_mqa_logits_v2,
          pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("kv_sf"),
          pybind11::arg("weights"), pybind11::arg("cu_seq_len_k_start"),
          pybind11::arg("cu_seq_len_k_end"), pybind11::arg("logits"),
          pybind11::arg("logits_dtype"), pybind11::arg("seq_len"),
          pybind11::arg("seq_len_kv"), pybind11::arg("max_seqlen_k"),
          pybind11::arg("logits_stride"), pybind11::arg("num_heads"),
          pybind11::arg("head_dim"),
          "SM120 FP8 MQA logits v2 (non-paged): replacement for "
          "sm120_fp8_mqa_logits_fallback. Default path is the C2a BF16 "
          "m16n8k16 tensor-core inner for supported shapes (head_dim=64, "
          "num_heads<=64). Set DG_SM120_MQA_LOGITS_V2_MMA=0 to force the "
          "C1 scalar inner (e.g. for A/B comparison). Unsupported shapes "
          "transparently fall back to the scalar inner unless "
          "DG_SM120_MQA_LOGITS_V2_STRICT=1 (in which case they raise a "
          "host error to catch silent shape fallbacks).");
}

} // namespace sm120_mla_v2
} // namespace deep_gemm
