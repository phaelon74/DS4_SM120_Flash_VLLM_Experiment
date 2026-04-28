// SM120 MQA logits v2 (FP8 non-paged).
//
// C1: this scaffold implemented the v2 entry point with a scalar inner that
// is bit-exact to ``deep_gemm::sm120_fallback::mqa_logits_kernel``. C2a
// adds a BF16 m16n8k16 tensor-core path (``sm120_mqa_logits_v2_mma.cu``);
// the host launcher below dispatches to it when ``DG_SM120_MQA_LOGITS_V2_MMA=1``
// is set in the environment. The MMA path is opt-in for C2a so that we can
// land it, validate correctness in isolation, and microbench it before
// promoting to default-on (and live wire-up) in C3.
//
// Profile attribution (live torch trace, single request, rank 0):
//   prompt 4k:   961.557 ms /  42 calls  (44% of 2.18s TTFT)
//   prompt 8k:  3646.926 ms /  63 calls  (62% of 5.87s TTFT)
//   prompt 16k: 13962.641 ms / 105 calls (76% of 18.44s TTFT)
// At the 16k size the existing scalar averages ~133 ms / call; the MMA
// path's theoretical floor (BF16 tensor-core at ~30% of peak) is well
// under 200 us / call.

#include <algorithm>
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
// C2 will replace the body below the ``--- C2 MMA REPLACEMENT ABOVE ---``
// marker with a BF16 m16n8k16 mma.sync implementation while keeping this
// host-side launch path unchanged.
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
// --- C2 MMA REPLACEMENT ABOVE ---

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

    // C2a: env-gated dispatch to the BF16 m16n8k16 tensor-core path.
    // If ``DG_SM120_MQA_LOGITS_V2_MMA=1`` is set AND the shape is supported,
    // run the MMA kernel and return. Otherwise fall through to the scalar
    // path below. The MMA launch returns ``false`` for unsupported shapes
    // (head_dim != 64, num_heads > 64, etc.) so the scalar remains a
    // correctness-preserving fallback for all inputs.
    if (env_flag_true("DG_SM120_MQA_LOGITS_V2_MMA")) {
        const bool launched = sm120_fp8_mqa_logits_v2_mma_try_launch(
            q, kv, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end,
            logits, logits_dtype, seq_len, seq_len_kv, max_seqlen_k,
            logits_stride, num_heads, head_dim);
        if (launched) return;
        // Fell through: shape unsupported by the MMA path. The scalar
        // dispatch below handles every input the v2 entry point accepts.
    }

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
          "sm120_fp8_mqa_logits_fallback. Default path is the C1 scalar "
          "inner (bit-exact to the existing fallback). Set the environment "
          "variable DG_SM120_MQA_LOGITS_V2_MMA=1 to dispatch through the "
          "C2a BF16 m16n8k16 tensor-core kernel for supported shapes "
          "(head_dim=64, num_heads<=64); unsupported shapes transparently "
          "fall back to the scalar inner.");
}

} // namespace sm120_mla_v2
} // namespace deep_gemm
