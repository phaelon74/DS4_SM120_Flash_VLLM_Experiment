#include <algorithm>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_mqa_logits_fallback.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace sm120_fallback {

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t raw) {
    const uint8_t mag = raw & 0x7fu;
    if (mag == 0)
        return 0.0f;
    const int fp8_exp = static_cast<int>((mag >> 3) & 0x0fu);
    const int mant = static_cast<int>(mag & 0x07u);
    const float value =
        fp8_exp == 0
            ? ldexpf(static_cast<float>(mant), -9)
            : ldexpf(1.0f + static_cast<float>(mant) * 0.125f, fp8_exp - 7);
    return (raw & 0x80u) ? -value : value;
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t code) {
    const uint8_t value_idx = code & 0x07;
    float value = 0.0f;
    switch (value_idx) {
        case 0: value = 0.0f; break;
        case 1: value = 0.5f; break;
        case 2: value = 1.0f; break;
        case 3: value = 1.5f; break;
        case 4: value = 2.0f; break;
        case 5: value = 3.0f; break;
        case 6: value = 4.0f; break;
        default: value = 6.0f; break;
    }
    return (code & 0x08) && value_idx != 0 ? -value : value;
}

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

template <typename out_t>
__global__ void paged_fp8_mqa_logits_fast_kernel(
    const uint8_t* q, const uint8_t* kv, const float* kv_sf,
    const float* weights, const int32_t* context_lens,
    const int32_t* block_table, out_t* logits, int batch_size, int next_n,
    int num_heads, int head_dim, int block_kv, int kv_stride0,
    int kv_stride1, int kv_sf_stride0, int block_table_stride,
    int logits_stride, int max_context_len, bool is_context_lens_2d) {
    __shared__ float partials[256];
    __shared__ float head_values[64];

    const int n = blockIdx.x;
    const int row = blockIdx.y;
    const int b = row / next_n;
    const int t = row - b * next_n;
    if (b >= batch_size || n >= max_context_len)
        return;

    const int q_limit = is_context_lens_2d
                            ? context_lens[b * next_n + t]
                            : context_lens[b] - next_n + t;
    float result = -INFINITY;
    if (n <= q_limit) {
        const int block_offset = n / block_kv;
        const int token_offset = n - block_offset * block_kv;
        const int block_idx = block_table[b * block_table_stride + block_offset];
        const float k_scale = kv_sf[block_idx * kv_sf_stride0 + token_offset];
        const int group_threads = 256 / num_heads;
        const int h = threadIdx.x / group_threads;
        const int lane = threadIdx.x - h * group_threads;
        float local = 0.0f;
        if (h < num_heads) {
            const int64_t q_base =
                static_cast<int64_t>(row * num_heads + h) * head_dim;
            const int kv_base = block_idx * kv_stride0 + token_offset * kv_stride1;
            for (int d = lane; d < head_dim; d += group_threads) {
                local += fp8_e4m3fn_to_float(q[q_base + d]) *
                         (fp8_e4m3fn_to_float(kv[kv_base + d]) * k_scale);
            }
        }
        partials[threadIdx.x] = local;
        __syncthreads();

        if (h < num_heads && lane == 0) {
            float dot = 0.0f;
            for (int i = 0; i < group_threads; ++i)
                dot += partials[h * group_threads + i];
            head_values[h] = fmaxf(dot, 0.0f) * weights[row * num_heads + h];
        }
        __syncthreads();

        float sum = 0.0f;
        for (int i = threadIdx.x; i < num_heads; i += blockDim.x)
            sum += head_values[i];
        partials[threadIdx.x] = sum;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset)
                partials[threadIdx.x] += partials[threadIdx.x + offset];
            __syncthreads();
        }
        result = partials[0];
    }

    if (threadIdx.x == 0)
        store_logit(logits, static_cast<int64_t>(row) * logits_stride + n,
                    result);
}

__device__ __forceinline__ float ue8m0_scale_from_pack(int32_t packed_sf,
                                                       int group_idx) {
    const auto raw = static_cast<uint32_t>(packed_sf);
    const int exp = static_cast<int>((raw >> (8 * group_idx)) & 0xffu);
    return __uint_as_float(static_cast<uint32_t>(exp) << 23);
}

__device__ __forceinline__ float load_fp4(const int8_t* data, int packed_offset,
                                          int32_t packed_sf, int dim) {
    const uint8_t packed = static_cast<uint8_t>(data[packed_offset + dim / 2]);
    const uint8_t code = (dim & 1) ? (packed >> 4) : (packed & 0x0f);
    return fp4_e2m1_to_float(code) *
           ue8m0_scale_from_pack(packed_sf, dim / 32);
}

template <typename out_t, bool kIsFP4>
__global__ void mqa_logits_kernel(const void* q_ptr, const int32_t* q_sf_ptr,
                                  const void* kv_ptr, const void* kv_sf_ptr,
                                  const float* weights,
                                  const int32_t* cu_seq_len_k_start,
                                  const int32_t* cu_seq_len_k_end,
                                  out_t* logits, int seq_len, int seq_len_kv,
                                  int num_heads, int head_dim,
                                  int packed_head_dim, int out_cols,
                                  int logits_stride, bool compressed_logits) {
    const int64_t total = static_cast<int64_t>(seq_len) * out_cols;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total; linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int m = static_cast<int>(linear / out_cols);
        const int out_col = static_cast<int>(linear - static_cast<int64_t>(m) * out_cols);
        const int start = max(0, min(cu_seq_len_k_start[m], seq_len_kv));
        const int end = max(0, min(cu_seq_len_k_end[m], seq_len_kv));
        const int n = compressed_logits ? start + out_col : out_col;

        float result = -INFINITY;
        if (n >= start && n < end) {
            float sum = 0.0f;
            for (int h = 0; h < num_heads; ++h) {
                float dot = 0.0f;
                if constexpr (kIsFP4) {
                    const auto* q = static_cast<const int8_t*>(q_ptr);
                    const auto* kv = static_cast<const int8_t*>(kv_ptr);
                    const int32_t q_sf = q_sf_ptr[m * num_heads + h];
                    const int32_t kv_sf = static_cast<const int32_t*>(kv_sf_ptr)[n];
                    const int q_base = (m * num_heads + h) * packed_head_dim;
                    const int kv_base = n * packed_head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += load_fp4(q, q_base, q_sf, d) *
                               load_fp4(kv, kv_base, kv_sf, d);
                    }
                } else {
                    const auto* q = static_cast<const __nv_fp8_e4m3*>(q_ptr);
                    const auto* kv = static_cast<const __nv_fp8_e4m3*>(kv_ptr);
                    const float kv_scale = static_cast<const float*>(kv_sf_ptr)[n];
                    const int q_base = (m * num_heads + h) * head_dim;
                    const int kv_base = n * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += static_cast<float>(q[q_base + d]) *
                               (static_cast<float>(kv[kv_base + d]) * kv_scale);
                    }
                }
                sum += fmaxf(dot, 0.0f) * weights[m * num_heads + h];
            }
            result = sum;
        }
        store_logit(logits,
                    static_cast<int64_t>(m) * logits_stride + out_col,
                    result);
    }
}

template <typename out_t, bool kIsFP4>
__global__ void paged_mqa_logits_kernel(
    const void* q_ptr, const int32_t* q_sf_ptr, const void* kv_ptr,
    const void* kv_sf_ptr, const float* weights, const int32_t* context_lens,
    const int32_t* block_table, out_t* logits, int batch_size, int next_n,
    int num_heads, int head_dim, int packed_head_dim, int block_kv,
    int kv_stride0, int kv_stride1, int kv_sf_stride0, int block_table_stride,
    int logits_stride, int max_context_len, bool is_context_lens_2d) {
    const int num_rows = batch_size * next_n;
    const int64_t total = static_cast<int64_t>(num_rows) * max_context_len;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total; linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(linear / max_context_len);
        const int n = static_cast<int>(linear - static_cast<int64_t>(row) * max_context_len);
        const int b = row / next_n;
        const int t = row - b * next_n;
        const int q_limit = is_context_lens_2d
                                ? context_lens[b * next_n + t]
                                : context_lens[b] - next_n + t;

        float result = -INFINITY;
        if (n <= q_limit && n < max_context_len) {
            const int block_offset = n / block_kv;
            const int token_offset = n - block_offset * block_kv;
            const int block_idx = block_table[b * block_table_stride + block_offset];
            float sum = 0.0f;
            for (int h = 0; h < num_heads; ++h) {
                float dot = 0.0f;
                if constexpr (kIsFP4) {
                    const auto* q = static_cast<const int8_t*>(q_ptr);
                    const auto* kv = static_cast<const int8_t*>(kv_ptr);
                    const auto* kv_sf = static_cast<const int32_t*>(kv_sf_ptr);
                    const int32_t q_sf = q_sf_ptr[(row * num_heads) + h];
                    const int32_t k_sf = kv_sf[block_idx * kv_sf_stride0 + token_offset];
                    const int q_base = (row * num_heads + h) * packed_head_dim;
                    const int kv_base = block_idx * kv_stride0 + token_offset * kv_stride1;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += load_fp4(q, q_base, q_sf, d) *
                               load_fp4(kv, kv_base, k_sf, d);
                    }
                } else {
                    const auto* q = static_cast<const __nv_fp8_e4m3*>(q_ptr);
                    const auto* kv = static_cast<const __nv_fp8_e4m3*>(kv_ptr);
                    const auto* kv_sf = static_cast<const float*>(kv_sf_ptr);
                    const float k_scale =
                        kv_sf[block_idx * kv_sf_stride0 + token_offset];
                    const int q_base = (row * num_heads + h) * head_dim;
                    const int kv_base = block_idx * kv_stride0 + token_offset * kv_stride1;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += static_cast<float>(q[q_base + d]) *
                               (static_cast<float>(kv[kv_base + d]) * k_scale);
                    }
                }
                sum += fmaxf(dot, 0.0f) * weights[row * num_heads + h];
            }
            result = sum;
        }
        store_logit(logits, static_cast<int64_t>(row) * logits_stride + n, result);
    }
}

int fallback_grid(int64_t total) {
    constexpr int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    return static_cast<int>(std::min<int64_t>(blocks, 4096));
}

template <bool kIsFP4>
void launch_mqa_logits(const torch::Tensor& q, const torch::Tensor& q_sf,
                       const torch::Tensor& kv, const torch::Tensor& kv_sf,
                       const torch::Tensor& weights,
                       const torch::Tensor& cu_seq_len_k_start,
                       const torch::Tensor& cu_seq_len_k_end,
                       const torch::Tensor& logits,
                       const at::ScalarType& logits_dtype, int seq_len,
                       int seq_len_kv, int num_heads, int head_dim, int out_cols,
                       int logits_stride, bool compressed_logits) {
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t total = static_cast<int64_t>(seq_len) * out_cols;
    const int grid = fallback_grid(total);
    const int packed_head_dim = kIsFP4 ? head_dim / 2 : head_dim;
    const int32_t* q_sf_ptr = nullptr;
    if constexpr (kIsFP4)
        q_sf_ptr = q_sf.data_ptr<int32_t>();

    if (logits_dtype == torch::kFloat32) {
        mqa_logits_kernel<float, kIsFP4><<<grid, threads, 0, stream>>>(
            q.data_ptr(), q_sf_ptr, kv.data_ptr(), kv_sf.data_ptr(),
            weights.data_ptr<float>(), cu_seq_len_k_start.data_ptr<int32_t>(),
            cu_seq_len_k_end.data_ptr<int32_t>(), logits.data_ptr<float>(),
            seq_len, seq_len_kv, num_heads, head_dim, packed_head_dim, out_cols,
            logits_stride, compressed_logits);
    } else if (logits_dtype == torch::kBFloat16) {
        mqa_logits_kernel<__nv_bfloat16, kIsFP4><<<grid, threads, 0, stream>>>(
            q.data_ptr(), q_sf_ptr, kv.data_ptr(), kv_sf.data_ptr(),
            weights.data_ptr<float>(), cu_seq_len_k_start.data_ptr<int32_t>(),
            cu_seq_len_k_end.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()), seq_len,
            seq_len_kv, num_heads, head_dim, packed_head_dim, out_cols,
            logits_stride, compressed_logits);
    } else {
        DG_HOST_UNREACHABLE("Unsupported logits dtype for SM120 fallback");
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

template <bool kIsFP4>
void launch_paged_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& q_sf,
    const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_sf,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& logits, const torch::Tensor& block_table,
    const at::ScalarType& logits_dtype, int batch_size, int next_n,
    int num_heads, int head_dim, int block_kv, bool is_context_lens_2d,
    int logits_stride, int block_table_stride, int max_context_len) {
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t total = static_cast<int64_t>(batch_size) * next_n * max_context_len;
    const int grid = fallback_grid(total);
    const int packed_head_dim = kIsFP4 ? head_dim / 2 : head_dim;
    const int kv_stride0 = static_cast<int>(kv_cache.stride(0));
    const int kv_stride1 = static_cast<int>(kv_cache.stride(1));
    const int kv_sf_stride0 = static_cast<int>(kv_cache_sf.stride(0));
    const int32_t* q_sf_ptr = nullptr;
    if constexpr (kIsFP4)
        q_sf_ptr = q_sf.data_ptr<int32_t>();

    if constexpr (!kIsFP4) {
        const dim3 grid(max_context_len, batch_size * next_n);
        if (logits_dtype == torch::kFloat32) {
            paged_fp8_mqa_logits_fast_kernel<float><<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(q.data_ptr()),
                reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
                kv_cache_sf.data_ptr<float>(), weights.data_ptr<float>(),
                context_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                logits.data_ptr<float>(), batch_size, next_n, num_heads,
                head_dim, block_kv, kv_stride0, kv_stride1, kv_sf_stride0,
                block_table_stride, logits_stride, max_context_len,
                is_context_lens_2d);
        } else if (logits_dtype == torch::kBFloat16) {
            paged_fp8_mqa_logits_fast_kernel<__nv_bfloat16>
                <<<grid, threads, 0, stream>>>(
                    reinterpret_cast<const uint8_t*>(q.data_ptr()),
                    reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
                    kv_cache_sf.data_ptr<float>(), weights.data_ptr<float>(),
                    context_lens.data_ptr<int32_t>(),
                    block_table.data_ptr<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),
                    batch_size, next_n, num_heads, head_dim, block_kv,
                    kv_stride0, kv_stride1, kv_sf_stride0, block_table_stride,
                    logits_stride, max_context_len, is_context_lens_2d);
        } else {
            DG_HOST_UNREACHABLE("Unsupported logits dtype for SM120 fallback");
        }
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    if (logits_dtype == torch::kFloat32) {
        paged_mqa_logits_kernel<float, kIsFP4><<<grid, threads, 0, stream>>>(
            q.data_ptr(), q_sf_ptr, kv_cache.data_ptr(), kv_cache_sf.data_ptr(),
            weights.data_ptr<float>(), context_lens.data_ptr<int32_t>(),
            block_table.data_ptr<int32_t>(), logits.data_ptr<float>(), batch_size,
            next_n, num_heads, head_dim, packed_head_dim, block_kv, kv_stride0,
            kv_stride1, kv_sf_stride0, block_table_stride, logits_stride,
            max_context_len, is_context_lens_2d);
    } else if (logits_dtype == torch::kBFloat16) {
        paged_mqa_logits_kernel<__nv_bfloat16, kIsFP4>
            <<<grid, threads, 0, stream>>>(
                q.data_ptr(), q_sf_ptr, kv_cache.data_ptr(),
                kv_cache_sf.data_ptr(), weights.data_ptr<float>(),
                context_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()), batch_size,
                next_n, num_heads, head_dim, packed_head_dim, block_kv,
                kv_stride0, kv_stride1, kv_sf_stride0, block_table_stride,
                logits_stride, max_context_len, is_context_lens_2d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported logits dtype for SM120 fallback");
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

} // namespace sm120_fallback

void sm120_fp8_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim) {
    const int out_cols = max_seqlen_k > 0 ? max_seqlen_k : seq_len_kv;
    sm120_fallback::launch_mqa_logits<false>(
        q, torch::Tensor(), kv, kv_sf, weights, cu_seq_len_k_start,
        cu_seq_len_k_end, logits, logits_dtype, seq_len, seq_len_kv, num_heads,
        head_dim, out_cols, logits_stride, max_seqlen_k > 0);
}

void sm120_fp4_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& q_sf, const torch::Tensor& kv,
    const torch::Tensor& kv_sf, const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim) {
    const int out_cols = max_seqlen_k > 0 ? max_seqlen_k : seq_len_kv;
    sm120_fallback::launch_mqa_logits<true>(
        q, q_sf, kv, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end,
        logits, logits_dtype, seq_len, seq_len_kv, num_heads, head_dim, out_cols,
        logits_stride, max_seqlen_k > 0);
}

void sm120_fp8_paged_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& kv_cache,
    const torch::Tensor& kv_cache_sf, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& logits,
    const torch::Tensor& block_table, const at::ScalarType& logits_dtype,
    int batch_size, int next_n, int num_heads, int head_dim, int block_kv,
    bool is_context_lens_2d, int logits_stride, int block_table_stride,
    int max_context_len) {
    sm120_fallback::launch_paged_mqa_logits<false>(
        q, torch::Tensor(), kv_cache, kv_cache_sf, weights, context_lens, logits,
        block_table, logits_dtype, batch_size, next_n, num_heads, head_dim,
        block_kv, is_context_lens_2d, logits_stride, block_table_stride,
        max_context_len);
}

void sm120_fp4_paged_mqa_logits_fallback(
    const torch::Tensor& q, const torch::Tensor& q_sf,
    const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_sf,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& logits, const torch::Tensor& block_table,
    const at::ScalarType& logits_dtype, int batch_size, int next_n,
    int num_heads, int head_dim, int block_kv, bool is_context_lens_2d,
    int logits_stride, int block_table_stride, int max_context_len) {
    sm120_fallback::launch_paged_mqa_logits<true>(
        q, q_sf, kv_cache, kv_cache_sf, weights, context_lens, logits, block_table,
        logits_dtype, batch_size, next_n, num_heads, head_dim, block_kv,
        is_context_lens_2d, logits_stride, block_table_stride, max_context_len);
}

} // namespace deep_gemm
