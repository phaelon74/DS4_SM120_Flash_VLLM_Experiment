// SM120 HyperConnection prenorm fallback (N > 64 path).
//
// dsl12x Phase 5c: this file is the fallback path for N > 64. The MMA
// restructure planned in Phase 5b applies here too, with two added
// considerations:
//
//   (a) N > 64 means at least 8 N sub-tiles of 8 per CTA. Two valid
//       layouts:
//         * one CTA per (split, m_tile_of_16) processing all N sub-tiles
//           in a sequential warp loop (fewer CTAs, more SMEM staging
//           per CTA). Recommended for N <= 128 based on the trace from
//           Phase 5a.
//         * one CTA per (split, m_tile_of_16, n_tile_of_8) (more CTAs,
//           less per-CTA work). Recommended for N >= 256.
//   (b) The fused per-row sum-of-squares stays bounded per CTA at 16
//       FP32 lanes regardless of N tiling.
//
// For now this fallback file holds the existing scalar implementation
// untouched. The Phase 5c MMA restructure will:
//
//   1. Add an mma_tf32_m16n8k8 helper (same PTX as
//      csrc/sm120_tf32_hc_prenorm_gemm.cu line ~70).
//   2. Add a templated hc_prenorm_mma_fallback_kernel<int kN_TILES> that
//      dispatches based on the chosen layout above.
//   3. Add a DG_SM120_HC_PRENORM_FALLBACK_MMA env-var gate in the host
//      dispatcher (mirrors DG_SM120_HC_PRENORM_V2_MMA from v2 path).
//   4. Default OFF until correctness + perf is validated on test system.

#include <algorithm>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_hc_prenorm_fallback.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace sm120_fallback {

__device__ __forceinline__ int split_k_begin(int k, int num_splits,
                                             int split_idx) {
    constexpr int block_k = 64;
    const int k_blocks = (k + block_k - 1) / block_k;
    const int blocks_per_split = k_blocks / num_splits;
    const int remain_blocks = k_blocks - blocks_per_split * num_splits;
    return (split_idx * blocks_per_split + min(split_idx, remain_blocks)) *
           block_k;
}

__device__ __forceinline__ int split_k_end(int k, int num_splits,
                                           int split_idx) {
    return min(k, split_k_begin(k, num_splits, split_idx + 1));
}

__device__ __forceinline__ float round_to_tf32(float value) {
    uint32_t bits = __float_as_uint(value);
    const uint32_t abs_bits = bits & 0x7fffffffu;
    if (abs_bits >= 0x7f800000u)
        return value;

    // Round FP32 mantissa to TF32's 10 explicit mantissa bits.
    const uint32_t lsb = (bits >> 13) & 1u;
    bits += 0x0fffu + lsb;
    bits &= 0xffffe000u;
    return __uint_as_float(bits);
}

__global__ void hc_prenorm_gemm_kernel(const __nv_bfloat16* a,
                                       const float* b, float* d,
                                       int64_t a_stride_m,
                                       int64_t a_stride_k,
                                       int64_t b_stride_n,
                                       int64_t b_stride_k,
                                       int64_t d_stride_split,
                                       int64_t d_stride_m,
                                       int64_t d_stride_n,
                                       int m, int n, int k,
                                       int num_splits) {
    const int64_t total =
        static_cast<int64_t>(num_splits) * static_cast<int64_t>(m) * n;
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int col = static_cast<int>(linear % n);
        const int64_t row_linear = linear / n;
        const int row = static_cast<int>(row_linear % m);
        const int split_idx = static_cast<int>(row_linear / m);
        const int k_begin = split_k_begin(k, num_splits, split_idx);
        const int k_end = split_k_end(k, num_splits, split_idx);

        float sum = 0.0f;
        for (int kk = k_begin; kk < k_end; ++kk) {
            const float av =
                __bfloat162float(a[static_cast<int64_t>(row) * a_stride_m +
                                   static_cast<int64_t>(kk) * a_stride_k]);
            const float bv =
                round_to_tf32(b[static_cast<int64_t>(col) * b_stride_n +
                                static_cast<int64_t>(kk) * b_stride_k]);
            sum += av * bv;
        }

        d[static_cast<int64_t>(split_idx) * d_stride_split +
          static_cast<int64_t>(row) * d_stride_m +
          static_cast<int64_t>(col) * d_stride_n] = sum;
    }
}

__global__ void hc_prenorm_sqr_sum_kernel(const __nv_bfloat16* a,
                                          float* sqr_sum,
                                          int64_t a_stride_m,
                                          int64_t a_stride_k,
                                          int64_t s_stride_split,
                                          int64_t s_stride_m,
                                          int m, int k,
                                          int num_splits) {
    const int64_t total =
        static_cast<int64_t>(num_splits) * static_cast<int64_t>(m);
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(linear % m);
        const int split_idx = static_cast<int>(linear / m);
        const int k_begin = split_k_begin(k, num_splits, split_idx);
        const int k_end = split_k_end(k, num_splits, split_idx);

        float sum = 0.0f;
        for (int kk = k_begin; kk < k_end; ++kk) {
            const float av =
                __bfloat162float(a[static_cast<int64_t>(row) * a_stride_m +
                                   static_cast<int64_t>(kk) * a_stride_k]);
            sum += av * av;
        }

        sqr_sum[static_cast<int64_t>(split_idx) * s_stride_split +
                static_cast<int64_t>(row) * s_stride_m] = sum;
    }
}

__global__ void hc_prenorm_block_reduce_kernel(
    const __nv_bfloat16* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ d, float* __restrict__ sqr_sum,
    int64_t a_stride_m, int64_t a_stride_k, int64_t b_stride_n,
    int64_t b_stride_k, int64_t d_stride_split, int64_t d_stride_m,
    int64_t d_stride_n, int64_t s_stride_split, int64_t s_stride_m, int m,
    int n, int k, int num_splits) {
    extern __shared__ float smem[];

    const int linear = blockIdx.x;
    const int row = linear % m;
    const int split_idx = linear / m;
    if (split_idx >= num_splits)
        return;

    const int k_begin = split_k_begin(k, num_splits, split_idx);
    const int k_end = split_k_end(k, num_splits, split_idx);
    float partial[32];
#pragma unroll
    for (int col = 0; col < 32; ++col)
        partial[col] = 0.0f;
    float sq_partial = 0.0f;

    for (int kk = k_begin + threadIdx.x; kk < k_end; kk += blockDim.x) {
        const float av =
            __bfloat162float(a[static_cast<int64_t>(row) * a_stride_m +
                               static_cast<int64_t>(kk) * a_stride_k]);
        sq_partial += av * av;
        for (int col = 0; col < n; ++col) {
            const float bv =
                round_to_tf32(b[static_cast<int64_t>(col) * b_stride_n +
                                static_cast<int64_t>(kk) * b_stride_k]);
            partial[col] += av * bv;
        }
    }

    for (int col = 0; col < n; ++col)
        smem[col * blockDim.x + threadIdx.x] = partial[col];
    smem[n * blockDim.x + threadIdx.x] = sq_partial;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            for (int col = 0; col < n; ++col) {
                smem[col * blockDim.x + threadIdx.x] +=
                    smem[col * blockDim.x + threadIdx.x + offset];
            }
            smem[n * blockDim.x + threadIdx.x] +=
                smem[n * blockDim.x + threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int col = 0; col < n; ++col) {
            d[static_cast<int64_t>(split_idx) * d_stride_split +
              static_cast<int64_t>(row) * d_stride_m +
              static_cast<int64_t>(col) * d_stride_n] =
                smem[col * blockDim.x];
        }
        sqr_sum[static_cast<int64_t>(split_idx) * s_stride_split +
                static_cast<int64_t>(row) * s_stride_m] =
            smem[n * blockDim.x];
    }
}

int hc_fallback_grid(int64_t total) {
    constexpr int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    return static_cast<int>(std::min<int64_t>(blocks, 4096));
}

} // namespace sm120_fallback

void sm120_tf32_hc_prenorm_gemm_fallback(const torch::Tensor& a,
                                         const torch::Tensor& b,
                                         const torch::Tensor& d,
                                         const torch::Tensor& sqr_sum,
                                         int m, int n, int k,
                                         int num_splits) {
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t d_total =
        static_cast<int64_t>(num_splits) * static_cast<int64_t>(m) * n;
    const int64_t s_total =
        static_cast<int64_t>(num_splits) * static_cast<int64_t>(m);

    const int64_t d_stride_split = num_splits == 1 ? 0 : d.stride(0);
    const int64_t d_stride_m = num_splits == 1 ? d.stride(0) : d.stride(1);
    const int64_t d_stride_n = num_splits == 1 ? d.stride(1) : d.stride(2);
    const int64_t s_stride_split = num_splits == 1 ? 0 : sqr_sum.stride(0);
    const int64_t s_stride_m = num_splits == 1 ? sqr_sum.stride(0)
                                               : sqr_sum.stride(1);

    if (n <= 32) {
        const int64_t total_blocks =
            static_cast<int64_t>(num_splits) * static_cast<int64_t>(m);
        const size_t shared_bytes =
            static_cast<size_t>(n + 1) * threads * sizeof(float);
        sm120_fallback::hc_prenorm_block_reduce_kernel<<<
            static_cast<unsigned>(total_blocks), threads, shared_bytes,
            stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
            b.data_ptr<float>(), d.data_ptr<float>(),
            sqr_sum.data_ptr<float>(), a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), d_stride_split, d_stride_m, d_stride_n,
            s_stride_split, s_stride_m, m, n, k, num_splits);
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return;
    }

    sm120_fallback::hc_prenorm_gemm_kernel<<<
        sm120_fallback::hc_fallback_grid(d_total), threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), b.data_ptr<float>(),
        d.data_ptr<float>(), a.stride(0), a.stride(1), b.stride(0),
        b.stride(1), d_stride_split, d_stride_m, d_stride_n, m, n, k,
        num_splits);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    sm120_fallback::hc_prenorm_sqr_sum_kernel<<<
        sm120_fallback::hc_fallback_grid(s_total), threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        sqr_sum.data_ptr<float>(), a.stride(0), a.stride(1), s_stride_split,
        s_stride_m, m, k, num_splits);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

} // namespace deep_gemm
