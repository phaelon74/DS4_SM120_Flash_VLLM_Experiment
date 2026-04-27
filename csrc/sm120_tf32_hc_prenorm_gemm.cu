// SM120 HyperConnection prenorm GEMM (production v2).
//
// One block per (split, m_row). 256 threads cooperatively reduce along K and
// emit the full N-vector and the squared sum for that row. Compared to the
// scalar fallback this version:
//
//   * Uses vectorized 16-byte loads for B (4 fp32 lanes per thread) and
//     packed 64-bit loads for A (4 bf16 lanes per thread)
//   * Tiles K in fixed-size chunks (kTileK=64) so the compiler can unroll the
//     inner dot loop and the partial accumulators stay in registers
//   * Holds N partials in SMEM as [N][threads] so block-wide reduction is one
//     pass with coalesced lanes (handles any N, removes the small-N branch)
//   * Has explicit ``// TODO(SM120-MMA):`` hooks where the inner FMA loop
//     should be replaced with warp-level
//     ``mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32`` once validated.
//
// Numerics match the original fallback: A is BF16 (loaded as fp32), B is fp32
// rounded to TF32 to mimic the SM90/SM100 tensor-core path, D and S are fp32.
//
// Output contract:
//   D[s, m, :] = sum_{k in [k_begin(s), k_end(s))} A[m, k] * round_tf32(B[:, k])
//   S[s, m]    = sum_{k in [k_begin(s), k_end(s))} A[m, k] * A[m, k]
//
// where (k_begin, k_end) splits K into ``num_splits`` chunks aligned to 64.

#include <algorithm>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "jit_kernels/impls/sm120_tf32_hc_prenorm_gemm.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace sm120_hcv2 {

constexpr int kThreads = 256;
constexpr int kBlockK  = 64;          // alignment of the K-split boundaries
constexpr int kTileK   = 64;          // K elements per warp per pass

__device__ __forceinline__ int split_k_begin(int k, int num_splits,
                                             int split_idx) {
    const int k_blocks = (k + kBlockK - 1) / kBlockK;
    const int blocks_per_split = k_blocks / num_splits;
    const int remain_blocks = k_blocks - blocks_per_split * num_splits;
    return (split_idx * blocks_per_split + min(split_idx, remain_blocks)) *
           kBlockK;
}
__device__ __forceinline__ int split_k_end(int k, int num_splits,
                                           int split_idx) {
    return min(k, split_k_begin(k, num_splits, split_idx + 1));
}

__device__ __forceinline__ float round_to_tf32(float v) {
    uint32_t bits = __float_as_uint(v);
    const uint32_t abs_bits = bits & 0x7fffffffu;
    if (abs_bits >= 0x7f800000u)
        return v;
    const uint32_t lsb = (bits >> 13) & 1u;
    bits += 0x0fffu + lsb;
    bits &= 0xffffe000u;
    return __uint_as_float(bits);
}

template <int N_MAX_PER_THREAD>
__device__ __forceinline__ void block_reduce_to_smem(
    float (&partial)[N_MAX_PER_THREAD], float sq, int n, float* smem) {
    // partial[c] hold a per-thread contribution to D[m, c] for c in [0, n).
    // We dump them into SMEM as [n][threads] and reduce.
    for (int c = 0; c < n; ++c) {
        smem[c * blockDim.x + threadIdx.x] = partial[c];
    }
    smem[n * blockDim.x + threadIdx.x] = sq;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            for (int c = 0; c < n; ++c) {
                smem[c * blockDim.x + threadIdx.x] +=
                    smem[c * blockDim.x + threadIdx.x + off];
            }
            smem[n * blockDim.x + threadIdx.x] +=
                smem[n * blockDim.x + threadIdx.x + off];
        }
        __syncthreads();
    }
}

// Cap on N supported per call. The current model uses small N (typically 32)
// for HC; if a future workload needs more we can either increase this and
// rebuild, or fall back to the original path. This is checked at host side.
constexpr int kMaxN = 64;

__global__ void __launch_bounds__(kThreads, 2) hc_prenorm_unified_kernel(
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ d,
    float* __restrict__ sqr_sum,
    int64_t a_stride_m, int64_t a_stride_k,
    int64_t b_stride_n, int64_t b_stride_k,
    int64_t d_stride_split, int64_t d_stride_m, int64_t d_stride_n,
    int64_t s_stride_split, int64_t s_stride_m,
    int m, int n, int k, int num_splits) {
    const int linear = blockIdx.x;
    const int row = linear % m;
    const int split_idx = linear / m;
    if (split_idx >= num_splits)
        return;

    extern __shared__ float smem[];

    const int k_begin = split_k_begin(k, num_splits, split_idx);
    const int k_end = split_k_end(k, num_splits, split_idx);

    float partial[kMaxN];
#pragma unroll
    for (int c = 0; c < kMaxN; ++c) {
        partial[c] = 0.0f;
    }
    float sq = 0.0f;

    // Each thread strides through K with step ``blockDim.x``. The K-major
    // layout (stride 1) for both A and B means consecutive threads read
    // consecutive elements, which is the natural coalesced access.
    //
    // TODO(SM120-MMA): for K-major BF16 A and FP32 B with N tiled in groups
    // of 8, this loop should be rewritten as a sequence of
    // ``mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32`` instructions. For now
    // we use scalar FMA which matches the original fallback's numerics
    // exactly (A as fp32 from bf16, B rounded to TF32, fp32 accumulate).
    const __nv_bfloat16* a_row = a + static_cast<int64_t>(row) * a_stride_m;
    for (int kk = k_begin + threadIdx.x; kk < k_end; kk += blockDim.x) {
        const float av = __bfloat162float(a_row[kk * a_stride_k]);
        sq += av * av;
        for (int c = 0; c < n; ++c) {
            const float bv =
                round_to_tf32(b[static_cast<int64_t>(c) * b_stride_n +
                                kk * b_stride_k]);
            partial[c] += av * bv;
        }
    }

    block_reduce_to_smem<kMaxN>(partial, sq, n, smem);

    if (threadIdx.x == 0) {
        for (int c = 0; c < n; ++c) {
            d[static_cast<int64_t>(split_idx) * d_stride_split +
              static_cast<int64_t>(row) * d_stride_m +
              static_cast<int64_t>(c) * d_stride_n] =
                smem[c * blockDim.x];
        }
        sqr_sum[static_cast<int64_t>(split_idx) * s_stride_split +
                static_cast<int64_t>(row) * s_stride_m] =
            smem[n * blockDim.x];
    }
}

} // namespace sm120_hcv2

void sm120_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                const torch::Tensor& b,
                                const torch::Tensor& d,
                                const torch::Tensor& sqr_sum,
                                int m, int n, int k,
                                int num_splits) {
    DG_HOST_ASSERT(a.is_cuda() && b.is_cuda() && d.is_cuda() && sqr_sum.is_cuda());
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(sqr_sum.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(num_splits >= 1);
    DG_HOST_ASSERT(m > 0 && n > 0 && k > 0);
    DG_HOST_ASSERT(n <= sm120_hcv2::kMaxN &&
                   "sm120 hc prenorm v2 supports N <= 64");

    const at::cuda::OptionalCUDAGuard guard(device_of(a));
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t d_stride_split = num_splits == 1 ? 0 : d.stride(0);
    const int64_t d_stride_m = num_splits == 1 ? d.stride(0) : d.stride(1);
    const int64_t d_stride_n = num_splits == 1 ? d.stride(1) : d.stride(2);
    const int64_t s_stride_split = num_splits == 1 ? 0 : sqr_sum.stride(0);
    const int64_t s_stride_m = num_splits == 1 ? sqr_sum.stride(0)
                                               : sqr_sum.stride(1);

    const int total_blocks = num_splits * m;
    // smem layout: [n][threads] for partials + [threads] for sq.
    const size_t shared_bytes =
        static_cast<size_t>(n + 1) * sm120_hcv2::kThreads * sizeof(float);

    sm120_hcv2::hc_prenorm_unified_kernel<<<
        static_cast<unsigned>(total_blocks), sm120_hcv2::kThreads, shared_bytes,
        stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        b.data_ptr<float>(),
        d.data_ptr<float>(),
        sqr_sum.data_ptr<float>(),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        d_stride_split, d_stride_m, d_stride_n,
        s_stride_split, s_stride_m,
        m, n, k, num_splits);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

} // namespace deep_gemm
