// SM120 Fused sparse MLA prefill (v2)
//
// Replaces the BF16 workspace + torch.bmm sparse-prefill bridge with a single
// fused kernel that reads directly from the FP8 ds_mla KV cache via a
// workspace_map (one int32 per workspace row giving the physical KV slot).
//
// Per (sequence_token, head) we do:
//   1. Stage Q in shared memory (head_dim, fp32)
//   2. Iterate sparse columns in chunks; each chunk:
//        - Resolve workspace_map -> physical linear slot per candidate
//        - Score Q*K^T (scalar over FP8 with UE8M0 scale + BF16 RoPE tail)
//        - FA2-style online softmax with running max/sum
//        - P*V accumulate into per-thread fp32 partial vector
//   3. Emit BF16 out and fp32 LSE in one launch
//
// The structural fusion is what wins; the inner Q*K^T and P*V loops are
// scalar today. ``// TODO(SM120-MMA):`` blocks mark where to slot in
// warp-level ``mma.sync.aligned.kind::f8f6f4.block_scale`` instructions once
// the structural path is validated on hardware.
//
// This kernel intentionally mirrors the decode v2 kernel layout so they can
// share a future MMA upgrade.

#include <algorithm>
#include <cstdint>
#include <limits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "jit_kernels/impls/sm120_mqa_logits_v2.hpp"
#include "jit_kernels/impls/sm120_sparse_mla_decode_v2.hpp"
#include "jit_kernels/impls/sm120_sparse_mla_prefill_v2.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace sm120_mla_v2 {
namespace {

constexpr int kHeadDim = 512;
constexpr int kFp8Dim = 448;
constexpr int kBf16Dim = 64;
constexpr int kQuantBlock = 64;
constexpr int kTokenDataBytes = kFp8Dim + kBf16Dim * 2;
constexpr int kScaleBytes = 8;
constexpr int kThreads = 256;

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

__device__ __forceinline__ float decode_ue8m0_scale(uint8_t exponent) {
    if (exponent == 0)
        return 0.0f;
    return exp2f(static_cast<float>(exponent) - 127.0f);
}

template <typename T>
__device__ __forceinline__ float load_q_value(const T* q, int64_t offset);
template <>
__device__ __forceinline__ float load_q_value<__nv_bfloat16>(
    const __nv_bfloat16* q, int64_t offset) {
    return __bfloat162float(q[offset]);
}
template <>
__device__ __forceinline__ float load_q_value<half>(const half* q,
                                                    int64_t offset) {
    return __half2float(q[offset]);
}

template <typename T>
__device__ __forceinline__ void store_out_value(T* out, int64_t offset,
                                                float value);
template <>
__device__ __forceinline__ void store_out_value<__nv_bfloat16>(
    __nv_bfloat16* out, int64_t offset, float value) {
    out[offset] = __float2bfloat16(value);
}
template <>
__device__ __forceinline__ void store_out_value<half>(half* out, int64_t offset,
                                                      float value) {
    out[offset] = __float2half(value);
}

__device__ __forceinline__ const uint8_t* token_ptr_from_linear(
    const uint8_t* cache_flat, int64_t block_stride_bytes, int block_size,
    int64_t linear_index) {
    const int64_t block_id = linear_index / block_size;
    const int block_offset = static_cast<int>(linear_index - block_id * block_size);
    const uint8_t* block = cache_flat + block_id * block_stride_bytes;
    return block + static_cast<int64_t>(block_offset) * kTokenDataBytes;
}
__device__ __forceinline__ const uint8_t* scale_ptr_from_linear(
    const uint8_t* cache_flat, int64_t block_stride_bytes, int block_size,
    int64_t linear_index) {
    const int64_t block_id = linear_index / block_size;
    const int block_offset = static_cast<int>(linear_index - block_id * block_size);
    const uint8_t* block = cache_flat + block_id * block_stride_bytes;
    return block + static_cast<int64_t>(block_size) * kTokenDataBytes +
           static_cast<int64_t>(block_offset) * kScaleBytes;
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    if (lane == 0)
        smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (kThreads >> 5) ? smem[lane] : -INFINITY;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        }
        if (lane == 0)
            smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    if (lane == 0)
        smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (kThreads >> 5) ? smem[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off);
        }
        if (lane == 0)
            smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

template <typename QT, typename OutT, typename IdxT>
__global__ void __launch_bounds__(kThreads, 2) sm120_fused_prefill_v2_kernel(
    const QT* __restrict__ q,                  // [S, H, head_dim]
    const uint8_t* __restrict__ cache,          // packed FP8 ds_mla cache
    const int* __restrict__ workspace_map,      // [Nw], -1 == invalid
    const IdxT* __restrict__ indices,           // [S, 1, K]
    const int* __restrict__ topk_lengths,       // [S] or nullptr
    const float* __restrict__ attn_sink,        // [H] or nullptr
    OutT* __restrict__ out,                     // [S, H, head_dim]
    float* __restrict__ lse_out,                // [S, H]
    int S, int H, int K,
    int64_t q_stride_s, int64_t q_stride_h,
    int64_t out_stride_s, int64_t out_stride_h,
    int64_t lse_stride_s,
    int64_t cache_block_stride_bytes,
    int block_size,
    int64_t indices_stride_s,
    float softmax_scale) {
    const int s = blockIdx.x;   // sequence-row
    const int h = blockIdx.y;   // head
    const int tid = threadIdx.x;

    const int klen = topk_lengths ? min(K, topk_lengths[s]) : K;

    extern __shared__ unsigned char smem_raw[];
    float* q_shared = reinterpret_cast<float*>(smem_raw);
    float* warp_red = q_shared + kHeadDim;

    const QT* q_row = q + s * q_stride_s + h * q_stride_h;
    for (int d = tid; d < kHeadDim; d += blockDim.x) {
        q_shared[d] = load_q_value<QT>(q_row, d);
    }
    __syncthreads();

    constexpr int kDimsPerThread = kHeadDim / kThreads;  // 2
    float out_acc[kDimsPerThread];
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
        out_acc[i] = 0.0f;
    }
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    const IdxT* idx_row = indices + s * indices_stride_s;

    constexpr int kCandidatesPerChunk = 32;
    __shared__ float chunk_logits[kCandidatesPerChunk];
    __shared__ int64_t chunk_linear[kCandidatesPerChunk];
    __shared__ uint8_t chunk_scales[kCandidatesPerChunk][8];

    for (int kc0 = 0; kc0 < klen; kc0 += kCandidatesPerChunk) {
        const int chunk_n = min(kCandidatesPerChunk, klen - kc0);

        // (a) Resolve workspace_map -> physical KV slot for each candidate.
        if (tid < chunk_n) {
            const IdxT raw = idx_row[kc0 + tid];
            int64_t lin = -1;
            if (raw >= 0) {
                const int wmap_v = workspace_map[raw];
                if (wmap_v >= 0)
                    lin = static_cast<int64_t>(wmap_v);
            }
            chunk_linear[tid] = lin;
            if (lin >= 0) {
                const uint8_t* sptr = scale_ptr_from_linear(
                    cache, cache_block_stride_bytes, block_size, lin);
#pragma unroll
                for (int sb = 0; sb < 8; ++sb) {
                    chunk_scales[tid][sb] = sptr[sb];
                }
            }
        }
        __syncthreads();

        // (b) Compute Q*K^T per candidate using 8 lanes per candidate.
        // TODO(SM120-MMA): replace the inner loop with warp-level
        // ``mma.sync.aligned.kind::f8f6f4.block_scale`` MMAs once validated.
        constexpr int kThreadsPerCand = kThreads / kCandidatesPerChunk;  // 8
        const int cand = tid / kThreadsPerCand;
        const int sublane = tid % kThreadsPerCand;

        float partial = 0.0f;
        if (cand < chunk_n) {
            const int64_t lin = chunk_linear[cand];
            if (lin >= 0) {
                const uint8_t* token = token_ptr_from_linear(
                    cache, cache_block_stride_bytes, block_size, lin);
                const auto* rope =
                    reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
                constexpr int kDimsPerSublane = kHeadDim / kThreadsPerCand;
                const int d0 = sublane * kDimsPerSublane;
#pragma unroll
                for (int dd = 0; dd < kDimsPerSublane; ++dd) {
                    const int d = d0 + dd;
                    float kv;
                    if (d < kFp8Dim) {
                        const float sc = decode_ue8m0_scale(
                            chunk_scales[cand][d / kQuantBlock]);
                        kv = fp8_e4m3fn_to_float(token[d]) * sc;
                    } else {
                        kv = __bfloat162float(rope[d - kFp8Dim]);
                    }
                    partial += q_shared[d] * kv;
                }
            }
        }
        for (int off = kThreadsPerCand >> 1; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(0xffffffffu, partial, off);
        }
        if (sublane == 0 && cand < chunk_n) {
            float logit = (chunk_linear[cand] >= 0)
                              ? partial * softmax_scale
                              : -INFINITY;
            chunk_logits[cand] = logit;
        }
        __syncthreads();

        // (c) Online softmax update (FA2-style).
        float chunk_max = -INFINITY;
        for (int c = tid; c < chunk_n; c += blockDim.x) {
            chunk_max = fmaxf(chunk_max, chunk_logits[c]);
        }
        chunk_max = block_reduce_max(chunk_max, warp_red);

        const bool first_chunk = (kc0 == 0);
        float new_max;
        if (first_chunk) {
            float sink = (attn_sink != nullptr) ? attn_sink[h] : -INFINITY;
            new_max = fmaxf(chunk_max, sink);
        } else {
            new_max = fmaxf(row_max, chunk_max);
        }

        const float scale_old =
            (row_max == -INFINITY) ? 0.0f : __expf(row_max - new_max);
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
            out_acc[i] *= scale_old;
        }
        row_sum *= scale_old;

        float chunk_sum = 0.0f;
        for (int c = tid; c < chunk_n; c += blockDim.x) {
            float p = __expf(chunk_logits[c] - new_max);
            chunk_logits[c] = p;
            chunk_sum += p;
        }
        chunk_sum = block_reduce_sum(chunk_sum, warp_red);
        row_sum += chunk_sum;

        if (first_chunk && attn_sink != nullptr) {
            float sink_p = __expf(attn_sink[h] - new_max);
            row_sum += sink_p;
        }

        row_max = new_max;
        __syncthreads();

        // (d) P*V accumulate per-dim.
        // TODO(SM120-MMA): replace with warp-level FP8 P*V MMA.
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
            const int d = tid + i * blockDim.x;
            if (d >= kHeadDim)
                continue;
            float acc = out_acc[i];
            for (int c = 0; c < chunk_n; ++c) {
                const int64_t lin = chunk_linear[c];
                if (lin < 0)
                    continue;
                const uint8_t* token = token_ptr_from_linear(
                    cache, cache_block_stride_bytes, block_size, lin);
                float kv;
                if (d < kFp8Dim) {
                    const float sc = decode_ue8m0_scale(
                        chunk_scales[c][d / kQuantBlock]);
                    kv = fp8_e4m3fn_to_float(token[d]) * sc;
                } else {
                    const auto* rope = reinterpret_cast<const __nv_bfloat16*>(
                        token + kFp8Dim);
                    kv = __bfloat162float(rope[d - kFp8Dim]);
                }
                acc += chunk_logits[c] * kv;
            }
            out_acc[i] = acc;
        }
        __syncthreads();
    }

    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    OutT* out_row = out + s * out_stride_s + h * out_stride_h;
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
        const int d = tid + i * blockDim.x;
        if (d < kHeadDim) {
            store_out_value<OutT>(out_row, d, out_acc[i] * inv_sum);
        }
    }
    if (tid == 0) {
        const float lse =
            (row_sum > 0.0f) ? (logf(row_sum) + row_max) : -INFINITY;
        lse_out[s * lse_stride_s + h] = lse;
    }
}

template <typename QT, typename OutT, typename IdxT>
void launch_sm120_fused_prefill_v2(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& workspace_map,
    const torch::Tensor& indices,
    const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink,
    torch::Tensor& out,
    torch::Tensor& lse,
    int block_size,
    float softmax_scale) {
    const int S = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int K = static_cast<int>(indices.size(2));

    const auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(S, H);
    dim3 block(kThreads);
    const size_t shared_bytes = static_cast<size_t>(kHeadDim + 8) * sizeof(float);

    const int* topk_ptr =
        topk_length.defined() ? topk_length.data_ptr<int>() : nullptr;
    const float* sink_ptr =
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr;

    int64_t cache_block_stride_bytes;
    if (k_cache.dim() >= 4) {
        cache_block_stride_bytes = static_cast<int64_t>(k_cache.stride(0));
    } else {
        cache_block_stride_bytes =
            static_cast<int64_t>(k_cache.stride(0)) * k_cache.element_size();
    }

    sm120_fused_prefill_v2_kernel<QT, OutT, IdxT><<<grid, block, shared_bytes,
                                                    stream>>>(
        reinterpret_cast<const QT*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
        workspace_map.data_ptr<int>(),
        indices.data_ptr<IdxT>(),
        topk_ptr,
        sink_ptr,
        reinterpret_cast<OutT*>(out.data_ptr()),
        lse.data_ptr<float>(),
        S, H, K,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        lse.stride(0),
        cache_block_stride_bytes,
        block_size,
        indices.stride(0),
        softmax_scale);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

inline torch::Tensor tensor_or_empty(const pybind11::object& obj) {
    if (obj.is_none())
        return {};
    return obj.cast<torch::Tensor>();
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_mla_prefill_v2(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& workspace_map,
    const torch::Tensor& indices,
    const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj,
    int block_size,
    int head_dim_v,
    double softmax_scale,
    const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && k_cache.is_cuda() && indices.is_cuda() &&
                   workspace_map.is_cuda());
    DG_HOST_ASSERT(q.dim() == 3 && q.size(2) == kHeadDim);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(workspace_map.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(workspace_map.dim() == 1);
    DG_HOST_ASSERT(indices.dim() == 3 && indices.size(0) == q.size(0) &&
                   indices.size(1) == 1);
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >=
                   kTokenDataBytes + kScaleBytes);

    auto topk_length = tensor_or_empty(topk_length_obj);
    if (topk_length.defined()) {
        DG_HOST_ASSERT(topk_length.is_cuda());
        DG_HOST_ASSERT(topk_length.numel() >= q.size(0));
        if (topk_length.scalar_type() == torch::kInt64)
            topk_length = topk_length.to(torch::kInt32);
    }
    auto attn_sink = tensor_or_empty(attn_sink_obj);
    if (attn_sink.defined() && attn_sink.scalar_type() != torch::kFloat32)
        attn_sink = attn_sink.to(torch::kFloat32);

    torch::Tensor out;
    if (out_obj.is_none()) {
        out = torch::empty({q.size(0), q.size(1), head_dim_v}, q.options());
    } else {
        out = out_obj.cast<torch::Tensor>();
    }
    auto max_logits =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));
    auto lse =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    const float scale = static_cast<float>(softmax_scale);

    if (q.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt32) {
            launch_sm120_fused_prefill_v2<__nv_bfloat16, __nv_bfloat16, int32_t>(
                q, k_cache, workspace_map, indices, topk_length, attn_sink,
                out, lse, block_size, scale);
        } else if (indices.scalar_type() == torch::kInt64) {
            launch_sm120_fused_prefill_v2<__nv_bfloat16, __nv_bfloat16, int64_t>(
                q, k_cache, workspace_map, indices, topk_length, attn_sink,
                out, lse, block_size, scale);
        } else {
            DG_HOST_UNREACHABLE("v2 prefill indices must be int32/int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt32) {
            launch_sm120_fused_prefill_v2<half, half, int32_t>(
                q, k_cache, workspace_map, indices, topk_length, attn_sink,
                out, lse, block_size, scale);
        } else if (indices.scalar_type() == torch::kInt64) {
            launch_sm120_fused_prefill_v2<half, half, int64_t>(
                q, k_cache, workspace_map, indices, topk_length, attn_sink,
                out, lse, block_size, scale);
        } else {
            DG_HOST_UNREACHABLE("v2 prefill indices must be int32/int64");
        }
    } else {
        DG_HOST_UNREACHABLE("v2 prefill requires matching bf16 or fp16 q/out");
    }

    max_logits.fill_(std::numeric_limits<float>::quiet_NaN());
    return std::make_tuple(out, max_logits, lse);
}

void register_prefill_v2_apis(pybind11::module& m) {
    m.def("sm120_sparse_mla_prefill_v2", &sparse_mla_prefill_v2,
          "SM120 fused sparse MLA prefill (v2): single CTA per (S,H), "
          "FP8 cache direct via workspace_map, online softmax, no BF16 "
          "workspace");
}

void register_apis(pybind11::module& m) {
    register_decode_v2_apis(m);
    register_prefill_v2_apis(m);
    register_mqa_logits_v2_apis(m);
}

} // namespace sm120_mla_v2
} // namespace deep_gemm
