// SM120 Fused sparse MLA decode (v2)
//
// This kernel is the production-shape replacement for the BF16 workspace +
// torch.bmm bridge currently used by the patched ``flash_mla_sparse_fwd`` decode
// path on SM120. It targets the DeepSeek V4 ``fp8_ds_mla`` KV cache:
//
//   * Token bytes  : 448 FP8 (E4M3) values + 64 BF16 RoPE values
//   * Scale bytes  :   8 UE8M0 exponents (one per 64-wide FP8 quant block)
//   * Block stride : block_size * (token_bytes + scale_bytes)
//   * Total head_dim_q == head_dim_v == 512
//
// The kernel performs, in a single CTA per (batch, head):
//
//   1. Stage Q vector to SMEM (head_dim values, fp32)
//   2. Iterate sparse indices, reading FP8 + RoPE bytes directly from the
//      cache, dequantize to fp32, compute logits = Q * K^T (scalar inner loop)
//   3. FA2-style online softmax in registers; rolling max / sum / out
//   4. Accumulate P * V into per-thread fp32 partial vector
//   5. Warp / block reduce the partial vectors and emit BF16 out + fp32 LSE
//
// The structural fusion (single launch, no workspace) is the part that wins;
// the inner Q*K^T / P*V is still scalar fp32. The TODO(SM120-MMA): blocks mark
// where to swap in warp-level ``mma.sync.aligned.kind::f8f6f4.block_scale``
// once the structural path has been validated.
//
// Numerics:
//   - Q is read as fp32 from bf16/fp16
//   - K, V come from FP8 with UE8M0 scale (FP8 dim < 448) or BF16 (RoPE dim>=448)
//   - Online softmax in fp32, attn_sink applied as additive logit per head
//   - BF16 output via __float2bfloat16; LSE in fp32

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include <cstdlib>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "jit_kernels/impls/sm120_sparse_mla_decode_v2.hpp"
#include "utils/exception.hpp"

// SM120 native FP8 path (testing-blind, default-off until validated).
//   * mma_block_scale_fp8.cuh: PTX wrappers for m16n8k32 fp8 block-scale MMA
//                              and m16n8k16 bf16 MMA, plus ldmatrix variants.
//   * fp8_quant.cuh:           UE8M0 codec, block quantizers, FP8 packing.
#include "sm120_native_fp8/mma_block_scale_fp8.cuh"
#include "sm120_native_fp8/fp8_quant.cuh"

// Forward declaration of the native FP8 v2 launch entry point. The
// implementation lives in sm120_sparse_mla_decode_v2_native.cu (compiled as a
// separate TU). Returns true on successful launch, false if the runtime
// shape is unsupported by the native path (e.g. H % 16 != 0) and the caller
// must fall back to the scalar kernel.
namespace deep_gemm {
namespace sm120_mla_v2 {
namespace native {
template <typename QT, typename OutT, typename IdxT>
bool launch_sm120_fused_decode_v2_native(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& indices,
    const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink,
    torch::Tensor&       out,
    torch::Tensor&       lse,
    int                  block_size,
    float                softmax_scale);
}  // namespace native
}  // namespace sm120_mla_v2
}  // namespace deep_gemm

namespace deep_gemm {
namespace sm120_mla_v2 {
namespace {

constexpr int kHeadDim = 512;
constexpr int kFp8Dim = 448;
constexpr int kBf16Dim = 64;
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = 7;     // kFp8Dim / kQuantBlock
constexpr int kTokenDataBytes = kFp8Dim + kBf16Dim * 2;  // 448 + 128 = 576
constexpr int kScaleBytes = 8;          // 7 used + 1 pad
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

// Resolve a token byte pointer for a given linear KV slot index.
__device__ __forceinline__ const uint8_t* token_ptr_from_linear(
    const uint8_t* cache_flat, int64_t block_stride_bytes, int block_size,
    int64_t linear_index) {
    const int64_t block_id = linear_index / block_size;
    const int block_offset = static_cast<int>(linear_index - block_id * block_size);
    const uint8_t* block = cache_flat + block_id * block_stride_bytes;
    return block + static_cast<int64_t>(block_offset) * kTokenDataBytes;
}

// Resolve scale-byte pointer for a given linear KV slot index. The scales come
// after the token data region inside each block.
__device__ __forceinline__ const uint8_t* scale_ptr_from_linear(
    const uint8_t* cache_flat, int64_t block_stride_bytes, int block_size,
    int64_t linear_index) {
    const int64_t block_id = linear_index / block_size;
    const int block_offset = static_cast<int>(linear_index - block_id * block_size);
    const uint8_t* block = cache_flat + block_id * block_stride_bytes;
    return block + static_cast<int64_t>(block_size) * kTokenDataBytes +
           static_cast<int64_t>(block_offset) * kScaleBytes;
}

// One full row read: dequantize all 512 dims for a token. Used in the inner
// loop. Each thread handles a stride-256 slice of the dim axis so the warp
// covers every dim once.
__device__ __forceinline__ float fetch_kv_dim(
    const uint8_t* cache_flat, int64_t block_stride_bytes, int block_size,
    int64_t linear_index, int dim, const uint8_t* scales_cached) {
    const uint8_t* token = token_ptr_from_linear(
        cache_flat, block_stride_bytes, block_size, linear_index);
    if (dim < kFp8Dim) {
        const float s = decode_ue8m0_scale(scales_cached[dim / kQuantBlock]);
        return fp8_e4m3fn_to_float(token[dim]) * s;
    }
    const auto* rope = reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
    return __bfloat162float(rope[dim - kFp8Dim]);
}

// Block-wide reduction over kThreads lanes producing the global max into
// smem[0] and a synced broadcast back. Uses warp shuffles + a 4-warp shared
// reduction.
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
__global__ void __launch_bounds__(kThreads, 2) sm120_fused_decode_v2_scalar_kernel(
    const QT* __restrict__ q,            // [B, H, head_dim]
    const uint8_t* __restrict__ cache,    // packed FP8 ds_mla cache, uint8
    const IdxT* __restrict__ indices,     // [B, 1, K]
    const int* __restrict__ topk_lengths, // [B] or nullptr
    const float* __restrict__ attn_sink,  // [H] or nullptr
    OutT* __restrict__ out,               // [B, H, head_dim]
    float* __restrict__ lse_out,          // [B, H]
    int B, int H, int K,
    int64_t q_stride_b, int64_t q_stride_h,
    int64_t out_stride_b, int64_t out_stride_h,
    int64_t lse_stride_b,
    int64_t cache_block_stride_bytes,
    int block_size,
    int64_t indices_stride_b,
    float softmax_scale) {
    // One CTA per (batch, head). Grid: (B, H).
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;

    const int klen_full = K;
    const int klen = topk_lengths ? min(klen_full, topk_lengths[b]) : klen_full;

    // ---- 1. Stage Q to shared memory in fp32. -----------------------------
    extern __shared__ unsigned char smem_raw[];
    float* q_shared = reinterpret_cast<float*>(smem_raw);
    float* warp_red = q_shared + kHeadDim;            // 8 floats

    const QT* q_row = q + b * q_stride_b + h * q_stride_h;
    for (int d = tid; d < kHeadDim; d += blockDim.x) {
        q_shared[d] = load_q_value<QT>(q_row, d);
    }
    __syncthreads();

    // ---- 2. Online softmax + P*V accumulator ------------------------------
    // Each thread owns a slice of the head_dim axis (size 2 with 256 threads).
    // We accumulate the dim-slice contribution to out[h, dim] across all
    // sparse columns.
    constexpr int kDimsPerThread = kHeadDim / kThreads;          // 2
    float out_acc[kDimsPerThread];
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
        out_acc[i] = 0.0f;
    }
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    const IdxT* idx_row = indices + b * indices_stride_b;

    // ---- 3. Iterate sparse columns in chunks ------------------------------
    // Each chunk is sized so we can hold 32 candidates' logits in shared
    // memory between the score and the apply phases.
    constexpr int kCandidatesPerChunk = 32;
    __shared__ float chunk_logits[kCandidatesPerChunk];
    __shared__ int64_t chunk_linear[kCandidatesPerChunk];
    __shared__ uint8_t chunk_scales[kCandidatesPerChunk][8];

    for (int kc0 = 0; kc0 < klen; kc0 += kCandidatesPerChunk) {
        const int chunk_n = min(kCandidatesPerChunk, klen - kc0);

        // (3a) Resolve indices and pre-load scale bytes for the chunk.
        if (tid < chunk_n) {
            const IdxT raw = idx_row[kc0 + tid];
            int64_t lin = static_cast<int64_t>(raw);
            if (raw < 0) {
                chunk_linear[tid] = -1;
            } else {
                chunk_linear[tid] = lin;
                const uint8_t* sptr = scale_ptr_from_linear(
                    cache, cache_block_stride_bytes, block_size, lin);
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    chunk_scales[tid][s] = sptr[s];
                }
            }
        }
        __syncthreads();

        // (3b) Compute per-candidate score = sum_d Q[d] * K[d].
        // Distribute candidates across threads so each candidate is dotted by
        // a contiguous group of threads, and the partial dots are reduced via
        // warp shuffle. We use 8 threads per candidate (256 / 32).
        // TODO(SM120-MMA): replace the inner loop with a warp-level
        // ``mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32`` block-scaled MMA on
        // the FP8 portion (dim < 448), then add the BF16 RoPE tail with a
        // separate ``mma.sync.kind::f16`` instruction. This is the path that
        // turns this kernel from "structurally fused" to "tensor-core fused".
        constexpr int kThreadsPerCand = kThreads / kCandidatesPerChunk;  // 8
        const int cand = tid / kThreadsPerCand;     // 0..31
        const int sublane = tid % kThreadsPerCand;  // 0..7

        float partial = 0.0f;
        if (cand < chunk_n) {
            const int64_t lin = chunk_linear[cand];
            if (lin >= 0) {
                const uint8_t* token = token_ptr_from_linear(
                    cache, cache_block_stride_bytes, block_size, lin);
                const auto* rope =
                    reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
                // Each sublane handles head_dim / kThreadsPerCand = 64 dims.
                constexpr int kDimsPerSublane = kHeadDim / kThreadsPerCand;
                const int d0 = sublane * kDimsPerSublane;
#pragma unroll
                for (int dd = 0; dd < kDimsPerSublane; ++dd) {
                    const int d = d0 + dd;
                    float kv;
                    if (d < kFp8Dim) {
                        const float s = decode_ue8m0_scale(
                            chunk_scales[cand][d / kQuantBlock]);
                        kv = fp8_e4m3fn_to_float(token[d]) * s;
                    } else {
                        kv = __bfloat162float(rope[d - kFp8Dim]);
                    }
                    partial += q_shared[d] * kv;
                }
            }
        }
        // Reduce within each sublane group (8 lanes) by warp-internal shuffles.
        // The 8 threads of each candidate are guaranteed to be inside the same
        // warp because kThreadsPerCand divides 32.
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

        // Apply optional attention sink at the head level on the very first
        // chunk, treating it as one extra logit folded into row_max/row_sum.
        if (kc0 == 0 && attn_sink != nullptr && tid == 0) {
            // Rather than allocating a synthetic sink "column", fold it into
            // row_max/row_sum AFTER we know the chunk's max. Done below.
        }

        // (3c) Online softmax update: find chunk max, rescale prior state.
        float chunk_max = -INFINITY;
        for (int c = tid; c < chunk_n; c += blockDim.x) {
            chunk_max = fmaxf(chunk_max, chunk_logits[c]);
        }
        chunk_max = block_reduce_max(chunk_max, warp_red);

        const bool first_chunk = (kc0 == 0);
        float new_max;
        if (first_chunk) {
            // On first chunk the running state is the chunk itself, optionally
            // including the sink logit.
            float sink = (attn_sink != nullptr) ? attn_sink[h] : -INFINITY;
            new_max = fmaxf(chunk_max, sink);
        } else {
            new_max = fmaxf(row_max, chunk_max);
        }

        const float scale_old =
            (row_max == -INFINITY) ? 0.0f : __expf(row_max - new_max);
        // Rescale running out_acc + row_sum.
#pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
            out_acc[i] *= scale_old;
        }
        row_sum *= scale_old;

        // Add chunk contribution to row_sum.
        float chunk_sum = 0.0f;
        for (int c = tid; c < chunk_n; c += blockDim.x) {
            float p = __expf(chunk_logits[c] - new_max);
            chunk_logits[c] = p;   // store the post-softmax weight
            chunk_sum += p;
        }
        chunk_sum = block_reduce_sum(chunk_sum, warp_red);
        row_sum += chunk_sum;

        // Sink term contribution on first chunk.
        if (first_chunk && attn_sink != nullptr) {
            float sink_p = __expf(attn_sink[h] - new_max);
            row_sum += sink_p;
            // The sink is a learned logit attending to a zero V, so it adds
            // 0 to out_acc and does not need a P*V step.
        }

        row_max = new_max;
        __syncthreads();

        // (3d) P*V accumulate: out[d] += sum_c chunk_logits[c] * V[c, d]
        // Each thread owns kDimsPerThread dims (2). It walks all valid
        // candidates in the chunk and accumulates.
        // TODO(SM120-MMA): replace this scalar accumulate with a warp-level
        // ``mma.sync.aligned.m16n8k32.f32.e4m3.f32.f32`` MMA over P (fp32) and
        // V (FP8/UE8M0) for dim < 448, plus a ``mma.sync.kind::f16`` for the
        // BF16 RoPE tail.
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
                    const float s = decode_ue8m0_scale(
                        chunk_scales[c][d / kQuantBlock]);
                    kv = fp8_e4m3fn_to_float(token[d]) * s;
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

    // ---- 4. Write output and LSE -----------------------------------------
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    OutT* out_row = out + b * out_stride_b + h * out_stride_h;
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
        lse_out[b * lse_stride_b + h] = lse;
    }
}

template <typename QT, typename OutT, typename IdxT>
void launch_sm120_fused_decode_v2(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& indices,
    const torch::Tensor& topk_length,
    const torch::Tensor& attn_sink,
    torch::Tensor& out,
    torch::Tensor& lse,
    int block_size,
    float softmax_scale) {
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int K = static_cast<int>(indices.size(2));

    const auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(B, H);
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

    // Native FP8 path dispatch (testing-blind, default-off). Only attempts
    // the native kernel when DG_SM120_FUSED_DECODE_V2_NATIVE=1 is set; if the
    // native launch returns false (shape unsupported, e.g. H % 16 != 0) we
    // fall through to the scalar kernel below. Compile errors in the native
    // path will surface at build time regardless of the env flag.
    {
        const char* native_env = std::getenv("DG_SM120_FUSED_DECODE_V2_NATIVE");
        const bool want_native =
            (native_env != nullptr && native_env[0] != '\0' &&
             std::atoi(native_env) != 0);
        if (want_native) {
            const bool ok =
                native::launch_sm120_fused_decode_v2_native<QT, OutT, IdxT>(
                    q, k_cache, indices, topk_length, attn_sink,
                    out, lse, block_size, softmax_scale);
            if (ok) return;
            // else: fall through to scalar.
        }
    }

    sm120_fused_decode_v2_scalar_kernel<QT, OutT, IdxT><<<grid, block, shared_bytes,
                                                   stream>>>(
        reinterpret_cast<const QT*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
        indices.data_ptr<IdxT>(),
        topk_ptr,
        sink_ptr,
        reinterpret_cast<OutT*>(out.data_ptr()),
        lse.data_ptr<float>(),
        B, H, K,
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

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_v2(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& indices,
    const pybind11::object& topk_length_obj,
    const pybind11::object& attn_sink_obj,
    int head_dim_v,
    double softmax_scale,
    int block_size,
    const pybind11::object& out_obj) {
    DG_HOST_ASSERT(q.is_cuda() && k_cache.is_cuda() && indices.is_cuda());
    DG_HOST_ASSERT(q.dim() == 3 && q.size(2) == kHeadDim);
    DG_HOST_ASSERT(head_dim_v == kHeadDim);
    DG_HOST_ASSERT(indices.dim() == 3 && indices.size(0) == q.size(0) &&
                   indices.size(1) == 1);
    DG_HOST_ASSERT(block_size > 0);
    DG_HOST_ASSERT(k_cache.size(k_cache.dim() - 1) >=
                   kTokenDataBytes + kScaleBytes);
    DG_HOST_ASSERT(q.scalar_type() == torch::kBFloat16 ||
                   q.scalar_type() == torch::kFloat16);

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
    auto lse =
        torch::empty({q.size(0), q.size(1)}, q.options().dtype(torch::kFloat32));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

    const float scale = static_cast<float>(softmax_scale);

    if (q.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16) {
        if (indices.scalar_type() == torch::kInt32) {
            launch_sm120_fused_decode_v2<__nv_bfloat16, __nv_bfloat16, int32_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                block_size, scale);
        } else if (indices.scalar_type() == torch::kInt64) {
            launch_sm120_fused_decode_v2<__nv_bfloat16, __nv_bfloat16, int64_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                block_size, scale);
        } else {
            DG_HOST_UNREACHABLE("v2 decode indices must be int32/int64");
        }
    } else if (q.scalar_type() == torch::kFloat16 && out.scalar_type() == torch::kFloat16) {
        if (indices.scalar_type() == torch::kInt32) {
            launch_sm120_fused_decode_v2<half, half, int32_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                block_size, scale);
        } else if (indices.scalar_type() == torch::kInt64) {
            launch_sm120_fused_decode_v2<half, half, int64_t>(
                q, k_cache, indices, topk_length, attn_sink, out, lse,
                block_size, scale);
        } else {
            DG_HOST_UNREACHABLE("v2 decode indices must be int32/int64");
        }
    } else {
        DG_HOST_UNREACHABLE("v2 decode requires matching bf16 or fp16 q/out");
    }

    return std::make_tuple(out, lse);
}

void register_decode_v2_apis(pybind11::module& m) {
    m.def("sm120_sparse_mla_decode_v2", &sparse_mla_decode_v2,
          "SM120 fused sparse MLA decode (v2): single CTA, FP8 cache direct, "
          "online softmax, no BF16 workspace");
}

} // namespace sm120_mla_v2
} // namespace deep_gemm
