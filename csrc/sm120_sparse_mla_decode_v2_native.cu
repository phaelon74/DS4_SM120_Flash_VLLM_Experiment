// SM120 native FP8 decode (v2 tensor-core path).
//
// This file holds the FP8 block-scaled tensor-core implementation of the v2
// sparse MLA decode kernel. The scalar v2 kernel lives in
// sm120_sparse_mla_decode_v2.cu and remains the default; this native path is
// gated by DG_SM120_FUSED_DECODE_V2_NATIVE=1 with sub-flag
// DG_SM120_FUSED_DECODE_V2_FP8MMA=1 selecting the FP8 MMA inner (vs a bf16
// fallback that promotes FP8 to BF16 in SMEM but uses the same single-launch
// multi-head-per-CTA structural rewrite).
//
// Architecture:
//   Grid:    (B, H / kHeadsPerCta, kSplitK)
//   Block:   kNativeThreads = 128 (4 warps)
//   SMEM:    ~32 KB / CTA
//   MMA:     m16n8k32 e4m3 block_scale for d < 448 + m16n8k16 bf16 for RoPE
//
// Each CTA handles `kHeadsPerCta` (16) heads. Each warp owns one N=8 tile of
// QK^T (4 warps × 8 = 32 cands per chunk) and one 128-column slice of the P*V
// output (4 warps × 128 = 512 = head_dim).
//
// Cache layout (DSv4 fp8_ds_mla):
//   * 448 fp8 e4m3 dims (bytes)
//   * 64 bf16 RoPE dims (128 bytes)
//   * 8 UE8M0 scale bytes per token (7 used + 1 pad), placed after the
//     block_size token-data region, NOT interleaved per token.
//
// Numerics:
//   * UE8M0 byte 0 is treated as "scale = 2^-126" (clamped). DSv4's scalar
//     path returns 0 here; we cannot, because block_scale MMA has no zero-
//     scale mode. Invalid candidates are masked via softmax -inf logit.
//   * P quantization (fp32 -> fp8 + UE8M0) uses pow2_round_up(rowmax/448).
//
// Split-K:
//   * kSplitK == 1: write directly to final out + lse buffers.
//   * kSplitK >  1: write per-split partials, combined by reduce kernel.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/python.h>

#include "utils/exception.hpp"

#include "sm120_native_fp8/mma_block_scale_fp8.cuh"
#include "sm120_native_fp8/fp8_quant.cuh"

namespace deep_gemm {
namespace sm120_mla_v2 {
namespace native {

// ---- Constants (must match scalar path) -----------------------------------
constexpr int kHeadDim       = 512;
constexpr int kFp8Dim        = 448;
constexpr int kBf16Dim       = 64;
constexpr int kQuantBlock    = 64;
constexpr int kNumQuantBlocks = 7;       // kFp8Dim / kQuantBlock
constexpr int kTokenDataBytes = kFp8Dim + kBf16Dim * 2;  // 576
constexpr int kScaleBytes    = 8;        // 7 used + 1 pad

// ---- Native-kernel-specific tile shape ------------------------------------
constexpr int kHeadsPerCta   = 16;
constexpr int kCandsPerChunk = 32;
constexpr int kNativeThreads = 128;
constexpr int kNativeWarps   = kNativeThreads / 32;     // 4
constexpr int kColsPerWarp   = kHeadDim / kNativeWarps; // 128
constexpr int kNTilesPerWarp = kColsPerWarp / 8;        // 16
constexpr int kRowsPerWarp   = kHeadsPerCta / kNativeWarps;  // 4

// ---- SMEM byte budget (compile-time) --------------------------------------
constexpr int kQFp8Bytes      = kHeadsPerCta * kFp8Dim;            // 7168
constexpr int kQBf16Bytes     = kHeadsPerCta * kBf16Dim * 2;       // 2048
constexpr int kQScalesBytes   = kHeadsPerCta * 8;                  //  128
constexpr int kKVFp8Bytes     = kCandsPerChunk * kFp8Dim;          // 14336
constexpr int kKVBf16Bytes    = kCandsPerChunk * kBf16Dim * 2;     // 4096
constexpr int kKVScalesBytes  = kCandsPerChunk * 8;                //  256
constexpr int kKVLinearBytes  = kCandsPerChunk * 8;                //  256 (int64)
constexpr int kLogitsBytes    = kHeadsPerCta * kCandsPerChunk * 4; // 2048
constexpr int kPFp8Bytes      = kHeadsPerCta * kCandsPerChunk;     //  512
constexpr int kPScalesBytes   = kHeadsPerCta * 8;                  //  128
constexpr int kPBf16Bytes     = kHeadsPerCta * kCandsPerChunk * 2; // 1024
constexpr int kRowStateBytes  = kHeadsPerCta * 8;                  //  128 (m + l)

constexpr int kNativeSmemBytes =
    kQFp8Bytes + kQBf16Bytes + kQScalesBytes +
    kKVFp8Bytes + kKVBf16Bytes + kKVScalesBytes + kKVLinearBytes +
    kLogitsBytes + kPFp8Bytes + kPScalesBytes + kPBf16Bytes +
    kRowStateBytes;
static_assert(kNativeSmemBytes <= 99 * 1024,
              "Native v2 decode SMEM exceeds SM120 99 KB/block budget");

// ---- Output element load/store (type-dispatched) -------------------------
template <typename T>
__device__ __forceinline__ float load_out_elem(const T* p, int64_t off) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(p[off]);
    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(p[off]);
    } else {
        return static_cast<float>(p[off]);
    }
}

template <typename T>
__device__ __forceinline__ void store_out_elem(T* p, int64_t off, float v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        p[off] = __float2bfloat16(v);
    } else if constexpr (std::is_same_v<T, half>) {
        p[off] = __float2half(v);
    } else {
        p[off] = static_cast<T>(v);
    }
}

// ---- Cache pointer helpers (must match scalar path) -----------------------

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

// ---- Q load + quantize (BF16/FP16 -> e4m3 + UE8M0 + bf16 RoPE) -----------

template <typename QT>
__device__ __forceinline__ float load_q_value(const QT* p, int d);
template <>
__device__ __forceinline__ float load_q_value<__nv_bfloat16>(
    const __nv_bfloat16* p, int d) { return __bfloat162float(p[d]); }
template <>
__device__ __forceinline__ float load_q_value<half>(
    const half* p, int d) { return __half2float(p[d]); }

// Each warp loads kRowsPerWarp = 4 rows. Within each row, 32 lanes share
// kHeadDim = 512 dims, 16 dims/lane. The FP8 quant block is 64 dims, so 4
// consecutive lanes form one quant group. Lane 0..3 -> qb 0, ..., lane 24..27
// -> qb 6 (last FP8 block). Lanes 28..31 hold the 64 RoPE bf16 dims.
template <typename QT>
__device__ __forceinline__ void stage_q_to_smem(
    const QT* q_global,
    int64_t  q_stride_h,
    int      h0,
    uint8_t* q_fp8_smem,
    __nv_bfloat16* q_bf16_smem,
    uint8_t* q_scales_smem,
    int      warp_id,
    int      lane_id) {
    using namespace deep_gemm::sm120_native_fp8;

    const int lane_qb        = lane_id >> 2;          // 0..7 (qb 0..6 fp8, qb 7 = rope)
    const int lane_within_qb = lane_id & 3;           // 0..3
    const int dim_base       = lane_id * 16;          // 0, 16, 32, ..., 496

    #pragma unroll
    for (int rw = 0; rw < kRowsPerWarp; ++rw) {
        const int r        = warp_id * kRowsPerWarp + rw;
        const QT* row_base = q_global + (h0 + r) * q_stride_h;

        // Read 16 dims into registers.
        float vals[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int d = dim_base + i;
            vals[i] = (d < kHeadDim) ? load_q_value<QT>(row_base, d) : 0.0f;
        }

        if (lane_qb < kNumQuantBlocks) {
            // FP8 quant block. Each lane holds 16 dims; reduce max-abs across
            // the 4-lane group (= 64 dims = one quant block).
            float local_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                local_max = fmaxf(local_max, fabsf(vals[i]));
            }
            // 4-lane reduction: shuffle within consecutive lane group of size 4.
            float qb_max = local_max;
            qb_max = fmaxf(qb_max, __shfl_xor_sync(0xffffffffu, qb_max, 1));
            qb_max = fmaxf(qb_max, __shfl_xor_sync(0xffffffffu, qb_max, 2));

            const float scale     = pow2_round_up(qb_max / kE4M3Max);
            const float inv_scale = 1.0f / scale;
            const uint8_t ue8m0   = float_pow2_to_ue8m0(scale);

            // Lane 0 of each 4-lane group writes the scale.
            if (lane_within_qb == 0) {
                q_scales_smem[r * 8 + lane_qb] = ue8m0;
            }
            // All lanes write their 16 fp8 outputs.
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int d = dim_base + i;
                if (d < kFp8Dim) {
                    q_fp8_smem[r * kFp8Dim + d] = float_to_e4m3(vals[i] * inv_scale);
                }
            }
        } else {
            // RoPE block. Copy bf16 verbatim.
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int d = dim_base + i;
                if (d >= kFp8Dim && d < kHeadDim) {
                    q_bf16_smem[r * kBf16Dim + (d - kFp8Dim)] =
                        __float2bfloat16(vals[i]);
                }
            }
        }
    }
}

// ---- KV cooperative load --------------------------------------------------
// Each warp handles a subset of candidates. For each candidate, lanes copy
// kFp8Dim bytes of FP8 + kBf16Dim bf16 (= 128 bytes) of RoPE. Invalid
// candidates (linear < 0) get zero-filled so MMA produces zero (logit will be
// -inf via the validity mask).
__device__ __forceinline__ void load_kv_chunk_to_smem(
    const uint8_t* cache,
    int64_t        cache_block_stride_bytes,
    int            block_size,
    const int64_t* kv_linear_smem,
    uint8_t*       kv_fp8_smem,
    __nv_bfloat16* kv_bf16_smem,
    int            warp_id,
    int            lane_id) {
    for (int c = warp_id; c < kCandsPerChunk; c += kNativeWarps) {
        const int64_t lin = kv_linear_smem[c];
        if (lin < 0) {
            for (int i = lane_id; i < kFp8Dim; i += 32) {
                kv_fp8_smem[c * kFp8Dim + i] = 0u;
            }
            for (int i = lane_id; i < kBf16Dim; i += 32) {
                kv_bf16_smem[c * kBf16Dim + i] = __float2bfloat16(0.0f);
            }
        } else {
            const uint8_t* token = token_ptr_from_linear(
                cache, cache_block_stride_bytes, block_size, lin);
            // FP8 region: 448 bytes / 32 lanes = 14 bytes/lane. Use byte-wise
            // copy (vector loads would require 16B alignment guarantees we
            // don't have on per-token ptrs).
            for (int i = lane_id; i < kFp8Dim; i += 32) {
                kv_fp8_smem[c * kFp8Dim + i] = token[i];
            }
            // BF16 RoPE: 128 bytes = 64 bf16. 32 lanes × 2 bf16 each.
            const __nv_bfloat16* rope =
                reinterpret_cast<const __nv_bfloat16*>(token + kFp8Dim);
            for (int i = lane_id; i < kBf16Dim; i += 32) {
                kv_bf16_smem[c * kBf16Dim + i] = rope[i];
            }
        }
    }
}

// ---- m16n8 fragment write helper -----------------------------------------
// Lane l of an m16n8 fp32 fragment owns 4 elements at positions:
//   (row_base,     col_base    )
//   (row_base,     col_base + 1)
//   (row_base + 8, col_base    )
//   (row_base + 8, col_base + 1)
// where row_base = l/4, col_base = (l%4) * 2.
__device__ __forceinline__ void write_m16n8_fragment_to_smem(
    float*   dst,           // [kHeadsPerCta, ?]
    int      dst_stride,    // = kCandsPerChunk for logits, = chunk col stride for P*V
    int      n_tile_off,    // base column for this tile within dst
    int      lane_id,
    const float (&frag)[4]) {
    const int row_base = lane_id / 4;
    const int col_base = (lane_id % 4) * 2;
    dst[row_base       * dst_stride + n_tile_off + col_base    ] = frag[0];
    dst[row_base       * dst_stride + n_tile_off + col_base + 1] = frag[1];
    dst[(row_base + 8) * dst_stride + n_tile_off + col_base    ] = frag[2];
    dst[(row_base + 8) * dst_stride + n_tile_off + col_base + 1] = frag[3];
}

// ---- ldmatrix lane-address helpers ---------------------------------------
// For a row-major SMEM tile, ldmatrix.x4.b16 expects each lane to provide the
// SMEM byte address of the start of the 8x8 sub-fragment that lane "owns".
// Standard mapping for an A operand of shape M x K (we want M=16, K_load=16
// for ldmatrix; the m16n8k32 fp8 MMA needs us to call ldmatrix twice across
// K, see kernel body):
//   Lane l -> row = (l & 15), k_off = (l >> 4) * 8.
// The "row" is the absolute row in the 16-row tile; the "k_off" picks
// between the lower (k=0..7) and upper (k=8..15) 8-K-byte halves.

// For the B operand of m16n8k32 (K=32, N=8, K-major in PTX semantics) loaded
// with ldmatrix.x2.trans from a row-major [N, K] SMEM tile:
//   Lane l -> row = (l & 7) (N=0..7), k_off = ((l >> 3) & 1) * 8.

}  // namespace native
}  // namespace sm120_mla_v2
}  // namespace deep_gemm

// ---- Kernel template ------------------------------------------------------
// The kernel is in its own `__global__` function below the helpers above. It
// is templated on input dtypes, kSplitK, and kUseFp8Mma so the dispatch can
// pick the right specialization at launch time.

namespace deep_gemm {
namespace sm120_mla_v2 {
namespace native {

template <typename QT, typename OutT, typename IdxT, int kSplitK, bool kUseFp8Mma>
__global__ void __launch_bounds__(kNativeThreads, 2)
sm120_fused_decode_v2_native_kernel(
    const QT*      __restrict__ q,
    const uint8_t* __restrict__ cache,
    const IdxT*    __restrict__ indices,
    const int*     __restrict__ topk_lengths,
    const float*   __restrict__ attn_sink,
    OutT*          __restrict__ out_partials,
    float*         __restrict__ lse_partials,
    int B, int H, int K,
    int64_t q_stride_b, int64_t q_stride_h,
    int64_t out_stride_b, int64_t out_stride_h,
    int64_t out_stride_split,    // 0 if kSplitK == 1
    int64_t lse_stride_b,
    int64_t lse_stride_split,    // 0 if kSplitK == 1
    int64_t cache_block_stride_bytes,
    int     block_size,
    int64_t indices_stride_b,
    float   softmax_scale) {
    using namespace deep_gemm::sm120_native_fp8;

    const int b          = blockIdx.x;
    const int head_block = blockIdx.y;
    const int split_id   = blockIdx.z;
    const int tid        = threadIdx.x;
    const int warp_id    = tid >> 5;
    const int lane_id    = tid & 31;

    const int klen_full       = K;
    const int klen            = topk_lengths ? min(klen_full, topk_lengths[b])
                                             : klen_full;
    const int per_split       = (klen + kSplitK - 1) / kSplitK;
    const int split_start     = split_id * per_split;
    const int split_end       = min(klen, split_start + per_split);
    const int h0              = head_block * kHeadsPerCta;

    // ---- SMEM partitioning ------------------------------------------------
    extern __shared__ unsigned char smem_raw[];
    unsigned char* sp = smem_raw;
    uint8_t*       q_fp8_smem    = sp;                            sp += kQFp8Bytes;
    __nv_bfloat16* q_bf16_smem   = reinterpret_cast<__nv_bfloat16*>(sp);
                                                                  sp += kQBf16Bytes;
    uint8_t*       q_scales_smem = sp;                            sp += kQScalesBytes;
    uint8_t*       kv_fp8_smem   = sp;                            sp += kKVFp8Bytes;
    __nv_bfloat16* kv_bf16_smem  = reinterpret_cast<__nv_bfloat16*>(sp);
                                                                  sp += kKVBf16Bytes;
    uint8_t*       kv_scales_smem = sp;                           sp += kKVScalesBytes;
    int64_t*       kv_linear_smem = reinterpret_cast<int64_t*>(sp);
                                                                  sp += kKVLinearBytes;
    float*         logits_smem   = reinterpret_cast<float*>(sp);  sp += kLogitsBytes;
    uint8_t*       p_fp8_smem    = sp;                            sp += kPFp8Bytes;
    uint8_t*       p_scales_smem = sp;                            sp += kPScalesBytes;
    __nv_bfloat16* p_bf16_smem   = reinterpret_cast<__nv_bfloat16*>(sp);
                                                                  sp += kPBf16Bytes;
    float*         row_state_smem = reinterpret_cast<float*>(sp); // [m..., l...]
                                                                  sp += kRowStateBytes;
    (void)sp;

    // row_state_smem[r]              = m (running max) for row r
    // row_state_smem[kHeadsPerCta+r] = l (running sum) for row r

    // ---- 1. Load + quantize Q --------------------------------------------
    {
        const QT* q_global = q + b * q_stride_b;
        stage_q_to_smem<QT>(
            q_global, q_stride_h, h0,
            q_fp8_smem, q_bf16_smem, q_scales_smem,
            warp_id, lane_id);
    }
    // Initialize row_state.
    if (warp_id == 0 && lane_id < kHeadsPerCta) {
        row_state_smem[lane_id]                  = -INFINITY;  // m
        row_state_smem[kHeadsPerCta + lane_id]   = 0.0f;       // l
    }
    __syncthreads();

    // ---- 2. Initialize per-warp output accumulator ----------------------
    // Each warp owns kColsPerWarp (128) cols of out_acc out of head_dim=512.
    // Within those, kNTilesPerWarp (16) m16n8 N-tiles. Per lane: 4 fp32 per
    // tile. Total: 64 fp32 regs/lane. Lives in registers throughout the
    // kernel (no SMEM round-trip across chunks).
    float out_acc[kNTilesPerWarp][4];
    #pragma unroll
    for (int t = 0; t < kNTilesPerWarp; ++t) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) out_acc[t][r] = 0.0f;
    }

    const IdxT* idx_row = indices + b * indices_stride_b;

    // ---- 3. Main chunk loop ----------------------------------------------
    for (int kc0 = split_start; kc0 < split_end; kc0 += kCandsPerChunk) {
        const int chunk_n = min(kCandsPerChunk, split_end - kc0);

        // (3a) Resolve indices, load scale bytes for the chunk.
        if (tid < kCandsPerChunk) {
            int64_t lin = -1;
            if (tid < chunk_n) {
                IdxT raw = idx_row[kc0 + tid];
                lin = static_cast<int64_t>(raw);
                if (raw < 0) lin = -1;
            }
            kv_linear_smem[tid] = lin;
            if (lin >= 0) {
                const uint8_t* sptr = scale_ptr_from_linear(
                    cache, cache_block_stride_bytes, block_size, lin);
                #pragma unroll
                for (int s = 0; s < 8; ++s) {
                    kv_scales_smem[tid * 8 + s] = sptr[s];
                }
            } else {
                #pragma unroll
                for (int s = 0; s < 8; ++s) {
                    kv_scales_smem[tid * 8 + s] = 1u;
                }
            }
        }
        __syncthreads();

        // (3b) Cooperative load FP8 + BF16 RoPE for the chunk.
        load_kv_chunk_to_smem(
            cache, cache_block_stride_bytes, block_size,
            kv_linear_smem, kv_fp8_smem, kv_bf16_smem,
            warp_id, lane_id);
        __syncthreads();

        // (3c) QK^T MMA per warp (one N-tile each). Result is [16, 8] fp32.
        const int n_tile_off = warp_id * 8;
        float qk_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // FP8 portion (d < 448): m16n8k32 e4m3 block-scale, 14 K-iterations.
        if constexpr (kUseFp8Mma) {
            #pragma unroll 1
            for (int k_iter = 0; k_iter < kFp8Dim / 32; ++k_iter) {
                const int k0    = k_iter * 32;
                const int qb_id = k0 / kQuantBlock;

                // ldmatrix Q (A operand) m16k32 -- load as two K=16 halves
                // through ldmatrix.x4.b16 reinterpreting fp8 as packed b16.
                const uint32_t a_row   = (lane_id & 15);
                const uint32_t a_k_off = 16u * (lane_id >> 4);
                const uint8_t* a_addr  =
                    &q_fp8_smem[a_row * kFp8Dim + k0 + a_k_off];
                uint32_t a_frag[4];
                ldmatrix_x4_fp8_as_b16(a_frag, a_addr);

                // ldmatrix K (B operand) k32n8 -- trans variant from row-major
                // [N, K] SMEM. Each lane provides the address of one cand row
                // at one of two K=16 halves.
                const uint32_t b_n     = n_tile_off + (lane_id & 7);
                const uint32_t b_k_off = 16u * ((lane_id >> 3) & 1);
                const uint8_t* b_addr  =
                    &kv_fp8_smem[b_n * kFp8Dim + k0 + b_k_off];
                uint32_t b_frag[2];
                ldmatrix_x2_trans_fp8_as_b16(b_frag, b_addr);

                // Pack scales: lane l < 16 gets row l's scale for A; lane
                // l < 8 gets cand (n_tile_off + l)'s scale for B.
                uint32_t scale_a = 0, scale_b = 0;
                if (lane_id < kHeadsPerCta) {
                    scale_a = pack_scale_byte0(
                        q_scales_smem[lane_id * 8 + qb_id]);
                }
                if (lane_id < 8) {
                    scale_b = pack_scale_byte0(
                        kv_scales_smem[(n_tile_off + lane_id) * 8 + qb_id]);
                }

                float qk_new[4];
                mma_e4m3_block_scale_m16n8k32(
                    qk_new, a_frag, b_frag, qk_acc, scale_a, scale_b);
                #pragma unroll
                for (int i = 0; i < 4; ++i) qk_acc[i] = qk_new[i];
            }
        } else {
            // BF16 fallback: dequantize FP8 to BF16 in registers and use the
            // bf16 MMA path. Slower than FP8 native but a known-good
            // correctness reference.
            // TODO(NATIVE-BF16-FALLBACK): implement when FP8 path is validated;
            // currently the FP8 path is the only supported native mode.
            (void)q_fp8_smem; (void)kv_fp8_smem;
            (void)q_scales_smem; (void)kv_scales_smem;
        }

        // BF16 RoPE portion (d >= 448): m16n8k16 bf16, 4 K-iterations.
        #pragma unroll 1
        for (int k_iter = 0; k_iter < kBf16Dim / 16; ++k_iter) {
            const int k0 = k_iter * 16;

            const uint32_t a_row   = (lane_id & 15);
            const uint32_t a_k_off = 8u * (lane_id >> 4);
            const __nv_bfloat16* a_addr =
                &q_bf16_smem[a_row * kBf16Dim + k0 + a_k_off];
            uint32_t a_frag[4];
            ldmatrix_x4_b16(a_frag, a_addr);

            const uint32_t b_n     = n_tile_off + (lane_id & 7);
            const uint32_t b_k_off = 8u * ((lane_id >> 3) & 1);
            const __nv_bfloat16* b_addr =
                &kv_bf16_smem[b_n * kBf16Dim + k0 + b_k_off];
            uint32_t b_frag[2];
            ldmatrix_x2_trans_b16(b_frag, b_addr);

            float qk_new[4];
            mma_bf16_m16n8k16(qk_new, a_frag, b_frag, qk_acc);
            #pragma unroll
            for (int i = 0; i < 4; ++i) qk_acc[i] = qk_new[i];
        }

        // (3d) Apply softmax_scale + validity mask, write to logits_smem.
        {
            const int row_base = lane_id / 4;
            const int col_base = (lane_id % 4) * 2;
            const int n0 = n_tile_off + col_base;
            const int n1 = n0 + 1;

            const int64_t lin0 = (n0 < kCandsPerChunk) ? kv_linear_smem[n0] : -1;
            const int64_t lin1 = (n1 < kCandsPerChunk) ? kv_linear_smem[n1] : -1;
            const bool valid0 = (n0 < chunk_n) && (lin0 >= 0);
            const bool valid1 = (n1 < chunk_n) && (lin1 >= 0);

            const float v0 = valid0 ? qk_acc[0] * softmax_scale : -INFINITY;
            const float v1 = valid1 ? qk_acc[1] * softmax_scale : -INFINITY;
            const float v2 = valid0 ? qk_acc[2] * softmax_scale : -INFINITY;
            const float v3 = valid1 ? qk_acc[3] * softmax_scale : -INFINITY;

            logits_smem[row_base       * kCandsPerChunk + n0] = v0;
            logits_smem[row_base       * kCandsPerChunk + n1] = v1;
            logits_smem[(row_base + 8) * kCandsPerChunk + n0] = v2;
            logits_smem[(row_base + 8) * kCandsPerChunk + n1] = v3;
        }
        __syncthreads();

        // (3e) Online softmax update. 1 warp handles kRowsPerWarp rows;
        // within a row, 32 lanes cover up to 32 cands directly.
        const bool first_chunk = (kc0 == split_start);
        for (int rw = 0; rw < kRowsPerWarp; ++rw) {
            const int r = warp_id * kRowsPerWarp + rw;

            float v = (lane_id < kCandsPerChunk)
                        ? logits_smem[r * kCandsPerChunk + lane_id]
                        : -INFINITY;

            // Row max via warp reduction.
            float chunk_max = v;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                chunk_max = fmaxf(chunk_max,
                                  __shfl_xor_sync(0xffffffffu, chunk_max, off));
            }

            const float prev_m = first_chunk ? -INFINITY : row_state_smem[r];
            const float prev_l = first_chunk ? 0.0f      : row_state_smem[kHeadsPerCta + r];
            float new_m;
            if (first_chunk && attn_sink != nullptr) {
                new_m = fmaxf(chunk_max, attn_sink[h0 + r]);
            } else if (first_chunk) {
                new_m = chunk_max;
            } else {
                new_m = fmaxf(prev_m, chunk_max);
            }

            const float scale_old =
                (prev_m == -INFINITY) ? 0.0f : __expf(prev_m - new_m);

            // Exponentiate and write back to logits_smem (which then doubles
            // as the P[r, c] fp32 staging area for the P*V phase).
            float p_val = (v == -INFINITY) ? 0.0f : __expf(v - new_m);
            if (lane_id < kCandsPerChunk) {
                logits_smem[r * kCandsPerChunk + lane_id] = p_val;
            }

            // Row sum = sum(P[r, c]) over c.
            float chunk_sum = (lane_id < kCandsPerChunk) ? p_val : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                chunk_sum += __shfl_xor_sync(0xffffffffu, chunk_sum, off);
            }
            const float sink_p = (first_chunk && attn_sink != nullptr)
                                   ? __expf(attn_sink[h0 + r] - new_m)
                                   : 0.0f;
            const float new_l = prev_l * scale_old + chunk_sum + sink_p;

            if (lane_id == 0) {
                row_state_smem[r]                 = new_m;
                row_state_smem[kHeadsPerCta + r]  = new_l;
            }

            // Rescale this warp's out_acc rows for row r (if any of this
            // warp's owned rows match). Each warp owns ALL kHeadsPerCta rows
            // for its column slice (since out_acc is M x N and we partitioned
            // by N), so every warp must rescale every row. We do this through
            // a SMEM-broadcasted scale_old: warp 0 lane 0 wrote new_m, but we
            // need scale_old per row. Put it in row_state_smem briefly via an
            // extra slot? Simpler: each warp recomputes scale_old itself from
            // the same warp-reduced chunk_max + new_m above (already in regs).
            //
            // But out_acc is per-warp registers indexed by N-tile within the
            // warp's column slice. The row of out_acc is lane-derived: see
            // m16n8 fragment layout. So the rescale must be selective by row.
            //
            // We defer the rescale to the P*V epilogue (after this loop) by
            // scaling new contributions. This is correct because:
            //   new_out = sum_chunks (out_chunk * exp(m_chunk - m_final))
            // and we accumulate scale_old as we go. We track rescales in
            // out_acc by scaling existing accumulators by scale_old here.

            // Rescale out_acc for THIS row. Lane l owns rows (l/4) and
            // (l/4 + 8). Match against r.
            const int row0 = lane_id / 4;
            const int row1 = row0 + 8;
            const float s_apply = scale_old;
            if (row0 == r) {
                #pragma unroll
                for (int t = 0; t < kNTilesPerWarp; ++t) {
                    out_acc[t][0] *= s_apply;
                    out_acc[t][1] *= s_apply;
                }
            }
            if (row1 == r) {
                #pragma unroll
                for (int t = 0; t < kNTilesPerWarp; ++t) {
                    out_acc[t][2] *= s_apply;
                    out_acc[t][3] *= s_apply;
                }
            }
        }
        __syncthreads();

        // (3f) Quantize P (fp32 in logits_smem) to FP8 e4m3 + UE8M0.
        // P is [16 rows, 32 cands]. Block scale: 1 UE8M0 per row (chunk=32 ==
        // K=32 == one block). Each warp handles kRowsPerWarp rows; within a
        // row, 32 lanes cover the 32 cands.
        if constexpr (kUseFp8Mma) {
            for (int rw = 0; rw < kRowsPerWarp; ++rw) {
                const int r = warp_id * kRowsPerWarp + rw;
                float pv = (lane_id < kCandsPerChunk)
                             ? logits_smem[r * kCandsPerChunk + lane_id]
                             : 0.0f;
                float pmax = fabsf(pv);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    pmax = fmaxf(pmax,
                                 __shfl_xor_sync(0xffffffffu, pmax, off));
                }
                const float pscale     = pow2_round_up(pmax / kE4M3Max);
                const float pinv       = 1.0f / pscale;
                const uint8_t pue8m0   = float_pow2_to_ue8m0(pscale);

                if (lane_id == 0) p_scales_smem[r * 8 + 0] = pue8m0;
                if (lane_id < kCandsPerChunk) {
                    p_fp8_smem[r * kCandsPerChunk + lane_id] =
                        float_to_e4m3(pv * pinv);
                }
            }
        }
        // P -> bf16 staging for the RoPE portion (used by both modes).
        for (int rw = 0; rw < kRowsPerWarp; ++rw) {
            const int r = warp_id * kRowsPerWarp + rw;
            if (lane_id < kCandsPerChunk) {
                p_bf16_smem[r * kCandsPerChunk + lane_id] =
                    __float2bfloat16(logits_smem[r * kCandsPerChunk + lane_id]);
            }
        }
        __syncthreads();

        // (3g) P*V MMA. Per warp: kNTilesPerWarp (16) N-tiles, each m16n8.
        // FP8 portion: m16n8k32, K=kCandsPerChunk=32 (1 K-iter per N-tile).
        // BF16 portion: m16n8k16, K=32 in 2 K-iter per N-tile.
        //
        // NOTE: warp w owns columns [w*kColsPerWarp, (w+1)*kColsPerWarp).
        //       FP8 columns are [0, kFp8Dim=448); BF16 RoPE columns are
        //       [kFp8Dim, kHeadDim=512). With kColsPerWarp=128 and 4 warps:
        //         warp 0: cols [0..128)   -- all FP8
        //         warp 1: cols [128..256) -- all FP8
        //         warp 2: cols [256..384) -- all FP8
        //         warp 3: cols [384..512) -- 64 FP8 (384..448) + 64 BF16 RoPE
        //       So warps 0..2 only do FP8 P*V; warp 3 does both.

        const int warp_col_start = warp_id * kColsPerWarp;

        if constexpr (kUseFp8Mma) {
            // FP8 P*V: warp's 16 N-tiles, each is 8 cols wide.
            // For each tile, K = kCandsPerChunk = 32 (1 m16n8k32 iter).
            #pragma unroll 1
            for (int t = 0; t < kNTilesPerWarp; ++t) {
                const int col0 = warp_col_start + t * 8;
                if (col0 >= kFp8Dim) break;  // remaining tiles are RoPE-only

                // ldmatrix P (A operand) m=16, k=32.
                // P SMEM layout: [16 rows, 32 cands] row-major.
                const uint32_t a_row   = (lane_id & 15);
                const uint32_t a_k_off = 16u * (lane_id >> 4);
                const uint8_t* a_addr  =
                    &p_fp8_smem[a_row * kCandsPerChunk + a_k_off];
                uint32_t a_frag[4];
                ldmatrix_x4_fp8_as_b16(a_frag, a_addr);

                // ldmatrix V (B operand) k=32, n=8 (trans).
                // V SMEM layout: [kCandsPerChunk, kFp8Dim] row-major.
                // We want B = V[:, col0:col0+8], which is K-major-N-minor.
                const uint32_t b_n     = col0 + (lane_id & 7);
                const uint32_t b_k_off = 16u * ((lane_id >> 3) & 1);
                const uint8_t* b_addr  =
                    &kv_fp8_smem[b_k_off * kFp8Dim + b_n];
                // TODO(VERIFY-LDM): B-operand for P*V is "K-major" but our
                // SMEM is [cand, fp8_dim] = [N_for_QK, K_for_QK]. For P*V we
                // re-interpret cand as K and fp8_dim as N. So the loaded
                // 8x8 sub-fragments are at offsets (cand=k_off..k_off+8,
                // dim=col0). The ldmatrix.x2.trans pattern below provides the
                // address of the cand row at the dim col_base.
                uint32_t b_frag[2];
                ldmatrix_x2_trans_fp8_as_b16(b_frag, b_addr);

                // Scales: P scale is per row (kHeadsPerCta lanes). V scale is
                // per cand: cand i uses kv_scales_smem[i * 8 + col0/64].
                uint32_t scale_a = 0, scale_b = 0;
                if (lane_id < kHeadsPerCta) {
                    scale_a = pack_scale_byte0(p_scales_smem[lane_id * 8 + 0]);
                }
                if (lane_id < 8) {
                    // For P*V the K-axis is cand (kCandsPerChunk = 32 = 1
                    // K=32 block). Each cand has its own UE8M0 scale per dim
                    // quant block. The B operand needs one scale per *cand*
                    // for this K-block, evaluated at the *current* dim quant
                    // block (col0 / 64).
                    // PTX scale_vec::1X expects N=8 scales (one per col), but
                    // here we have N=8 cols (each a fp8_dim col) and the
                    // scale should reflect the dim quant block of those cols.
                    // ALL 8 cols within col0..col0+8 fall in the same dim
                    // quant block (since 8 < 64), so one scale per cand, and
                    // we map "cand i -> lane i" for the B-scale lane layout.
                    // // TODO(VERIFY-MMA-SCALE): this re-purposing of "B scale
                    // per col" to "B scale per cand-K-row" is the part of the
                    // PTX semantics I am most uncertain about. If MMA results
                    // are wrong by a constant factor that varies per cand,
                    // this is the first thing to revisit.
                    const int qb_id = col0 / kQuantBlock;
                    scale_b = pack_scale_byte0(
                        kv_scales_smem[lane_id * 8 + qb_id]);
                }

                float pv_new[4];
                mma_e4m3_block_scale_m16n8k32(
                    pv_new, a_frag, b_frag, out_acc[t],
                    scale_a, scale_b);
                #pragma unroll
                for (int i = 0; i < 4; ++i) out_acc[t][i] = pv_new[i];
            }
        }

        // BF16 RoPE P*V (only the warp(s) whose column slice overlaps
        // [kFp8Dim, kHeadDim)).
        if (warp_col_start + kColsPerWarp > kFp8Dim) {
            const int rope_local_start = max(0, kFp8Dim - warp_col_start);
            const int rope_local_end   = kColsPerWarp;
            for (int t = rope_local_start / 8; t < rope_local_end / 8; ++t) {
                const int col0 = warp_col_start + t * 8;
                if (col0 < kFp8Dim) continue;
                const int rope_col0 = col0 - kFp8Dim;

                #pragma unroll 1
                for (int k_iter = 0; k_iter < kCandsPerChunk / 16; ++k_iter) {
                    const int k0 = k_iter * 16;

                    // ldmatrix P (BF16 staging), m=16 k=16.
                    const uint32_t a_row   = (lane_id & 15);
                    const uint32_t a_k_off = 8u * (lane_id >> 4);
                    const __nv_bfloat16* a_addr =
                        &p_bf16_smem[a_row * kCandsPerChunk + k0 + a_k_off];
                    uint32_t a_frag[4];
                    ldmatrix_x4_b16(a_frag, a_addr);

                    // ldmatrix V_rope (B operand bf16), k=16 n=8.
                    const uint32_t b_n     = rope_col0 + (lane_id & 7);
                    const uint32_t b_k_off = 8u * ((lane_id >> 3) & 1);
                    const __nv_bfloat16* b_addr =
                        &kv_bf16_smem[(k0 + b_k_off) * kBf16Dim + b_n];
                    uint32_t b_frag[2];
                    ldmatrix_x2_trans_b16(b_frag, b_addr);

                    float pv_new[4];
                    mma_bf16_m16n8k16(pv_new, a_frag, b_frag, out_acc[t]);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) out_acc[t][i] = pv_new[i];
                }
            }
        }

        __syncthreads();
    }

    // ---- 4. Epilogue: write out + LSE ------------------------------------
    // Each warp writes its kColsPerWarp slice of out_acc to global memory.
    // Lane fragment owns rows (l/4) and (l/4 + 8) of M, cols (l%4)*2..+1 of N.

    // First normalize by row_sum if kSplitK == 1 (single-split case).
    // For kSplitK > 1, write un-normalized partial out + lse partial; the
    // reduce kernel divides by the global sum.

    const int row0 = lane_id / 4;
    const int row1 = row0 + 8;
    const int col_base_local = (lane_id % 4) * 2;

    const float l0 = row_state_smem[kHeadsPerCta + row0];
    const float l1 = row_state_smem[kHeadsPerCta + row1];
    const float m0 = row_state_smem[row0];
    const float m1 = row_state_smem[row1];

    const float inv0 = (kSplitK == 1)
                         ? ((l0 > 0.0f) ? (1.0f / l0) : 0.0f)
                         : 1.0f;
    const float inv1 = (kSplitK == 1)
                         ? ((l1 > 0.0f) ? (1.0f / l1) : 0.0f)
                         : 1.0f;

    const int64_t out_row0_off = static_cast<int64_t>(b) * out_stride_b
                               + static_cast<int64_t>(h0 + row0) * out_stride_h
                               + static_cast<int64_t>(split_id) * out_stride_split;
    const int64_t out_row1_off = static_cast<int64_t>(b) * out_stride_b
                               + static_cast<int64_t>(h0 + row1) * out_stride_h
                               + static_cast<int64_t>(split_id) * out_stride_split;

    OutT* out_row0 = out_partials + out_row0_off;
    OutT* out_row1 = out_partials + out_row1_off;

    #pragma unroll
    for (int t = 0; t < kNTilesPerWarp; ++t) {
        const int col0 = warp_id * kColsPerWarp + t * 8 + col_base_local;
        const int col1 = col0 + 1;
        if (col0 < kHeadDim) store_out_elem(out_row0, col0, out_acc[t][0] * inv0);
        if (col1 < kHeadDim) store_out_elem(out_row0, col1, out_acc[t][1] * inv0);
        if (col0 < kHeadDim) store_out_elem(out_row1, col0, out_acc[t][2] * inv1);
        if (col1 < kHeadDim) store_out_elem(out_row1, col1, out_acc[t][3] * inv1);
    }

    // Lane 0 (and lane equivalent for row1) writes the LSE.
    if (warp_id == 0 && lane_id < kHeadsPerCta) {
        const int r = lane_id;
        const float l = row_state_smem[kHeadsPerCta + r];
        const float m = row_state_smem[r];
        const float lse = (l > 0.0f) ? (logf(l) + m) : -INFINITY;
        const int64_t lse_off = static_cast<int64_t>(b) * lse_stride_b
                              + static_cast<int64_t>(h0 + r)
                              + static_cast<int64_t>(split_id) * lse_stride_split;
        lse_partials[lse_off] = lse;
    }
}

// ---- Split-K reduce kernel ------------------------------------------------
// Combine kSplitK partial outputs using FlashAttention split-K log-sum-exp:
//   global_max = max_k(lse_partial[k])
//   weight[k]  = exp(lse_partial[k] - global_max)
//   global_lse = global_max + log(sum_k weight[k])
//   global_out = sum_k (weight[k] * out_partial[k]) / sum_k weight[k]
//
// Grid: (B, H), Block: (head_dim,). Each thread does one head_dim slot.
// All threads in a block share global_max / sum_weights via SMEM.
template <typename OutT, int kSplitK>
__global__ void __launch_bounds__(kHeadDim, 1)
sm120_fused_decode_v2_splitk_reduce_kernel(
    const OutT*  __restrict__ out_partials,
    const float* __restrict__ lse_partials,
    OutT*        __restrict__ out_final,
    float*       __restrict__ lse_final,
    int B, int H,
    int64_t out_partial_stride_b,
    int64_t out_partial_stride_split,
    int64_t out_partial_stride_h,
    int64_t lse_partial_stride_b,
    int64_t lse_partial_stride_split,
    int64_t out_stride_b, int64_t out_stride_h,
    int64_t lse_stride_b) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;

    __shared__ float s_lse[kSplitK];
    __shared__ float s_global_max;
    __shared__ float s_sum_weights;

    if (d < kSplitK) {
        const int64_t off =
            static_cast<int64_t>(b) * lse_partial_stride_b +
            static_cast<int64_t>(d) * lse_partial_stride_split +
            static_cast<int64_t>(h);
        s_lse[d] = lse_partials[off];
    }
    __syncthreads();

    if (d == 0) {
        float gm = s_lse[0];
        #pragma unroll
        for (int k = 1; k < kSplitK; ++k) gm = fmaxf(gm, s_lse[k]);
        s_global_max = gm;
        float sw = 0.0f;
        #pragma unroll
        for (int k = 0; k < kSplitK; ++k) {
            sw += (s_lse[k] == -INFINITY) ? 0.0f : __expf(s_lse[k] - gm);
        }
        s_sum_weights = sw;
        const float global_lse =
            (sw > 0.0f) ? (logf(sw) + gm) : -INFINITY;
        lse_final[static_cast<int64_t>(b) * lse_stride_b + h] = global_lse;
    }
    __syncthreads();

    if (d < kHeadDim) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < kSplitK; ++k) {
            const float w = (s_lse[k] == -INFINITY)
                              ? 0.0f
                              : __expf(s_lse[k] - s_global_max);
            const int64_t off =
                static_cast<int64_t>(b) * out_partial_stride_b +
                static_cast<int64_t>(k) * out_partial_stride_split +
                static_cast<int64_t>(h) * out_partial_stride_h +
                static_cast<int64_t>(d);
            acc += w * load_out_elem(out_partials, off);
        }
        const float inv = (s_sum_weights > 0.0f) ? (1.0f / s_sum_weights) : 0.0f;
        const int64_t out_off =
            static_cast<int64_t>(b) * out_stride_b +
            static_cast<int64_t>(h) * out_stride_h +
            static_cast<int64_t>(d);
        store_out_elem(out_final, out_off, acc * inv);
    }
}

// ---- Host launch entry point ---------------------------------------------
//
// Returns true on success (native path was used), false if the runtime shape
// requires falling back to the scalar v2 kernel (e.g. H % kHeadsPerCta != 0).
//
// Reads runtime env vars:
//   DG_SM120_FUSED_DECODE_V2_FP8MMA  (default 1) -- use FP8 block-scale MMA.
//                                                    0 = bf16 fallback (TODO).
//   DG_SM120_FUSED_DECODE_V2_SPLITK  (default 4) -- split-K factor; 1 disables.
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
    float                softmax_scale) {
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int K = static_cast<int>(indices.size(2));

    if ((H % kHeadsPerCta) != 0) return false;

    const auto stream = at::cuda::getCurrentCUDAStream();

    auto env_int = [](const char* name, int defv) -> int {
        const char* v = std::getenv(name);
        if (!v || *v == '\0') return defv;
        return std::atoi(v);
    };
    const bool use_fp8_mma = env_int("DG_SM120_FUSED_DECODE_V2_FP8MMA", 1) != 0;
    int split_k = env_int("DG_SM120_FUSED_DECODE_V2_SPLITK", 4);
    if (split_k < 1) split_k = 1;
    if (split_k > 8) split_k = 8;
    while (split_k > 1 && K < split_k * kCandsPerChunk) split_k /= 2;

    int64_t cache_block_stride_bytes;
    if (k_cache.dim() >= 4) {
        cache_block_stride_bytes = static_cast<int64_t>(k_cache.stride(0));
    } else {
        cache_block_stride_bytes =
            static_cast<int64_t>(k_cache.stride(0)) * k_cache.element_size();
    }

    const int* topk_ptr =
        topk_length.defined() ? topk_length.data_ptr<int>() : nullptr;
    const float* sink_ptr =
        attn_sink.defined() ? attn_sink.data_ptr<float>() : nullptr;

    const dim3 block(kNativeThreads);
    const size_t shared_bytes = static_cast<size_t>(kNativeSmemBytes);

#define DG_SET_SMEM_ATTR(KFN_PTR)                                            \
    do {                                                                     \
        if (shared_bytes > 48 * 1024) {                                      \
            cudaFuncSetAttribute(                                            \
                reinterpret_cast<const void*>(KFN_PTR),                      \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                 \
                static_cast<int>(shared_bytes));                             \
        }                                                                    \
    } while (0)

    if (split_k == 1) {
        const dim3 grid(B, H / kHeadsPerCta, 1);
        if (use_fp8_mma) {
            DG_SET_SMEM_ATTR(
                (&sm120_fused_decode_v2_native_kernel<QT, OutT, IdxT, 1, true>));
            sm120_fused_decode_v2_native_kernel<QT, OutT, IdxT, 1, true>
                <<<grid, block, shared_bytes, stream>>>(
                    reinterpret_cast<const QT*>(q.data_ptr()),
                    reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
                    indices.data_ptr<IdxT>(),
                    topk_ptr, sink_ptr,
                    reinterpret_cast<OutT*>(out.data_ptr()),
                    lse.data_ptr<float>(),
                    B, H, K,
                    q.stride(0), q.stride(1),
                    out.stride(0), out.stride(1),
                    /*out_stride_split=*/0,
                    lse.stride(0),
                    /*lse_stride_split=*/0,
                    cache_block_stride_bytes,
                    block_size,
                    indices.stride(0),
                    softmax_scale);
        } else {
            DG_SET_SMEM_ATTR(
                (&sm120_fused_decode_v2_native_kernel<QT, OutT, IdxT, 1, false>));
            sm120_fused_decode_v2_native_kernel<QT, OutT, IdxT, 1, false>
                <<<grid, block, shared_bytes, stream>>>(
                    reinterpret_cast<const QT*>(q.data_ptr()),
                    reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
                    indices.data_ptr<IdxT>(),
                    topk_ptr, sink_ptr,
                    reinterpret_cast<OutT*>(out.data_ptr()),
                    lse.data_ptr<float>(),
                    B, H, K,
                    q.stride(0), q.stride(1),
                    out.stride(0), out.stride(1),
                    /*out_stride_split=*/0,
                    lse.stride(0),
                    /*lse_stride_split=*/0,
                    cache_block_stride_bytes,
                    block_size,
                    indices.stride(0),
                    softmax_scale);
        }
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return true;
    }

    auto opts_q = q.options();
    auto opts_f32 = q.options().dtype(torch::kFloat32);
    auto out_partials = torch::empty({B, split_k, H, kHeadDim}, opts_q);
    auto lse_partials = torch::empty({B, split_k, H}, opts_f32);

    const int64_t op_stride_b      = out_partials.stride(0);
    const int64_t op_stride_split  = out_partials.stride(1);
    const int64_t op_stride_h      = out_partials.stride(2);
    const int64_t lp_stride_b      = lse_partials.stride(0);
    const int64_t lp_stride_split  = lse_partials.stride(1);

    const dim3 grid(B, H / kHeadsPerCta, split_k);

#define DG_LAUNCH_NATIVE_KERNEL(SPLITK_VAL, USE_FP8)                         \
    do {                                                                     \
        DG_SET_SMEM_ATTR((&sm120_fused_decode_v2_native_kernel<              \
            QT, OutT, IdxT, SPLITK_VAL, USE_FP8>));                          \
        sm120_fused_decode_v2_native_kernel<                                 \
            QT, OutT, IdxT, SPLITK_VAL, USE_FP8>                             \
            <<<grid, block, shared_bytes, stream>>>(                         \
                reinterpret_cast<const QT*>(q.data_ptr()),                   \
                reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),        \
                indices.data_ptr<IdxT>(),                                    \
                topk_ptr, sink_ptr,                                          \
                reinterpret_cast<OutT*>(out_partials.data_ptr()),            \
                lse_partials.data_ptr<float>(),                              \
                B, H, K,                                                     \
                q.stride(0), q.stride(1),                                    \
                op_stride_b, op_stride_h,                                    \
                op_stride_split,                                             \
                lp_stride_b,                                                 \
                lp_stride_split,                                             \
                cache_block_stride_bytes,                                    \
                block_size,                                                  \
                indices.stride(0),                                           \
                softmax_scale);                                              \
    } while (0)

    if (use_fp8_mma) {
        switch (split_k) {
            case 2: DG_LAUNCH_NATIVE_KERNEL(2, true); break;
            case 4: DG_LAUNCH_NATIVE_KERNEL(4, true); break;
            case 8: DG_LAUNCH_NATIVE_KERNEL(8, true); break;
            default: return false;
        }
    } else {
        switch (split_k) {
            case 2: DG_LAUNCH_NATIVE_KERNEL(2, false); break;
            case 4: DG_LAUNCH_NATIVE_KERNEL(4, false); break;
            case 8: DG_LAUNCH_NATIVE_KERNEL(8, false); break;
            default: return false;
        }
    }
#undef DG_LAUNCH_NATIVE_KERNEL
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const dim3 reduce_grid(B, H);
    const dim3 reduce_block(kHeadDim);

#define DG_LAUNCH_REDUCE(SPLITK_VAL)                                         \
    sm120_fused_decode_v2_splitk_reduce_kernel<OutT, SPLITK_VAL>             \
        <<<reduce_grid, reduce_block, 0, stream>>>(                          \
            reinterpret_cast<const OutT*>(out_partials.data_ptr()),          \
            lse_partials.data_ptr<float>(),                                  \
            reinterpret_cast<OutT*>(out.data_ptr()),                         \
            lse.data_ptr<float>(),                                           \
            B, H,                                                            \
            op_stride_b, op_stride_split, op_stride_h,                       \
            lp_stride_b, lp_stride_split,                                    \
            out.stride(0), out.stride(1),                                    \
            lse.stride(0))

    switch (split_k) {
        case 2: DG_LAUNCH_REDUCE(2); break;
        case 4: DG_LAUNCH_REDUCE(4); break;
        case 8: DG_LAUNCH_REDUCE(8); break;
        default: return false;
    }
#undef DG_LAUNCH_REDUCE
#undef DG_SET_SMEM_ATTR
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    return true;
}

template bool launch_sm120_fused_decode_v2_native<__nv_bfloat16, __nv_bfloat16, int32_t>(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    torch::Tensor&, torch::Tensor&, int, float);
template bool launch_sm120_fused_decode_v2_native<__nv_bfloat16, __nv_bfloat16, int64_t>(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    torch::Tensor&, torch::Tensor&, int, float);
template bool launch_sm120_fused_decode_v2_native<half, half, int32_t>(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    torch::Tensor&, torch::Tensor&, int, float);
template bool launch_sm120_fused_decode_v2_native<half, half, int64_t>(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    torch::Tensor&, torch::Tensor&, int, float);

}  // namespace native
}  // namespace sm120_mla_v2
}  // namespace deep_gemm
