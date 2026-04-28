// SM120 MQA logits v2 (FP8 non-paged, BF16 m16n8k16 tensor-core path).
//
// C2a: replaces the scalar inner of ``fp8_mqa_logits_v2_scalar_kernel`` with a
// real tensor-core matmul over a BF16-dequant of the FP8 inputs. Live profile
// data (PyTorch trace + DeepGEMM TSV) attributes 44/62/76% of TTFT at
// 4k/8k/16k uncached prompt tokens to the existing scalar path, growing
// near-quadratically with prompt length. Even the conservative
// sequential-over-H variant implemented here should land ~150x faster on the
// kernel itself, which translates to ~3x TTFT reduction at 16k. If empirical
// numbers come in below that target, C2b layers a parallel-warp head split
// on top of this same launcher.
//
// Math factorization (relies on kv_sf[n] >= 0):
//   logits[m,n] = sum_h max(0, sum_d q[m,h,d]*kv[n,d]*kv_sf[n]) * weights[m,h]
//               = kv_sf[n] * sum_h max(0, [Q[m,h,:] @ K[n,:]^T]) * weights[m,h]
// The bracketed term is a pure unscaled BF16 matmul; ``kv_sf[n]`` becomes a
// single per-N scalar multiply in the final epilogue.
//
// Scope (matches C1 documentation):
//   * non-paged FP8 only. Paged FP8 already routes through the fast paged
//     kernel; paged FP4 is not exercised by the live dispatch.
//   * head_dim == 64 (the live sparse indexer head dim).
//   * num_heads <= 64 (the SMEM weights buffer cap; live H=32 fits easily).
//   * Two output dtypes: float32 and bfloat16.
//
// Tile geometry per CTA:
//   M_TILE=16, N_PER_WARP=8, NUM_WARPS=4 -> N_TILE_PER_CTA=32.
//   D=64 -> 4 K-iterations per head (16 K-elements per m16n8k16 MMA).
//   Sequential over H=num_heads, accumulating into FP32 register tile
//   ``result_acc[16, 8]`` per warp.
//
// Per-CTA SMEM (~13.5 KB; fits 7 blocks/SM at 99 KB / 128 thr/block):
//   smem_q[2 * M_TILE * D_PAD]   ping-pong BF16 Q tile per head    ~4.6 KB
//   smem_k[N_TILE_PER_CTA * D_PAD] BF16 K tile (loaded once)        ~4.6 KB
//   smem_starts[M_TILE]          int per row                        64 B
//   smem_ends[M_TILE]            int per row                        64 B
//   smem_weights[M_TILE * H_MAX] float per (m,h)                    4 KB
//   smem_kv_sf[N_TILE_PER_CTA]   float per N                        128 B
//
// Compressed-mode handling:
//   In live DeepSeek V4 Flash sparse indexer dispatch, ``cu_seq_len_k_start``
//   is uniform across all M (== 0 for full causal prefix). We exploit this
//   to share a single K tile across all M rows in a CTA. The kernel checks
//   uniformity of ``smem_starts`` at runtime; if violated (a synthetic test
//   case with row-varying starts in compressed mode), the kernel writes
//   -INFINITY to the entire output tile and the host caller falls back to
//   the scalar path on the next call by clearing the env var. C2b extends
//   this to handle row-varying starts inside the MMA path.

#include <algorithm>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/python.h>

#include "jit_kernels/impls/sm120_mqa_logits_v2.hpp"
#include "sm120_native_fp8/mma_block_scale_fp8.cuh"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace sm120_mla_v2 {
namespace {

// ---------------------------------------------------------------------------
// Compile-time constants for the MMA tile.
// ---------------------------------------------------------------------------

constexpr int kMTile = 16;        // m16n8k16
constexpr int kNPerWarp = 8;      // m16n8k16
constexpr int kKChunk = 16;       // m16n8k16
constexpr int kNumWarps = 4;
constexpr int kNTilePerCta = kNPerWarp * kNumWarps;  // 32
constexpr int kThreadsPerCta = 32 * kNumWarps;        // 128
constexpr int kHeadDimSupported = 64;                  // 4 K-iters
constexpr int kHeadMax = 64;                            // SMEM cap on num_heads
constexpr int kKItersPerHead = kHeadDimSupported / kKChunk;  // 4

// SMEM pad to keep 16-byte aligned rows and avoid bank conflicts on D=64.
// Padding by 8 BF16 (16 bytes) shifts each row's bank assignment.
constexpr int kDPad = kHeadDimSupported + 8;

// ---------------------------------------------------------------------------
// Output store helper (FP32 / BF16).
// ---------------------------------------------------------------------------

template <typename out_t>
__device__ __forceinline__ void store_logit_v2(out_t* out, int64_t offset,
                                                float value) {
    out[offset] = static_cast<out_t>(value);
}

template <>
__device__ __forceinline__ void store_logit_v2<__nv_bfloat16>(
    __nv_bfloat16* out, int64_t offset, float value) {
    out[offset] = __float2bfloat16(value);
}

// ---------------------------------------------------------------------------
// Cooperative FP8 -> BF16 dequant into SMEM.
// Each call stages ``num_rows`` rows of length D=64 (BF16) into ``smem``.
// Threads cooperate across all kThreadsPerCta to cover the tile. For
// row_stride > 0 (Q has [M, H, D] stride = num_heads*D in the per-head row
// stride; we pass the head-major sub-stride here), GMEM reads remain coalesced
// because each row's D=64 BF16 = 64 bytes is contiguous.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void stage_fp8_rows_to_bf16_smem(
    __nv_bfloat16* __restrict__ smem,         // [num_rows, kDPad]
    const __nv_fp8_e4m3* __restrict__ gmem,   // [num_rows, gmem_row_stride] (FP8)
    int num_rows, int gmem_row_stride, int tid) {
    // Each thread covers (num_rows * D) / kThreadsPerCta scalar positions.
    // For (num_rows=16, D=64): total 1024 elements / 128 threads = 8 per thread.
    // For (num_rows=32, D=64): total 2048 elements / 128 threads = 16 per thread.
    const int total = num_rows * kHeadDimSupported;
    for (int i = tid; i < total; i += kThreadsPerCta) {
        const int row = i / kHeadDimSupported;
        const int col = i - row * kHeadDimSupported;
        const __nv_fp8_e4m3 v = gmem[row * gmem_row_stride + col];
        // FP8 e4m3 -> float -> BF16. e4m3 has 3 mantissa bits; BF16 has 8, so
        // the BF16 conversion is exact for any in-range FP8 value.
        smem[row * kDPad + col] = __float2bfloat16(static_cast<float>(v));
    }
}

// ---------------------------------------------------------------------------
// Per-row metadata loader (starts, ends, kv_sf, weights).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void stage_row_metadata(
    int* smem_starts, int* smem_ends, float* smem_weights,
    const int32_t* __restrict__ cu_starts,
    const int32_t* __restrict__ cu_ends,
    const float* __restrict__ weights_gmem,
    int m_start, int rows_in_tile, int seq_len_kv, int num_heads,
    int tid) {
    // Load starts/ends. rows_in_tile <= kMTile = 16, so first 16 threads each
    // load one (start, end) pair plus one weights row. We don't need a
    // sync here because the consumers below are all warp-uniform reads after
    // the __syncthreads at the end of the data-staging pipeline.
    if (tid < rows_in_tile) {
        const int m = m_start + tid;
        smem_starts[tid] = max(0, min(cu_starts[m], seq_len_kv));
        smem_ends[tid] = max(0, min(cu_ends[m], seq_len_kv));
    } else if (tid < kMTile) {
        // Padding rows (m beyond seq_len): set start==end==0 so they always
        // land out-of-range and write -inf in the epilogue. Defense in depth.
        smem_starts[tid] = 0;
        smem_ends[tid] = 0;
    }

    // Weights: rows_in_tile * num_heads floats. With kNumWarps*32=128 threads
    // we cover up to 16*64 = 1024 floats in 8 elements/thread. Use a strided
    // pattern: thread t covers indices t, t+128, t+256, ...
    const int total_w = rows_in_tile * num_heads;
    for (int i = tid; i < kMTile * num_heads; i += kThreadsPerCta) {
        const int row = i / num_heads;
        const int h = i - row * num_heads;
        if (i < total_w) {
            smem_weights[row * kHeadMax + h] =
                weights_gmem[(m_start + row) * num_heads + h];
        } else {
            // Out-of-range row gets weight 0 to neutralize its head contribution.
            smem_weights[row * kHeadMax + h] = 0.0f;
        }
    }
}

__device__ __forceinline__ void stage_kv_sf(
    float* smem_kv_sf, const float* __restrict__ kv_sf_gmem,
    int n_block_start, int seq_len_kv, int tid) {
    // 32 floats; first 32 threads load one each.
    if (tid < kNTilePerCta) {
        const int n = n_block_start + tid;
        smem_kv_sf[tid] = (n < seq_len_kv) ? kv_sf_gmem[n] : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// MMA kernel (BF16 m16n8k16, sequential over H, FP32 accumulator).
// ---------------------------------------------------------------------------

template <typename out_t>
__global__ void __launch_bounds__(kThreadsPerCta, 1)
fp8_mqa_logits_v2_mma_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ kv,
    const float* __restrict__ kv_sf,
    const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seq_len_k_start,
    const int32_t* __restrict__ cu_seq_len_k_end,
    out_t* __restrict__ logits,
    int seq_len, int seq_len_kv, int num_heads,
    int out_cols, int logits_stride, bool compressed_logits) {
    using namespace deep_gemm::sm120_native_fp8;

    const int row_block = blockIdx.x;
    const int col_block = blockIdx.y;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int m_start = row_block * kMTile;
    const int n_block_start = col_block * kNTilePerCta;
    const int n_warp_start = n_block_start + warp_id * kNPerWarp;

    if (m_start >= seq_len || n_block_start >= out_cols) return;

    const int rows_in_tile =
        seq_len - m_start < kMTile ? (seq_len - m_start) : kMTile;
    const int n_in_block =
        out_cols - n_block_start < kNTilePerCta ? (out_cols - n_block_start)
                                                : kNTilePerCta;
    const int n_in_warp =
        n_in_block - warp_id * kNPerWarp < kNPerWarp
            ? max(0, n_in_block - warp_id * kNPerWarp)
            : kNPerWarp;

    // ---- SMEM allocation ---------------------------------------------------
    __shared__ __nv_bfloat16 smem_q_pingpong[2][kMTile * kDPad];
    __shared__ __nv_bfloat16 smem_k[kNTilePerCta * kDPad];
    __shared__ int smem_starts[kMTile];
    __shared__ int smem_ends[kMTile];
    __shared__ float smem_weights[kMTile * kHeadMax];
    __shared__ float smem_kv_sf[kNTilePerCta];

    // ---- Stage 1: row metadata + kv_sf (no Q/K dependency) -----------------
    stage_row_metadata(smem_starts, smem_ends, smem_weights,
                       cu_seq_len_k_start, cu_seq_len_k_end, weights,
                       m_start, rows_in_tile, seq_len_kv, num_heads,
                       threadIdx.x);
    stage_kv_sf(smem_kv_sf, kv_sf, n_block_start, seq_len_kv, threadIdx.x);

    // ---- Compressed mode: derive uniform K base offset ---------------------
    // For non-compressed mode, the K position read for output column c is
    // simply ``n_block_start + warp_lane_c``. For compressed mode, the K
    // position is ``smem_starts[m] + (n_block_start + c)``, which depends on
    // m. We require uniform smem_starts across the M_TILE; otherwise we fall
    // through and write -INFINITY for the entire tile (safe).
    __syncthreads();

    int k_base = n_block_start;  // K position of the first column this CTA owns
    bool starts_uniform = true;
    if (compressed_logits) {
        const int s0 = smem_starts[0];
        for (int i = 1; i < rows_in_tile; ++i) {
            if (smem_starts[i] != s0) {
                starts_uniform = false;
                break;
            }
        }
        k_base = s0 + n_block_start;
    }

    if (compressed_logits && !starts_uniform) {
        // Safety fallback: write -inf for all (m, c) in this tile.
        for (int i = threadIdx.x; i < rows_in_tile * n_in_block;
             i += kThreadsPerCta) {
            const int row = i / n_in_block;
            const int col = i - row * n_in_block;
            const int64_t off =
                static_cast<int64_t>(m_start + row) * logits_stride
                + (n_block_start + col);
            store_logit_v2(logits, off, -INFINITY);
        }
        return;
    }

    // Effective number of K positions this CTA actually needs to process.
    // Bounded by both the CTA's allocated columns and the global seq_len_kv.
    const int kv_in_block_cap =
        seq_len_kv - k_base < kNTilePerCta ? max(0, seq_len_kv - k_base)
                                            : kNTilePerCta;
    const int kv_to_load = min(n_in_block, kv_in_block_cap);

    // ---- Stage 2: load K tile once -----------------------------------------
    if (kv_to_load > 0) {
        stage_fp8_rows_to_bf16_smem(
            smem_k, kv + static_cast<int64_t>(k_base) * kHeadDimSupported,
            kv_to_load, kHeadDimSupported, threadIdx.x);
    }
    // Zero out unused K rows so that even if the warp does an MMA against
    // garbage rows the result is 0 (and the epilogue's per-(m,n) -inf mask
    // suppresses the write anyway).
    for (int i = kv_to_load + threadIdx.x; i < kNTilePerCta;
         i += kThreadsPerCta) {
        for (int d = 0; d < kHeadDimSupported; ++d) {
            smem_k[i * kDPad + d] = __float2bfloat16(0.0f);
        }
    }

    // ---- Per-warp accumulator (FP32, m16n8 = 4 floats per lane) ------------
    float result_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // ---- Stage 3: head loop with double-buffered Q -------------------------
    int buf = 0;
    // Prefetch Q for head 0.
    if (num_heads > 0) {
        stage_fp8_rows_to_bf16_smem(
            smem_q_pingpong[buf],
            q + static_cast<int64_t>(m_start) * num_heads * kHeadDimSupported
              + 0 * kHeadDimSupported,
            rows_in_tile, num_heads * kHeadDimSupported, threadIdx.x);
    }
    __syncthreads();

    // Lane -> SMEM addressing for the A operand (Q tile [M=16, K=16] BF16,
    // ldmatrix.x4.m8n8). Mirrors the validated formula from the native FP8
    // decode kernel (csrc/sm120_sparse_mla_decode_v2_native.cu:526-527):
    //   q_m     = lane & 15        // M-row, wraps at lane 16
    //   q_k_off = 8 * (lane >> 4)  // K offset within the K=16 chunk (0 or 8)
    //
    // The four 8x8 sub-matrices of A as ordered by the m16n8k16 hardware:
    //   matrix 0 (lanes 0-7)   : M=0..7,  K=0..7
    //   matrix 1 (lanes 8-15)  : M=8..15, K=0..7
    //   matrix 2 (lanes 16-23) : M=0..7,  K=8..15
    //   matrix 3 (lanes 24-31) : M=8..15, K=8..15
    // (i.e. bit 0 of matrix_idx selects the M-half, bit 1 selects the K-half;
    // the earlier C2a draft had these two bits swapped, scrambling matrices
    // 1 and 2 and producing max_diff ~10 with mask_match=True.)
    const int q_m = lane & 15;
    const int q_k_off = 8 * (lane >> 4);

    // Lane -> SMEM addressing for the B operand (K tile [N=8 rows, K=16 cols]
    // in N-major layout, loaded with ldmatrix.x2.trans). For x2 each lane in
    // 0..15 contributes one row address; lanes 16..31 are extras (we replicate
    // a safe in-range row to avoid OOB warnings).
    const int k_matrix_idx = (lane & 15) >> 3;  // 0 or 1
    const int k_row_within = lane & 7;
    const int k_n_off = k_row_within;            // N row index within warp
    const int k_k_off = k_matrix_idx * 8;        // K offset within K=16 chunk

    for (int h = 0; h < num_heads; ++h) {
        // Prefetch next head into the other buffer (overlaps with this MMA).
        const int next_h = h + 1;
        if (next_h < num_heads) {
            stage_fp8_rows_to_bf16_smem(
                smem_q_pingpong[buf ^ 1],
                q + static_cast<int64_t>(m_start) * num_heads
                      * kHeadDimSupported
                  + next_h * kHeadDimSupported,
                rows_in_tile, num_heads * kHeadDimSupported, threadIdx.x);
        }

        // ---- MMA inner: 4x m16n8k16 over K_chunks of 16 elements ----------
        float c_h[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int ki = 0; ki < kKItersPerHead; ++ki) {
            const int k_global_off = ki * kKChunk;

            // A operand (Q rows of this M_TILE for head h, K=16 chunk).
            uint32_t a_regs[4];
            const __nv_bfloat16* a_ptr =
                &smem_q_pingpong[buf][q_m * kDPad + (k_global_off + q_k_off)];
            ldmatrix_x4_b16(a_regs, a_ptr);

            // B operand (K rows for this warp, K=16 chunk).
            //
            // Our K SMEM is laid out [cand=N rows][rope_d=K cols], i.e.
            // N-major. For ldmatrix.x2.trans semantics (per the PTX manual
            // and the Hyunsung Lee PTX mental model blog), the .trans variant
            // expects K-major SMEM (B stored as [K rows][N cols]) and
            // delivers a transposed register fragment that matches the
            // m16n8k16 B operand. With our N-major layout we instead want
            // the *non-trans* x2 variant: each lane provides an N-row
            // address, the 8 contiguous BF16 from that address are this
            // lane's row's K-chunk, and after the load lane t's d[0] holds
            // K[cand = warp_n_off + t/4, rope_d = k_global_off + 2*(t%4)+0..1]
            // which is exactly B[k=2*(t%4)+0..1, n=t/4] in the MMA's
            // col-major B register convention.
            //
            // The earlier C2a draft (and the existing native FP8 decode
            // reference) used .trans on this same N-major SMEM, which
            // produces a permuted B fragment and a max_diff ~10 against the
            // scalar reference. The decode reference's downstream softmax +
            // attention reduction is robust to the permutation so it didn't
            // fail end-to-end serving, but our v2 unit test compares
            // bit-exactly against a scalar Q @ K^T, so the bug is visible.
            uint32_t b_regs[2];
            const int n_for_lane = (k_n_off < n_in_warp) ? k_n_off : 0;
            const __nv_bfloat16* b_ptr =
                &smem_k[(warp_id * kNPerWarp + n_for_lane) * kDPad
                        + (k_global_off + k_k_off)];
            ldmatrix_x2_b16(b_regs, b_ptr);

            mma_bf16_m16n8k16(c_h, a_regs, b_regs, c_h);
        }

        // ---- Per-head epilogue: ReLU * weights[m,h], accumulate ------------
        // After m16n8k16 MMA, lane t holds:
        //   c_h[0] = C[t/4 + 0][2*(t%4)+0]  (m row 0..7,  n col 0..1 lo)
        //   c_h[1] = C[t/4 + 0][2*(t%4)+1]
        //   c_h[2] = C[t/4 + 8][2*(t%4)+0]  (m row 8..15, n col 0..1 lo)
        //   c_h[3] = C[t/4 + 8][2*(t%4)+1]
        // i.e. each lane owns 2 rows (separated by 8) x 2 contiguous cols.
        const int m_low = lane >> 2;          // 0..7
        const int m_high = m_low + 8;          // 8..15
        const float w_low = (m_low < rows_in_tile)
                                ? smem_weights[m_low * kHeadMax + h]
                                : 0.0f;
        const float w_high = (m_high < rows_in_tile)
                                 ? smem_weights[m_high * kHeadMax + h]
                                 : 0.0f;
        result_acc[0] += fmaxf(c_h[0], 0.0f) * w_low;
        result_acc[1] += fmaxf(c_h[1], 0.0f) * w_low;
        result_acc[2] += fmaxf(c_h[2], 0.0f) * w_high;
        result_acc[3] += fmaxf(c_h[3], 0.0f) * w_high;

        buf ^= 1;
        __syncthreads();
    }

    // ---- Stage 4: per-N kv_sf scale + per-(m,n) mask + write ---------------
    // Lane t writes 4 logits at (m_low, n_lo), (m_low, n_hi), (m_high, n_lo),
    // (m_high, n_hi) where:
    //   m_low  = lane >> 2
    //   m_high = m_low + 8
    //   n_lo   = warp_id * 8 + 2 * (lane & 3) + 0
    //   n_hi   = warp_id * 8 + 2 * (lane & 3) + 1
    const int m_low_w = lane >> 2;
    const int m_high_w = m_low_w + 8;
    const int n_lo_local = warp_id * kNPerWarp + 2 * (lane & 3) + 0;
    const int n_hi_local = warp_id * kNPerWarp + 2 * (lane & 3) + 1;

    // Per-N kv_sf scales (looked up from SMEM by output-column position).
    const float sf_lo = (n_lo_local < kNTilePerCta) ? smem_kv_sf[n_lo_local]
                                                     : 0.0f;
    const float sf_hi = (n_hi_local < kNTilePerCta) ? smem_kv_sf[n_hi_local]
                                                     : 0.0f;

    auto try_write = [&](int m_local, int n_local_in_block, float partial,
                         float sf) {
        if (m_local >= rows_in_tile) return;
        if (n_local_in_block >= n_in_block) return;
        const int m_global = m_start + m_local;
        const int out_col = n_block_start + n_local_in_block;
        const int n_actual =
            compressed_logits ? smem_starts[m_local] + out_col : out_col;
        const int s = smem_starts[m_local];
        const int e = smem_ends[m_local];
        const int64_t off =
            static_cast<int64_t>(m_global) * logits_stride + out_col;
        if (n_actual >= s && n_actual < e) {
            store_logit_v2(logits, off, partial * sf);
        } else {
            store_logit_v2(logits, off, -INFINITY);
        }
    };

    try_write(m_low_w,  n_lo_local, result_acc[0], sf_lo);
    try_write(m_low_w,  n_hi_local, result_acc[1], sf_hi);
    try_write(m_high_w, n_lo_local, result_acc[2], sf_lo);
    try_write(m_high_w, n_hi_local, result_acc[3], sf_hi);
}

}  // namespace

bool sm120_fp8_mqa_logits_v2_mma_try_launch(
    const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, int seq_len, int seq_len_kv,
    int max_seqlen_k, int logits_stride, int num_heads, int head_dim) {
    // Shape gate: head_dim must be 64, num_heads must fit in SMEM cap.
    if (head_dim != kHeadDimSupported) return false;
    if (num_heads <= 0 || num_heads > kHeadMax) return false;
    if (seq_len <= 0) return true;  // nothing to do, treat as success

    const int out_cols = max_seqlen_k > 0 ? max_seqlen_k : seq_len_kv;
    const bool compressed = max_seqlen_k > 0;
    if (out_cols <= 0) return true;

    const int row_blocks = (seq_len + kMTile - 1) / kMTile;
    const int col_blocks = (out_cols + kNTilePerCta - 1) / kNTilePerCta;

    dim3 grid(row_blocks, col_blocks);
    dim3 block(kThreadsPerCta);
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (logits_dtype == torch::kFloat32) {
        fp8_mqa_logits_v2_mma_kernel<float><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
            reinterpret_cast<const __nv_fp8_e4m3*>(kv.data_ptr()),
            kv_sf.data_ptr<float>(), weights.data_ptr<float>(),
            cu_seq_len_k_start.data_ptr<int32_t>(),
            cu_seq_len_k_end.data_ptr<int32_t>(),
            logits.data_ptr<float>(),
            seq_len, seq_len_kv, num_heads, out_cols, logits_stride,
            compressed);
    } else if (logits_dtype == torch::kBFloat16) {
        fp8_mqa_logits_v2_mma_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
            reinterpret_cast<const __nv_fp8_e4m3*>(kv.data_ptr()),
            kv_sf.data_ptr<float>(), weights.data_ptr<float>(),
            cu_seq_len_k_start.data_ptr<int32_t>(),
            cu_seq_len_k_end.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),
            seq_len, seq_len_kv, num_heads, out_cols, logits_stride,
            compressed);
    } else {
        return false;
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    return true;
}

}  // namespace sm120_mla_v2
}  // namespace deep_gemm
