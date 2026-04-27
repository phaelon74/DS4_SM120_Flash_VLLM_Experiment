// SM120 native FP8 path: UE8M0 codec + block quantizers + FP8 packing.
//
// Three jobs in this header:
//
// 1. UE8M0 scale codec
//    UE8M0 = unsigned 8-bit exponent, no mantissa, bias 127. byte 0 is a
//    "scale = 0" sentinel in DSv4's scalar path, but the block-scale MMA
//    treats scale as a multiplicative factor in the inner loop and CANNOT
//    distinguish a zero-scale from a normal one. So we clamp byte 0 -> byte 1
//    (= 2^-126) at quantization time and rely on the calling kernel to mask
//    invalid candidates via softmax (-inf logit) instead of via a zero scale.
//
// 2. Block quantizers
//    BF16 -> FP8 e4m3 + UE8M0 (block size 64): used to quantize Q on entry.
//    FP32 -> FP8 e4m3 + UE8M0 (block size = chunk_size): used to quantize P
//    after softmax. The FP32->FP8 path is novel; no reference impl. We use
//    pow2-round-up of (rowmax / 448.0) for the scale, which is the same
//    convention DSv4 uses for activation quant in the indexer.
//
// 3. FP8 packing
//    The block-scaled m16n8k32 MMA wants its A and B operands as packed
//    uint32_t (4 fp8 per u32). This header provides the per-thread pack
//    routines that take a contiguous run of 4 fp8 values from SMEM and
//    return a u32.
//
// Numerical discipline:
//   * E4M3FN max = 448.0 (S.1111.110); we saturate to that on overflow.
//   * UE8M0 max byte = 254 (= 2^127); above that we saturate to 254.
//   * UE8M0 byte 1 = 2^-126 (sub-tiny scale clamp for would-be-zero blocks).
//   * Round-to-nearest-even is provided by NVIDIA's __nv_cvt_float_to_fp8.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace deep_gemm {
namespace sm120_native_fp8 {

constexpr float kE4M3Max = 448.0f;

// ---- UE8M0 codec ----------------------------------------------------------

// Decode a UE8M0 byte to its multiplicative scale. byte 0 is encoded as
// "scale = 0" in DSv4's scalar path; we treat it as a clamp-to-tiny here so
// the block_scale MMA sees a finite scale. The kernel masks invalid rows via
// softmax -inf, not via zero-scale, so this is safe.
__device__ __forceinline__ float ue8m0_to_float(uint8_t byte) {
    const uint8_t b = byte == 0 ? 1u : byte;
    return __uint_as_float(static_cast<uint32_t>(b) << 23);
}

// Encode a power-of-two `scale` into a UE8M0 byte. `scale` MUST already be
// a non-negative power of two (caller used pow2_round_up). Subnormals and
// negatives are clamped to byte 1.
__device__ __forceinline__ uint8_t float_pow2_to_ue8m0(float scale) {
    if (!(scale > 0.0f)) return 1u;
    const uint32_t bits = __float_as_uint(scale);
    const uint32_t exp = (bits >> 23) & 0xffu;
    if (exp == 0u) return 1u;     // denormal: clamp to 2^-126
    if (exp == 0xffu) return 254u; // inf/NaN: clamp to max representable
    return static_cast<uint8_t>(exp);
}

// pow2_round_up(x) = 2^ceil(log2(x)). Used to derive the block scale from a
// raw max-abs value. For x <= 0 returns 2^-126 (matches UE8M0 byte 1).
__device__ __forceinline__ float pow2_round_up(float x) {
    if (!(x > 0.0f)) return __uint_as_float(1u << 23);
    const uint32_t bits = __float_as_uint(x);
    const uint32_t mantissa = bits & 0x007fffffu;
    uint32_t exp = (bits >> 23) & 0xffu;
    if (mantissa != 0u) exp += 1u;
    if (exp == 0u) exp = 1u;
    if (exp >= 0xffu) exp = 0xfeu;
    return __uint_as_float(exp << 23);
}

// ---- FP8 e4m3 conversion --------------------------------------------------

// float -> fp8 e4m3 (saturating, round-to-nearest-even). NVIDIA provides
// __nv_cvt_float_to_fp8 in cuda_fp8.h with E4M3 + saturation behavior.
__device__ __forceinline__ uint8_t float_to_e4m3(float v) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        v, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint8_t bf16_to_e4m3(__nv_bfloat16 v) {
    return float_to_e4m3(__bfloat162float(v));
}

// fp8 e4m3 -> float (decode for the bf16-debug-mode load path).
__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
    __nv_fp8_e4m3 f;
    f.__x = b;
    return static_cast<float>(f);
}

// ---- FP8 packing ----------------------------------------------------------

// Pack 4 fp8 bytes into a uint32_t (byte-0 lowest). The block-scaled MMA
// expects this exact packing for both A and B operands.
__device__ __forceinline__ uint32_t pack_4_fp8(
    uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return (static_cast<uint32_t>(b0))       |
           (static_cast<uint32_t>(b1) <<  8) |
           (static_cast<uint32_t>(b2) << 16) |
           (static_cast<uint32_t>(b3) << 24);
}

__device__ __forceinline__ uint32_t pack_4_fp8(const uint8_t* src) {
    return pack_4_fp8(src[0], src[1], src[2], src[3]);
}

// Pack a single UE8M0 scale byte into byte 0 of a uint32_t. The MMA uses
// byte_id_a/byte_id_b immediates fixed at 0 (see mma_block_scale_fp8.cuh).
__device__ __forceinline__ uint32_t pack_scale_byte0(uint8_t b) {
    return static_cast<uint32_t>(b);
}

// ---- Warp-level block reductions -----------------------------------------

// Max-abs over `kWidth` consecutive lanes. Result is broadcast to all lanes
// in the group. Caller picks `kWidth` from {2, 4, 8, 16, 32}.
template <int kWidth>
__device__ __forceinline__ float warp_max_abs(float v) {
    v = fabsf(v);
#pragma unroll
    for (int off = kWidth >> 1; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

// Sum over `kWidth` consecutive lanes.
template <int kWidth>
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = kWidth >> 1; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

// ---- Block quantizers ----------------------------------------------------

// Quantize a single block of `kBlockSize` FP32 values held in registers
// (`vals[kPerThread]` per thread, where `kBlockSize == kPerThread * kWidth`)
// into FP8 e4m3 + a single UE8M0 scale byte for the block.
//
// Returns the per-thread fp8 outputs in `out_fp8[kPerThread]` and the shared
// UE8M0 scale byte (broadcast to every lane in the group).
//
// `kWidth` is the warp lane group cooperating on this block (1, 2, 4, 8, 16,
// or 32). All threads in the group must call this with their respective
// values; the caller is responsible for arranging that grouping.
template <int kPerThread, int kWidth>
__device__ __forceinline__ void quantize_block_fp32_to_e4m3(
    const float (&vals)[kPerThread],
    uint8_t     (&out_fp8)[kPerThread],
    uint8_t&     ue8m0_byte) {
    static_assert(kWidth >= 1 && kWidth <= 32, "kWidth must be in [1,32]");

    float local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
        local_max = fmaxf(local_max, fabsf(vals[i]));
    }
    const float block_max =
        kWidth == 1 ? local_max : warp_max_abs<kWidth>(local_max);

    const float ideal = block_max / kE4M3Max;
    const float scale = pow2_round_up(ideal);
    const float inv_scale = 1.0f / scale;
    ue8m0_byte = float_pow2_to_ue8m0(scale);

#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
        out_fp8[i] = float_to_e4m3(vals[i] * inv_scale);
    }
}

// BF16 variant for the Q-quantization on kernel entry. Identical structure;
// just promotes to FP32 first.
template <int kPerThread, int kWidth>
__device__ __forceinline__ void quantize_block_bf16_to_e4m3(
    const __nv_bfloat16 (&vals)[kPerThread],
    uint8_t             (&out_fp8)[kPerThread],
    uint8_t&             ue8m0_byte) {
    float fvals[kPerThread];
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
        fvals[i] = __bfloat162float(vals[i]);
    }
    quantize_block_fp32_to_e4m3<kPerThread, kWidth>(fvals, out_fp8, ue8m0_byte);
}

// ---- SMEM swizzle (inline because it has no PTX, just integer math) -------

// XOR-swizzle for a row-major SMEM tile staged at granularity `kRowBytes`
// bytes per row. Inputs are (row, col_byte) where col_byte is the byte offset
// within the row. Returns the swizzled byte offset within the tile.
//
// The swizzle XORs bits 4..6 of the within-128B-block column offset with the
// low 3 bits of the row index, so that adjacent rows accessing the same
// 16-byte fragment column hit distinct 4-byte banks. With 32 banks per row
// (128B / 4B), this is the canonical bank-conflict-free pattern for
// ldmatrix.x4 16-byte fragment loads.
//
// Usage convention: `kRowBytes` should be a multiple of 128. For non-power-of
// -2 row strides (e.g., head_dim=448 fp8 + 64 bf16 = 576 bytes), pad to 640
// at the layout level and use the padded row stride here.
template <int kRowBytes>
__device__ __forceinline__ uint32_t smem_xor_swizzle(uint32_t row, uint32_t col_byte) {
    static_assert((kRowBytes % 128) == 0,
                  "smem_xor_swizzle requires row stride padded to 128B");
    const uint32_t row_off    = row * kRowBytes;
    const uint32_t block_base = col_byte & ~static_cast<uint32_t>(0x7f);
    const uint32_t in_block   = col_byte &  static_cast<uint32_t>(0x7f);
    const uint32_t swizzled   = in_block ^ ((row & 0x7u) << 4);
    return row_off + block_base + swizzled;
}

// Plain row-major addressing (used when bank-conflict-free access is not
// needed, e.g. cooperative loads written by individual threads).
template <int kRowBytes>
__device__ __forceinline__ uint32_t smem_row_major(uint32_t row, uint32_t col_byte) {
    return row * kRowBytes + col_byte;
}

}  // namespace sm120_native_fp8
}  // namespace deep_gemm
