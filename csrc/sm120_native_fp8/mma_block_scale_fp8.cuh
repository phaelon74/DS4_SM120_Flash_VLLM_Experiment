// SM120 native FP8 block-scaled MMA wrappers.
//
// This header is the *only* place where SM120 PTX MMA instructions live for
// the v2 native FP8 path. All testing-blind PTX risk is concentrated here so
// nvcc/ptxas errors point at one well-marked file.
//
// Targets:
//   * FP8 block-scaled m16n8k32 (DeepSeek V4 sparse MLA, dim < 448 portion):
//       mma.sync.aligned.m16n8k32.kind::f8f6f4.block_scale.scale_vec::1X
//                       .f32.e4m3.e4m3.f32
//     This consumes E4M3 A/B with a UE8M0 scale per K=32 block. DSv4's cache
//     uses UE8M0 with quantization block = 64, so the kernel feeds the same
//     scale byte to two consecutive K=32 MMA tiles -- handled at the call
//     site, not here.
//
//   * BF16 m16n8k16 (RoPE tail dim >= 448, plus bf16-debug-mode):
//       mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32
//     This is the verified PR #1 pattern. Used for the BF16 RoPE contribution
//     to QK^T and to P*V in both FP8 and BF16 modes, and used for the entire
//     QK^T / P*V when DG_SM120_FUSED_DECODE_V2_FP8MMA=0.
//
// Compile target:
//   * Requires nvcc with `-gencode=arch=compute_120f,code=sm_120f` (CUDA 13+).
//   * The `kind::f8f6f4.block_scale` PTX is gated on __CUDA_ARCH__ >= 1200 and
//     compiled out otherwise (so this header is benign on older GPUs).
//
// Register-layout conventions (m16n8k32, fp8):
//   * A is 16 rows x 32 K (4 fp8 per lane, packed in 1 u32) -> 4 u32 regs/lane
//   * B is 32 K x  8 cols (4 fp8 per lane, packed in 1 u32) -> 2 u32 regs/lane
//   * C/D are 16 rows x  8 cols, fp32 -> 4 fp32 regs/lane
//
// Scale-operand layout (scale_vec::1X, K=32 block, M=16 / N=8):
//   * For A: 16 scales total (one per row), distributed across the 32 lanes.
//     Per the PTX 8.7 spec, lanes 0-15 each provide 1 scale byte; the byte
//     position within the lane's u32 is selected by a per-instruction
//     immediate `byte_id_a`. The wrapper here always packs a single scale byte
//     into byte 0 of the u32 (so byte_id_a is hard-coded to 0). Lanes 16-31
//     pass don't-care zeros. NOTE: this is the layout I am running with
//     testing-blind. If ptxas complains about "operand byte_id" or "scale
//     vector size", this is the first thing to revisit. // TODO(VERIFY-PTX)
//
//   * For B: 8 scales total (one per col), distributed across lanes 0-7. Same
//     pack-in-byte-0 convention, byte_id_b hard-coded to 0. // TODO(VERIFY-PTX)
//
// Numerical conventions:
//   * UE8M0 byte 0 means "scale = 0"; we follow PR #1 and pre-clamp this to
//     UE8M0 byte 1 (== 2^-126) at quantization time so MMA never sees a zero
//     scale. This is also where the v2 scalar path's `decode_ue8m0_scale`
//     returned 0.0f for byte 0, but block-scaled MMA does not support a
//     zero-scale sentinel, so we must clamp at quant time.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace deep_gemm {
namespace sm120_native_fp8 {

// ---- Architecture gate -----------------------------------------------------

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
#define DG_SM120_NATIVE_FP8_HAS_F8MMA 1
#else
#define DG_SM120_NATIVE_FP8_HAS_F8MMA 0
#endif

// ---- m16n8k16 BF16 MMA (verified PR #1 pattern) ---------------------------

// Compute D = A * B + C with:
//   * A: bf16, 16 rows x 16 K, 8 bf16 per lane = 4 b32 packed regs
//   * B: bf16, 16 K x  8 cols, 4 bf16 per lane = 2 b32 packed regs
//   * C/D: fp32, 16 rows x 8 cols, 4 fp32 per lane
//
// `a` packs two bf16 values per uint32_t (low / high halves), 4 uint32_t per
// lane, total 16 row-K halfwords per lane.
// `b` packs the same way, 2 uint32_t per lane.
__device__ __forceinline__ void mma_bf16_m16n8k16(
    float       (&d)[4],
    const uint32_t a[4],
    const uint32_t b[2],
    const float  c[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    // Pre-Ampere: no bf16 MMA. Provide a scalar fallback so the wrapper is
    // compilable for host-only TUs. Should never execute on real targets.
    #pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = c[i];
#endif
}

// ---- m16n8k32 FP8 block-scaled MMA (testing-blind path) --------------------

// Compute D = A * B + C with:
//   * A: e4m3, 16 rows x 32 K, 4 fp8 per lane = 4 b32 packed regs
//   * B: e4m3, 32 K x  8 cols, 4 fp8 per lane = 2 b32 packed regs
//   * C/D: fp32, 16 rows x 8 cols, 4 fp32 per lane
//   * scale_a: u32 packing 4 UE8M0 bytes (only byte 0 used here)
//   * scale_b: u32 packing 4 UE8M0 bytes (only byte 0 used here)
//
// scale_a is broadcast across lanes 0-15 (one scale byte per row of A) and
// scale_b across lanes 0-7 (one scale byte per col of B). Lanes outside that
// range contribute zero. The caller is responsible for routing the per-row /
// per-col scale into the correct lane's `scale_a` / `scale_b` register.
//
// // TODO(VERIFY-PTX): the exact PTX form of the block_scale variant. PTX
// 8.7 syntax (CUDA 12.8+) is:
//   mma.sync.aligned.m16n8k32.kind::f8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32
//     {d}, {a}, {b}, {c}, scale_a, scale_b, byte_id_a, byte_id_b;
// where byte_id_a/byte_id_b are 2-bit immediates selecting which byte of the
// u32 scale registers. We hard-code byte 0 (immediate 0) below. If ptxas
// rejects the "byte_id" operand naming or the "scale_vec::1X" qualifier, the
// alternates to try are:
//   (a) scale_vec::1X -> scale_vec::2X
//   (b) drop byte_id immediates (some PTX revisions encode them differently)
//   (c) prefix with "kind::f8f6f4" before "block_scale"
__device__ __forceinline__ void mma_e4m3_block_scale_m16n8k32(
    float       (&d)[4],
    const uint32_t a[4],
    const uint32_t b[2],
    const float  c[4],
    uint32_t     scale_a,
    uint32_t     scale_b) {
#if DG_SM120_NATIVE_FP8_HAS_F8MMA
    asm volatile(
        "mma.sync.aligned.m16n8k32.kind::f8f6f4.block_scale.scale_vec::1X."
        "f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "%14, %15, 0, 0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(scale_a), "r"(scale_b));
#else
    // Compile-time fallback for non-SM120 builds: behave like a no-op MMA.
    // Should never execute at runtime on supported hardware.
    #pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = c[i];
    (void)a; (void)b; (void)scale_a; (void)scale_b;
#endif
}

// ---- ldmatrix wrappers ----------------------------------------------------

// ldmatrix.x4.b16 from SMEM into 4 uint32_t fragments. Used to load BF16
// (m16k16 A operand) or BF16 RoPE B operand. Address is per-lane SMEM byte
// pointer; the PTX selects rows by lane id.
__device__ __forceinline__ void ldmatrix_x4_b16(
    uint32_t (&out)[4], const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(addr));
}

// ldmatrix.x2.b16 (used for the bf16 B operand of m16n8k16).
__device__ __forceinline__ void ldmatrix_x2_b16(
    uint32_t (&out)[2], const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared::cta.b16 {%0, %1}, [%2];\n"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(addr));
}

// ldmatrix.x4 for FP8 fragments. PTX has no native b8 form; we reuse the
// b16 variant and reinterpret the loaded data as packed FP8 (4 fp8 per b32
// quarter, which matches what the m16n8k32 MMA expects).
//
// Caller responsibility: smem layout must be such that 8 consecutive 16-bit
// halfwords (= 16 consecutive bytes = 16 fp8 values) sit contiguous in SMEM
// at the address each lane provides, and the standard ldmatrix lane->row
// mapping deposits them into the K-axis of the fragment.
__device__ __forceinline__ void ldmatrix_x4_fp8_as_b16(
    uint32_t (&out)[4], const void* smem_ptr) {
    ldmatrix_x4_b16(out, smem_ptr);
}

__device__ __forceinline__ void ldmatrix_x2_fp8_as_b16(
    uint32_t (&out)[2], const void* smem_ptr) {
    ldmatrix_x2_b16(out, smem_ptr);
}

// .trans variants reinterpreted for FP8: same b16 PTX, caller treats output
// as packed FP8 (4 fp8 per b32). Used when the SMEM layout has the source
// matrix transposed relative to the MMA fragment's expected K-major layout.
__device__ __forceinline__ void ldmatrix_x4_trans_fp8_as_b16(
    uint32_t (&out)[4], const void* smem_ptr) {
    ldmatrix_x4_trans_b16(out, smem_ptr);
}

__device__ __forceinline__ void ldmatrix_x2_trans_fp8_as_b16(
    uint32_t (&out)[2], const void* smem_ptr) {
    ldmatrix_x2_trans_b16(out, smem_ptr);
}

// ldmatrix.trans variants (B operand for the fp8 m16n8k32 mma is K-major in
// SMEM but needs K-major-fragment loads; A operand is row-major in SMEM and
// needs row-major loads; trans is selected by caller).
__device__ __forceinline__ void ldmatrix_x4_trans_b16(
    uint32_t (&out)[4], const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans_b16(
    uint32_t (&out)[2], const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared::cta.b16 {%0, %1}, [%2];\n"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(addr));
}

}  // namespace sm120_native_fp8
}  // namespace deep_gemm
