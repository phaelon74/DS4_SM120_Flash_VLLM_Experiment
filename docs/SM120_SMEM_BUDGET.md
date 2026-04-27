# SM120 SMEM Budget Audit

This is the Phase 1.2 SMEM-budget audit. Every `csrc/sm120_*.cu` file is listed
here with an explicit (tile + barrier + epilogue + overhead) <= 99 KB
derivation. SM120 has **99 KB shared memory per block**, vs 227 KB on SM100.

The numbers below are upper bounds computed from the kernel sources. They are
intentionally conservative — when the kernel shapes change, recompute and
update this table.

---

## Hardware envelope

| Feature                | SM120                     | SM100 (for comparison)    |
| ---------------------- | ------------------------- | ------------------------- |
| SMEM per block         | 99 KB                     | 227 KB                    |
| TMEM                   | None                      | 256 KB                    |
| Tensor-core path       | Warp `mma.sync.aligned`   | `tcgen05.mma` UMMA        |
| Cluster                | Forced 1x1x1              | Up to 2x4 with multicast  |
| TMA multicast          | None                      | Yes                       |
| Block-scale instr.     | `kind::{f8f6f4,mxf*}.bs`  | UMMA blockscaled          |

The 99 KB cap is the hard constraint behind every tile/stage decision on this
fork.

---

## Kernel-by-kernel audit

### 1. `csrc/sm120_fp8_fp8_cutlass.cu`

Dense FP8xFP8 GEMM via CUTLASS 3.x. Drives DeepSeek V4 dense projections.

| Component           | Source                                       | Bytes       |
| ------------------- | -------------------------------------------- | ----------- |
| Tile A (E4M3)       | M=128, K=128 = 16 KB                         | 16384       |
| Tile B (E4M3)       | N=128, K=128 = 16 KB                         | 16384       |
| Tile A scales (UE8M0) | (M/128)*(K/32) ~= 4 entries per stage      |   128       |
| Tile B scales (UE8M0) | (N/128)*(K/32) ~= 4 entries per stage      |   128       |
| Epilogue (BF16)     | M*N*sizeof(bf16) shared epilogue tile        | 32768       |
| Barriers / mbarrier | ~32 bytes per stage * stages                 |   192       |
| Pipeline (StageCountAutoCarveout) | computed by CUTLASS to fit       |  -          |

For the K=128 path, CUTLASS' `StageCountAutoCarveout` arithmetic computes
**stages = floor((99 KB - epilogue - overhead) / per_stage_bytes)**.
With 32-33 KB per stage (A+B+scales) and ~36 KB epilogue, that lands at
**2 stages** — typical for SM120 dense. PR #3121 (K=64 blockscaled fix) would
unlock 7-25 stages at K=64; the optional K=64 tile variant in
`docs/CUTLASS_REBASE.md` is gated behind that rebase.

Total resident:  ~33 KB tile + ~36 KB epilogue + ~1 KB scaling/overhead
            ~= **70 KB / 99 KB** budgeted, with stages set by CUTLASS auto.

### 2. `csrc/sm120_fp8_fp4_cutlass.cu`

MoE FP8 (activation) x FP4 (packed expert weights) grouped GEMM via CUTLASS
4.x. Uses `KernelPtrArrayTmaWarpSpecializedPingpong` for grouped problems.

| Component           | Source                                       | Bytes       |
| ------------------- | -------------------------------------------- | ----------- |
| Tile A (E4M3)       | M_tile * K_tile (depends on expert tile)     | 16384       |
| Tile B (FP4 packed) | N_tile * K_tile / 2 (4-bit packed)           |  8192       |
| Block-scale buffers | ~256 bytes per stage * 2                     |   512       |
| Group ptr metadata  | per-group base/stride/scale ptrs             |  4096       |
| Epilogue            | shared BF16 epilogue tile                    | 32768       |
| Pingpong barriers   | mbarrier + warp-spec state                   |   512       |

Total: ~62 KB resident, with **2 stages pingpong**. NVFP4 doubles the
effective stages-vs-FP8 because B is 4-bit, freeing ~half the per-stage cost.
The `DG_SM120_MOE_*` env knobs control compact / row-grouped / direct-groups
layout but do not change this total.

### 3. `csrc/sm120_fp8_gemm_fallback.cu`

Fallback FP8xFP8 hand-written kernels for shapes CUTLASS rejects (notably the
BHR `bhr,hdr->bhd` einsum). Two modes: warp-column and M1-MMA.

| Component (warp-column path) | Source                                     | Bytes  |
| ----------------------------- | ------------------------------------------ | ------ |
| A row staging                 | per-warp BF16 row, ~1 KB per warp          |  4096  |
| B column tile                 | per-warp E4M3 col tile, ~2 KB per warp     |  8192  |
| Output accumulators           | held in registers                          |     0  |
| Reduction SMEM                | per-warp partial buffer                    |  1024  |

Total: ~13 KB resident, single-stage. Fits with margin.

| Component (M1-MMA path)       | Source                                     | Bytes  |
| ----------------------------- | ------------------------------------------ | ------ |
| A row staging (single row)    | bf16 K-vector                              |  4096  |
| Block scales                  | UE8M0 per K=32 group                       |   256  |
| B tile (e4m3)                 | one warp's column slab                     |  4096  |

Total: ~9 KB resident. Gated to R<=1024 by AGENTS.md heuristic.

### 4. `csrc/sm120_mqa_logits_fallback.cu`

MQA logits scalar fallback (used pre-DG_SM120_*). Per-CTA SMEM is just
register spill area + warp reduction scratch — under 4 KB.

### 5. `csrc/sm120_hc_prenorm_fallback.cu`  *(legacy scalar fallback)*

Per-block: `(n + 1) * threads * 4 B`. With kThreads=256 and n<=64,
**that is (64+1) * 256 * 4 = 66.5 KB**. This was the original SM120 scalar
fallback; the v2 unified kernel below uses the same SMEM layout. Both fit
under the 99 KB cap with room for the launch-bounds residency target of 2
CTAs/SM (which would need <=49 KB; we deliberately use 1 CTA/SM here).

### 6. `csrc/sm120_tf32_hc_prenorm_gemm.cu`  *(v2 production HC prenorm)*

Mirrors the fallback's layout for bit-exactness; same SMEM budget.

| Component               | Bytes (n<=64, threads=256)             |
| ----------------------- | -------------------------------------- |
| Partials [n][threads]   | 64 * 256 * 4 = **64 KB**               |
| sq partial [threads]    | 256 * 4 = 1 KB                         |
| Total                   | **65 KB**, 1 CTA/SM                    |

Resident: 65 KB / 99 KB. Headroom for n increase to 128 would push the
partials buffer to 128 KB — over budget; the kernel asserts n <= 64.

### 7. `csrc/sm120_sparse_mla_decode.cu`  *(v1 scalar workspace decode)*

| Component               | Bytes                                  |
| ----------------------- | -------------------------------------- |
| Q in SMEM (head_dim=512)| 512 * 4 = 2048                         |
| Per-warp partial        | 32 * 4 = 128 per warp * 8 warps = 1024 |
| Total                   | ~3 KB — well under cap                  |

This kernel is bandwidth/compute bound, not SMEM bound. Replaced by v2.

### 8. `csrc/sm120_sparse_mla_decode_v2.cu`  *(NEW — Phase 2 fused decode)*

| Component                | Source                                 | Bytes |
| ------------------------ | -------------------------------------- | ----- |
| Q in SMEM (fp32, 512 dim)| `kHeadDim * sizeof(float)` = 2048      |  2048 |
| Warp-reduce scratch      | `8 * sizeof(float)`                    |    32 |
| chunk_logits             | 32 floats                              |   128 |
| chunk_linear             | 32 int64                               |   256 |
| chunk_scales             | 32 * 8 = 256                           |   256 |
| Total dynamic SMEM       | (`kHeadDim + 8) * sizeof(float)`       |  2080 |
| Total static SMEM        | logits/linear/scales                   |   640 |

Total resident: **~2.7 KB / 99 KB**. With `__launch_bounds__(256, 2)` we get 2
CTAs/SM, well within budget. The MMA upgrade (TODO(SM120-MMA): in source) will
add register pressure but the SMEM footprint stays bounded by the chunk size
and Q tile.

### 9. `csrc/sm120_sparse_mla_prefill_v2.cu`  *(NEW — Phase 3 fused prefill)*

Identical SMEM layout to decode v2 — one CTA per (sequence-token, head):

| Component       | Bytes |
| --------------- | ----- |
| Q tile (fp32)   |  2048 |
| chunk_logits    |   128 |
| chunk_linear    |   256 |
| chunk_scales    |   256 |
| warp_red        |    32 |
| Total           | ~2.7 KB |

Total resident: **~2.7 KB / 99 KB**. Headroom for future MMA upgrade.

### 10. `csrc/sm120_moe_activation_quant.cu`

Fused SiLU + FP8 activation quant helper. Per-block: small bf16 staging
buffer for groups of activations + UE8M0 scale per quant block. Empirically
~6 KB. Well under cap.

### 11. `csrc/sm120_metadata.cu`

Graph-native metadata builders (compressor metadata, sparse-SWA metadata,
FlashMLA sparse req_id). All trivially under 4 KB SMEM each.

---

## Aggregate

| Kernel                                  | SMEM (resident)  | Status        |
| --------------------------------------- | ---------------- | ------------- |
| sm120_fp8_fp8_cutlass                   | ~70 KB           | Production    |
| sm120_fp8_fp4_cutlass                   | ~62 KB           | Production    |
| sm120_fp8_gemm_fallback (warp-col)      | ~13 KB           | Fallback      |
| sm120_fp8_gemm_fallback (m1-mma)        | ~9 KB            | Fallback      |
| sm120_mqa_logits_fallback               | ~4 KB            | Fallback      |
| sm120_hc_prenorm_fallback               | ~65 KB (legacy)  | Legacy        |
| sm120_tf32_hc_prenorm_gemm   (Phase 1)  | ~65 KB           | Default-on    |
| sm120_sparse_mla_decode      (v1)       | ~3 KB            | Workspace     |
| sm120_sparse_mla_decode_v2   (Phase 2)  | ~2.7 KB          | Opt-in v2     |
| sm120_sparse_mla_prefill_v2  (Phase 3)  | ~2.7 KB          | Opt-in v2     |
| sm120_moe_activation_quant              | ~6 KB            | Default-on    |
| sm120_metadata                          | ~4 KB each       | Default-on    |

The two v2 sparse-MLA kernels intentionally have a small SMEM footprint to
leave headroom for the MMA upgrade pass. When the MMA path lands, a typical
warp-spec tile (Q tile 32 KB + K-ring 16 KB/stage * 3 stages + barriers 1 KB)
stays at ~80 KB / 99 KB — fits with margin.

## Acceptance

The audit is recorded here and referenced from AGENTS.md.
No behavior change implied: every kernel is already within budget.
