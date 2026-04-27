# SM120 Fused Kernels — Architecture and Status

This document describes the **production v2** SM120 kernel paths that were
added to replace the duct-tape scalar / workspace-bridge implementations in
this fork. It captures the design, current state, and the explicit MMA-upgrade
work that remains.

---

## Why these kernels exist

Per the `sm120_v4flash_100tps` plan and AGENTS.md, the dominant single-request
bottlenecks on SM120 (RTX PRO 6000, 300 W cap) are:

1. **Sparse MLA prefill** — long-prompt TTFT collapses because the prior
   bridge materializes a BF16 workspace and runs `torch.bmm`/softmax/`torch.bmm`
   per chunk. The bridge is tensor-core via cuBLAS but is **not fused sparse
   attention** and pays per-chunk Python/framework launch costs.
2. **Sparse MLA decode** — every step of MTP/c1 decoding goes through the
   gather + workspace-split bridge. Decode improvement compounds across all
   tokens.
3. **HyperConnection prenorm** — runs every layer; the original SM120 path was
   a CUDA-core scalar fallback, not a tensor-core kernel.

Helper-kernel tweaks of 1-3 us each cannot close the remaining ~16 us/layer
gap to 100 tok/s. Only **larger-fusion / fused tensor-core sparse attention**
plus a **proper HC prenorm GEMM** can.

---

## Kernels delivered

### 1. `csrc/sm120_tf32_hc_prenorm_gemm.cu`  *(Phase 1.1, default-on)*

Replaces the scalar `csrc/sm120_hc_prenorm_fallback.cu` for any HC prenorm
call with `n <= 64`.

- **Output contract:** identical to the fallback (A:[M,K] BF16 K-major,
  B:[N,K] FP32 K-major, D:[num_splits,M,N] FP32, S:[num_splits,M] FP32).
- **Numerics:** A loaded as FP32 from BF16, B rounded to TF32, FP32 accumulate.
- **Layout:** one block per (split, m_row), 256 threads, partials reduced
  through SMEM as [n][threads]. Tiles K in fixed 64-element chunks for
  unrolled inner loop and keeps partials in registers.
- **Gating:** `DG_SM120_HC_PRENORM_V2=1` (default ON; safe to leave on across
  any DeepSeek V4 Flash workload). Falls back to the scalar implementation if
  N > 64 or the env var is set to "0".
- **MMA upgrade hook:** `// TODO(SM120-MMA):` markers indicate where to slot
  in `mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32` warp-level instructions.

### 2. `csrc/sm120_sparse_mla_decode_v2.cu`  *(Phase 2, opt-in v2)*

Single-CTA fused sparse MLA decode that reads directly from the `fp8_ds_mla`
KV cache, computes Q*K^T / online softmax / P*V, and emits BF16 output + FP32
LSE in one launch.

- **Cache layout (unchanged):** `head_dim = 512` split as 448 FP8 (E4M3) +
  64 BF16 RoPE tail; UE8M0 scales packed per-64.
- **Layout:** one CTA per (batch, head); `kThreads = 256`. Each thread owns 2
  dims of the head_dim axis. Online softmax (FA2-style) keeps `(m, l)` in
  registers; `out_acc` per-thread accumulator gets rescaled across chunks.
- **Q*K^T:** 32 candidates per chunk, 8 lanes per candidate, 64 dims per
  sublane. Currently scalar FP32 inner loop; will be replaced with
  `mma.sync.aligned.m16n8k32.kind::f8f6f4.block_scale` once validated on
  hardware.
- **P*V:** scalar accumulate per dim; same MMA upgrade target.
- **Indices contract:** `[B, 1, K]` int32/int64; `-1` rows are masked at
  score level via `-INFINITY`, no safe-clamped gather required.
- **Optional `attn_sink`:** folded into the first chunk's max/sum; matches the
  original DeepSeek-V4 Flash sink semantics.
- **Returns** `(out, lse)` where lse has shape `[B, H]`.
- **Gating:** `DG_SM120_FUSED_DECODE_V2=1` (default OFF until live
  validation). Patcher skips the v2 path for two-cache callers (i.e. when
  `extra_k_cache is not None`); those still go through the workspace path.
  `DG_SM120_FUSED_DECODE_V2_STRICT=1` re-raises errors instead of falling back.
- **API:** `deep_gemm._C.sm120_sparse_mla_decode_v2(q, cache, indices,
  topk_length, attn_sink, head_dim_v, softmax_scale, block_size, out)`.

### 3. `csrc/sm120_sparse_mla_prefill_v2.cu`  *(Phase 3, opt-in v2)*

Single-CTA fused sparse MLA prefill that reads directly from the FP8 ds_mla
KV cache via a `workspace_map` (one int32 per workspace row giving the
physical KV slot). Mirrors decode v2's structure so they share a future MMA
upgrade.

- **Workspace_map contract:** `[N_workspace]` int32; entries are physical
  cache linear-slot indices, or -1 for invalid (mapped/unmapped row).
- **Indices contract:** `[S, 1, K]` int32/int64; entries reference
  workspace_map rows. -1 entries mask at score level.
- **Returns** `(out, max_logits, lse)`. `max_logits` is filled with NaN to
  signal it tracks `row_max` internally; the LSE is fp32.
- **Gating:** `DG_SM120_FUSED_PREFILL_V2=1` (default OFF).
  `DG_SM120_FUSED_PREFILL_V2_STRICT=1` re-raises errors. Only used in the
  patcher when the chunk has no SWA half (compressed-only chunks); two-cache
  chunks fall through to the existing
  `sm120_sparse_mla_prefill_from_two_fp8_workspace_map` helper.
- **API:** `deep_gemm._C.sm120_sparse_mla_prefill_v2(q, cache, workspace_map,
  indices, topk_length, attn_sink, block_size, head_dim_v, softmax_scale,
  out)`.

---

## What is NOT yet replaced

These remain as scaffolding (`fallback`, `_legacy`, or workspace bridges) and
will be retired only after Phases 4-5 of the plan complete:

- `csrc/sm120_sparse_mla_decode.cu` (workspace + split decode) — still the
  default while v2 is opt-in.
- `csrc/sm120_hc_prenorm_fallback.cu` — still compiled and reachable for the
  N > 64 edge case (and as a `DG_SM120_HC_PRENORM_V2=0` escape hatch).
- `sparse_mla_prefill_from_bf16_workspace_split` — still the default prefill
  bridge while v2 is opt-in.
- BF16 workspace gather / `torch.bmm` bridge in
  `docker/patch_vllm_deepseekv4.py` — still used for two-cache chunks.

---

## MMA-upgrade roadmap (the part that closes the gap)

Both v2 fused kernels are **structurally fused** today: single launch, no
intermediate workspace, online softmax in registers. The inner Q*K^T and P*V
loops are **scalar fp32**. The plan's headline wins come from upgrading those
inner loops to warp-level `mma.sync` instructions.

The exact replacement patterns are marked in source:

```cpp
// TODO(SM120-MMA): replace the inner loop with a warp-level
// ``mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32`` block-scaled MMA on the
// FP8 portion (dim < 448), then add the BF16 RoPE tail with a separate
// ``mma.sync.kind::f16`` instruction. This is the path that turns this
// kernel from "structurally fused" to "tensor-core fused".
```

Reference substrate: [gau-nernst/learn-cuda
09a_block_scaled_mm_sm120](https://github.com/gau-nernst/learn-cuda/tree/main/09a_block_scaled_mm_sm120).
The block-scaled GEMM kernel there is the model for adapting MMA to a sparse
attention shape (gather Q row, gather K rows by index, MMA with scales).

### Scale-layout note

DeepSeek V4 Flash uses 64-wide UE8M0 scales. `mma.sync.kind::mxf8f6f4
scale_vec::1X` is 32-wide; `scale_vec::2X` is 64-wide. Two valid options:

1. **Native:** use `m16n8k64 scale_vec::2X` to match V4's 64-wide scales
   directly inside MMA.
2. **Out-of-MMA:** apply scale multiply outside MMA (gau-nernst v2 pattern).

The v2 prototypes apply scale **outside MMA** via per-quant-block decoded
multipliers. Once the MMA upgrade lands, prefer option (1) for natural fit.

---

## Validation strategy

The repo contains synthetic correctness harnesses for each kernel:

| Kernel              | Test script                               | Reference             |
| ------------------- | ----------------------------------------- | --------------------- |
| HC prenorm v2       | `scripts/bench_sm120_hc_prenorm.py`       | scalar fallback       |
| Decode v2           | `scripts/test_sm120_fused_decode.py`      | gather + split decode |
| Prefill v2          | `scripts/test_sm120_fused_prefill.py`     | gather + split prefill|

There is also a regression gate `scripts/regression_gate.sh` that asserts:
- c1 256-token streaming steady decode  >= 85 tok/s  (default)
- c4 256-token streaming aggregate      >= 175 tok/s
- 4k-token TTFT                         <= 2.5 s
- 8k-token TTFT                         <= 6.0 s
- 16k-token TTFT                        <= 18.0 s

These thresholds are the **Phase 0 baseline floors**, not the 100 tok/s
target. They tighten as later phases land.

---

## Feature-flag matrix

| Flag                                     | Default | What it does                                    |
| ---------------------------------------- | ------- | ----------------------------------------------- |
| `DG_SM120_HC_PRENORM_V2`                 | 1       | Use tiled HC prenorm v2                         |
| `DG_SM120_FUSED_DECODE_V2`               | 0       | Use fused decode v2 (single-cache only)         |
| `DG_SM120_FUSED_DECODE_V2_STRICT`        | 0       | Re-raise on v2 decode error instead of fallback |
| `DG_SM120_FUSED_PREFILL_V2`              | 0       | Use fused prefill v2 (compressed-only chunks)   |
| `DG_SM120_FUSED_PREFILL_V2_STRICT`       | 0       | Re-raise on v2 prefill error                    |

The "v2" flags are intentionally opt-in until live validation on real DeepSeek
V4 Flash traffic. After the MMA upgrade and stress-test pass, they will flip
to default-on and the scalar/workspace paths will be archived as `_legacy`.
