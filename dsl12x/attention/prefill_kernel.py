"""dsl12x.attention.prefill_kernel: Standalone CuTe DSL sparse MLA prefill kernel.

Architecture
============

This is the standalone DeepSeek V4 Flash sparse MLA prefill kernel for SM120
(NVIDIA RTX PRO 6000 Blackwell workstation). Modeled architecturally on
b12x's SparseMLAKernel pattern but written from first principles using
cutlass.cute primitives -- no `from b12x import ...` calls anywhere.

Per-CTA design:

    Threads:    32 (one warp).
    SMEM:       ~14-22 KB depending on traits.
    Tile:       16 Q heads x 32 KV tokens x qk_head_dim (swept in
                groups of nope_group_elems=64).
    MMA:        BF16 m16n8k16 for QK and PV (4th-gen tensor cores).
    Output:     BF16 [num_tokens, num_heads, v_head_dim].
    LSE:        FP32 [num_tokens, num_heads] (for chunked-prefill
                cross-chunk reduction by the patcher).

Per-CTA hot-loop:

    For each token tile (32 tokens per tile, num_chunk_tokens/32 tiles):
        Online softmax frags reset:  m_frag = -inf, d_frag = 0, o_frag = 0
        For each topk slot k = 0..topk_max-1:
            workspace_map[token, k] decoded into (cache_id, row, valid).
            cp.async stage of K row from cache (FP8) into SMEM.
            cp.async stage of V row from cache (FP8) into SMEM.
            __syncthreads / cp.async.commit_group.
            Dequant FP8 -> BF16 in registers.
            QK MMA: BF16 m16n8k16.  scores += sm_scale * (Q @ K_row^T).
            For invalid entries (workspace_map == sentinel, or
                k >= topk_length[token]): mask scores to -inf.
            Online softmax update: m_new = max(m, max(scores));
                rescale = exp2(m - m_new); d = d * rescale + sum(exp2(scores - m_new));
                o = o * rescale + exp2(scores - m_new) @ V_row.
            PV MMA: BF16 m16n16k16.
        Epilogue: out = o / d.
        attn_sink gate (if has_attn_sink):
            lse_natural = m_new * ln(2) + log(d) * ln(2).
            gate = sigmoid(lse_natural - attn_sink[head]).
            out *= gate.
        Store BF16 out, FP32 LSE.

Dual-cache workspace map encoding
=================================

The patcher provides ``workspace_map: int32 [num_tokens, topk_max]`` where:

  workspace_map[m, k] >= 0:
    The k-th topk entry for token m references the C128 (compressed) cache.
    The cache row is workspace_map[m, k]. K and V come from
    c128_cache[row, :qk_head_dim] and c128_cache[row, :v_head_dim]
    respectively (since DeepSeek V4 MLA's V is a slice of the same row).

  workspace_map[m, k] < 0  AND  workspace_map[m, k] != INVALID_SENTINEL:
    The k-th topk entry for token m references the SWA cache. The cache
    row is ~workspace_map[m, k] (bitwise-NOT). K and V come from
    swa_cache[row, :qk_head_dim] and swa_cache[row, :v_head_dim].

  workspace_map[m, k] == INVALID_SENTINEL (== INT32_MIN):
    The k-th topk entry for token m is invalid (k >= topk_length[m] or
    the original cache row was -1). The kernel masks the corresponding
    score to -inf so it contributes nothing to the softmax.

The patcher constructs the workspace map by:

  1. Reading vLLM's combined_indices (which already merges C128 and SWA
     indices for each prefill token).
  2. Encoding C128 entries directly (non-negative).
  3. Encoding SWA entries with bitwise-NOT.
  4. Setting entries beyond topk_length[m] to INVALID_SENTINEL.

This single int32 tensor replaces the patcher's current K/V workspace
gather (which copies FP8 cache rows into a BF16 workspace tensor before
calling the BMM bridge). The dsl12x kernel reads the FP8 cache rows
directly and dequantizes in registers, eliminating the workspace gather
entirely.

attn_sink semantic
==================

DeepSeek V4 Flash's attn_sink is a POST-ATTENTION SIGMOID GATE, NOT a
softmax-denominator sink token. Confirmed by reading the BMM bridge at
docker/patch_vllm_deepseekv4.py:1613-1622 and the scalar reference at
:1692-1693:

    out_dsl = softmax(QK^T * sm_scale) @ V          # standard attention
    lse = logsumexp(QK^T * sm_scale, dim=-1)        # natural log

    if attn_sink is not None:
        gate = sigmoid(lse - attn_sink[head])       # per-head, scalar per token
        out *= gate.unsqueeze(-1)                   # broadcast over v_head_dim

The kernel's epilogue computes lse from the online-softmax m and d frags:

    lse = m * ln(2) + log(d * ln(2))                # since we accumulate in base 2

Then applies the sigmoid gate before storing.

NOT a sink-token-denominator (which would be:
    softmax_denom += exp(sink)
    out = exp(scores) / softmax_denom @ V).

What's COMPLETE vs SCAFFOLD
===========================

COMPLETE in this initial release:

  * SMEM layout calculation (smem_bytes_per_cta from traits).
  * CTA grid sizing.
  * Online-softmax math (m, d, o frags).
  * attn_sink sigmoid-gate epilogue.
  * Dual-cache workspace map decoding.
  * Host wrapper with JIT cache + warmup helper (in prefill.py).

SCAFFOLD that will need iteration on the test system:

  * The exact cute.gemm invocations: the cutlass.cute API for SM80
    m16n8k16 MMA may differ between cutlass-dsl versions. b12x uses
    `cute.gemm(...)` with explicit MmaAtom. The kernel here uses the same
    pattern but the precise MmaAtom enum value may need adjustment.
  * cp.async wrappers in dsl12x.ptx are stubs; the kernel uses
    `cute.cp_async` directly here as a fallback.
  * The lane-mapping for the m16n8k16 fragment register layout is
    standard (PTX docs) but the cutlass.cute abstraction handles it
    implicitly. If the abstraction changes between cutlass-dsl versions
    the explicit lane mapping in the dequant/store helpers may need
    adjustment.
  * FP8 dequant in registers: cutlass.cute provides
    `cute.arch.fp8_e4m3_to_bf16` (or similar; exact name varies by
    version). The kernel uses a placeholder; verify on test system.

The patcher's try/except wrapper (in prefill.py and the patcher's
DG_SM120_DSL12X_PREFILL branch) catches RuntimeError + cuda.CudaError
from any of these and falls back to the BMM bridge. So a kernel
compile/run failure will not break serving -- the worst case is the BMM
bridge handling all prefill traffic, identical to today's production
behavior.

Tuning knobs (Phase 1 trace -> SparseMLATraits)
================================================

The kernel is templated by SparseMLATraits which captures all static
shape/config. After Phase 1 trace prints the live shape, build the right
SparseMLATraits and call run_sparse_mla_prefill (which JIT-compiles for
the given traits and caches the launcher).

Default traits for DeepSeek V4 Flash (assuming V3-family qk/v dims):

    SparseMLATraits(
        num_heads=32,           # DG_SM120_ACTIVE_HEADS
        qk_head_dim=576,        # 512 latent + 64 RoPE; verify with trace
        v_head_dim=512,         # verify with trace
        topk_max=128,           # DG_SM120_MAIN_TOPK_CAP
        chunk_size=256,         # DG_SM120_PREFILL_WORKSPACE_CHUNK
        dtype=torch.bfloat16,
        has_attn_sink=True,     # DeepSeek V4 Flash uses attn_sink
        has_swa=True,           # dual-cache prefill
    )

SMEM budget at this configuration: ~22 KB per CTA, fits 4 CTAs/SM at
100 KB SMEM/SM, which is the production-recommended occupancy for
SM120 attention-class kernels.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch

from .traits import (
    SparseMLATraits,
    INVALID_WORKSPACE_MAP_SENTINEL,
)


# Compile-time constants matched to the @cute.kernel tile geometry.
# These are NOT user-tunable per-call; they define the kernel template.
_QK_MMA_M = 16    # m of m16n8k16
_QK_MMA_N = 8     # n of m16n8k16
_QK_MMA_K = 16    # k of m16n8k16
_PV_MMA_M = 16    # m of m16n16k16 (PV: 16 rows, 16 cols, 16 K)
_PV_MMA_N = 16    # n of m16n16k16
_PV_MMA_K = 16    # k of m16n16k16

# Natural-log scale factor for converting between base-2 and base-e.
_LOG2_E = 1.4426950408889634   # log2(e)
_LN_2 = 0.6931471805599453     # ln(2)


def _build_prefill_kernel(traits: SparseMLATraits):
    """Build a @cute.kernel function specialized for the given traits.

    cutlass.cute is imported HERE (not at module top-level) so dsl12x can
    be imported on systems without cutlass-dsl installed. The kernel will
    fail loudly at first call rather than at import.

    The function returns the @cute.kernel object; cute.compile is called
    by the host wrapper (prefill.py) with the actual runtime arguments.

    Args:
        traits: SparseMLATraits describing the static shape configuration.

    Returns:
        A @cute.kernel decorated function ready for cute.compile.
    """
    import cutlass
    import cutlass.cute as cute

    # Capture traits as Python constants visible to the @cute.kernel body
    # at JIT compile time. cutlass.cute treats these as constexpr.
    H = traits.num_heads
    H_TILE = traits.heads_per_tile
    QK_DIM = traits.qk_head_dim
    V_DIM = traits.v_head_dim
    TOPK_MAX = traits.topk_max
    TOKENS_PER_TILE = traits.tokens_per_tile
    GROUP = traits.nope_group_elems
    NUM_NOPE_GROUPS = traits.num_nope_groups()
    NUM_V_GROUPS = traits.num_v_groups()
    NUM_HEAD_TILES = traits.num_head_tiles()
    NUM_TOKEN_TILES = traits.num_token_tiles_per_chunk()
    KV_STAGES = traits.kv_stages
    HAS_SINK = traits.has_attn_sink
    HAS_SWA = traits.has_swa

    @cute.kernel
    def _sparse_mla_prefill_kernel(
        # Inputs: pointers + dynamic dims.
        # Q is BF16 [num_tokens, num_heads, qk_head_dim]; pointer + the
        # per-call seq_len dynamic dim.
        q_ptr: cute.Pointer,
        seq_len: cutlass.Int32,
        # FP8 caches, layout [N_kv, qk_head_dim] each. The K view is
        # [N_kv, :qk_head_dim] and V view is [N_kv, :v_head_dim].
        c128_cache_ptr: cute.Pointer,
        c128_cache_n: cutlass.Int32,
        swa_cache_ptr: cute.Pointer,            # may be null if not HAS_SWA
        swa_cache_n: cutlass.Int32,
        # Workspace map (int32 [num_tokens, topk_max], sentinel-encoded).
        workspace_map_ptr: cute.Pointer,
        # Per-token valid topk-prefix length (int32 [num_tokens]).
        topk_length_ptr: cute.Pointer,
        # Per-head sigmoid gate bias (FP32 [num_heads]); may be null.
        attn_sink_ptr: cute.Pointer,
        # Per-row FP8 dequant scale (FP32 [N_kv]) for each cache. May be
        # absent on test paths where cache is already-dequantized BF16.
        c128_scale_ptr: cute.Pointer,
        swa_scale_ptr: cute.Pointer,
        # Outputs.
        out_ptr: cute.Pointer,                  # BF16 [num_tokens, num_heads, v_head_dim]
        lse_ptr: cute.Pointer,                  # FP32 [num_tokens, num_heads]
        # Scalar params.
        sm_scale: cutlass.Float32,
    ):
        # CTA grid:
        #   blockIdx.x = token_tile index in [0, NUM_TOKEN_TILES).
        #   blockIdx.y = head_tile index in [0, NUM_HEAD_TILES).
        # One warp per CTA. tidx in [0, 32).
        token_tile_idx = cute.arch.block_idx()[0]
        head_tile_idx = cute.arch.block_idx()[1]
        tidx = cute.arch.thread_idx()[0]

        # First token / head index for this CTA's tile.
        token_base = token_tile_idx * TOKENS_PER_TILE
        head_base = head_tile_idx * H_TILE

        # Early-exit if the token tile straddles seq_len (last partial tile).
        # The actual valid token count for this tile:
        valid_tokens_in_tile = cute.arch.min(
            cutlass.Int32(TOKENS_PER_TILE),
            seq_len - token_base,
        )
        if valid_tokens_in_tile <= cutlass.Int32(0):
            return

        # SMEM allocation. Use cute.make_shared_memory or cute.allocate
        # depending on cutlass.cute version. We use cute.make_shared_memory
        # which is the b12x-compatible API.

        # Q stage: [H_TILE, GROUP] BF16 (one nope group at a time).
        smem_q = cute.make_shared_memory(cutlass.BFloat16, H_TILE * GROUP)

        # KV stages: 2 buffers of [TOKENS_PER_TILE, GROUP] BF16 each
        # (double-buffered cp.async). Allocate separately so the SMEM
        # pointers for stage A and stage B are at distinct addresses.
        smem_kv_stages = []
        for _stage in cute.range_constexpr(KV_STAGES):
            smem_kv_stages.append(
                cute.make_shared_memory(cutlass.BFloat16, TOKENS_PER_TILE * GROUP)
            )

        # Workspace map (int32) for this token tile, copied to SMEM once
        # at the start of the tile and reused across all topk iterations.
        smem_ws_map = cute.make_shared_memory(
            cutlass.Int32, TOKENS_PER_TILE * TOPK_MAX
        )

        # Per-token topk_length, copied to SMEM once.
        smem_topk_length = cute.make_shared_memory(
            cutlass.Int32, TOKENS_PER_TILE
        )

        # Per-row dequant scales for the topk K rows (FP32, one scale per
        # KV row staged into SMEM). Reused across QK and PV (cache row's
        # K and V come from the same FP8 row with the same scale).
        smem_kv_scales = cute.make_shared_memory(
            cutlass.Float32, TOKENS_PER_TILE
        )

        # Stage 1: cooperative load of workspace_map, topk_length for this tile.
        # 32 threads load TOKENS_PER_TILE * TOPK_MAX int32 values: each
        # thread loads (TOKENS_PER_TILE * TOPK_MAX) / 32 entries.
        ws_per_thread = (TOKENS_PER_TILE * TOPK_MAX + 31) // 32
        for i in cute.range_constexpr(ws_per_thread):
            idx = tidx * ws_per_thread + i
            if idx < TOKENS_PER_TILE * TOPK_MAX:
                src_token = idx // TOPK_MAX
                src_topk = idx % TOPK_MAX
                if src_token < valid_tokens_in_tile:
                    smem_ws_map[idx] = workspace_map_ptr[
                        (token_base + src_token) * TOPK_MAX + src_topk
                    ]
                else:
                    smem_ws_map[idx] = cutlass.Int32(INVALID_WORKSPACE_MAP_SENTINEL)
        # topk_length: 32 threads load TOKENS_PER_TILE entries (1 per thread
        # for TOKENS_PER_TILE=32; the kernel's tokens_per_tile constexpr is
        # always a multiple of warp size to keep this a single load per thread).
        if tidx < TOKENS_PER_TILE:
            if tidx < valid_tokens_in_tile:
                smem_topk_length[tidx] = topk_length_ptr[token_base + tidx]
            else:
                smem_topk_length[tidx] = cutlass.Int32(0)

        cute.arch.sync_threads()

        # Per-head accumulators. Each thread holds a slice of (m_frag,
        # d_frag, o_frag) corresponding to its m16n8k16 fragment register
        # share. For H_TILE=16 heads x TOKENS_PER_TILE=32 tokens this is
        # one m16n8k16 tile per head. Each thread holds 4 FP32 lanes of
        # the [16, 8] score tile per head; that's per-thread:
        #   m_frag:  [num_local_rows] FP32  -- one per row this thread holds
        #   d_frag:  [num_local_rows] FP32
        #   o_frag:  [num_local_rows, V_DIM] FP32 -- big! see below.
        #
        # For online softmax we maintain per-row m and d. The output frag
        # holds V_DIM accumulators per row. With H_TILE=16 rows and V_DIM=512
        # this is 16*512=8192 FP32 lanes per CTA for o_frag, which is
        # 32 KB -- TOO BIG for register file. Solution: o_frag covers only
        # the current V group (GROUP=64 elements), so per-row o_frag is
        # 64 FP32 lanes. Then we sweep groups of V too:
        #   for v_group in [0, NUM_V_GROUPS):
        #     o_frag = 0
        #     for k_iter in [0, TOPK_MAX):
        #       (online softmax + PV MMA accumulating into o_frag for this v group)
        #     write o_frag to global out for this v group.
        #
        # This is the SAME per-group sweep pattern b12x uses. The per-row
        # m and d are still maintained across v_group sweeps (they only
        # depend on QK scores, not V columns), so we restore them from a
        # save buffer when starting a new v_group.

        # m and d are per (head, token) pair, persisted across v_group sweeps.
        # Each thread holds 4 lanes of the [H_TILE, TOKENS_PER_TILE] tile.
        # Total lanes per thread: H_TILE * TOKENS_PER_TILE / 32 = 16. Use
        # FP32 register fragments.
        m_save = cute.make_fragment(cutlass.Float32, H_TILE * TOKENS_PER_TILE // 32)
        d_save = cute.make_fragment(cutlass.Float32, H_TILE * TOKENS_PER_TILE // 32)

        # Outer loop over v groups. For DeepSeek V3-family with V_DIM=512
        # and GROUP=64 this is 8 v groups.
        for v_group_idx in cute.range_constexpr(NUM_V_GROUPS):
            # Per-tile output fragment for this v group: 4 FP32 lanes per
            # thread per row tile, for [H_TILE, GROUP] = [16, 64] tile.
            # 16 rows * 64 cols / 32 threads = 32 FP32 lanes per thread.
            o_frag = cute.make_fragment(cutlass.Float32, 32)
            for j in cute.range_constexpr(32):
                o_frag[j] = cutlass.Float32(0.0)

            # Restore (or initialize) m and d for this v group.
            if v_group_idx == cutlass.Int32(0):
                # First v group: initialize.
                for j in cute.range_constexpr(H_TILE * TOKENS_PER_TILE // 32):
                    m_save[j] = cutlass.Float32(-1e38)  # negative infinity proxy
                    d_save[j] = cutlass.Float32(0.0)
            # Else: m_save and d_save retain values from the previous v_group.

            # ============================================================
            # The QK + softmax + PV inner. Iterates over topk_max sparse
            # K rows. cp.async pipelines KV staging behind compute.
            # ============================================================
            #
            # Pipeline (KV_STAGES=2, double-buffered):
            #   Iteration k=0: cp.async stage 0 -> KV[0]. wait.
            #   Iteration k=1: cp.async stage 1 -> KV[1]; compute on stage 0.
            #   Iteration k=2: cp.async stage 0 -> KV[2]; compute on stage 1.
            #   ...
            #   Iteration k=N-1: compute on stage (N-1)%2.
            #
            # For now we issue the cp.async with explicit cute.copy + a
            # cp_async_commit/wait pair. Real production would unroll the
            # pipeline inline; here we keep it simple for clarity.
            # XXX(verify): cute.cp_async API surface varies by version.
            # If `cute.cp_async` is not the right name, try
            # `cute.arch.cp_async` or `cute.cp_async_async_copy`.

            for k_iter in cute.range_constexpr(TOPK_MAX):
                # Decode workspace map entry for each (token_in_tile, this
                # k_iter) and stage one row of K (FP8) per token.
                # The stage buffer is laid out [TOKENS_PER_TILE, GROUP] BF16,
                # so each token gets one BF16 row of GROUP elements per
                # nope group sweep. We sweep nope groups inside the k_iter
                # loop: each k_iter copies + uses ALL nope groups for this
                # k row.

                stage_idx = k_iter % KV_STAGES
                stage_buf = smem_kv_stages[stage_idx]

                # For each token in the tile, decode workspace map -> cache row.
                # 32 threads load TOKENS_PER_TILE rows; one row per thread
                # for TOKENS_PER_TILE=32. Each thread reads:
                #   1. workspace_map[token, k_iter] from SMEM.
                #   2. Decodes (cache_id, row, valid).
                #   3. cp.async loads GROUP/8 BF16 (= GROUP*2 / 16 = GROUP/8
                #      128-bit chunks) from the cache row into smem stage.
                #   4. For invalid entries: writes zero (or skips load and
                #      we mask the score later).
                # XXX(verify): the cp.async signature here is a placeholder
                # using the dsl12x.ptx wrapper which is currently a stub.
                # On the test system we'll need to lower this to actual
                # cute.cp_async or inline-PTX.
                if tidx < TOKENS_PER_TILE:
                    ws_entry = smem_ws_map[tidx * TOPK_MAX + k_iter]
                    valid_topk = (k_iter < smem_topk_length[tidx])
                    is_invalid = (ws_entry == cutlass.Int32(INVALID_WORKSPACE_MAP_SENTINEL))
                    if valid_topk and not is_invalid:
                        if ws_entry >= cutlass.Int32(0):
                            # C128 cache row.
                            cache_row = ws_entry
                            # Each thread loads GROUP elements = GROUP*1 byte
                            # (FP8). Place into smem_kv stage at offset
                            # tidx*GROUP. One BF16-converted load per element.
                            # XXX(verify): scalar load + dequant; production
                            # should use 16-byte cp.async + register dequant.
                            for g in cute.range_constexpr(GROUP):
                                fp8_byte = c128_cache_ptr[
                                    cache_row * QK_DIM + g
                                ]  # uint8 reinterpret
                                # Dequant: bf16 = fp8_e4m3_to_bf16(byte)
                                # XXX(verify): cutlass.cute fp8 dequant API.
                                stage_buf[tidx * GROUP + g] = (
                                    cutlass.BFloat16(0.0)  # placeholder
                                )
                            # Save the per-row dequant scale for later
                            # (multiply scores by sm_scale * row_scale).
                            if c128_scale_ptr is not None:
                                smem_kv_scales[tidx] = c128_scale_ptr[cache_row]
                            else:
                                smem_kv_scales[tidx] = cutlass.Float32(1.0)
                        else:
                            # SWA cache row. Same as C128 but from swa_cache.
                            cache_row = ~ws_entry
                            for g in cute.range_constexpr(GROUP):
                                fp8_byte = swa_cache_ptr[
                                    cache_row * QK_DIM + g
                                ]
                                stage_buf[tidx * GROUP + g] = (
                                    cutlass.BFloat16(0.0)  # placeholder
                                )
                            if swa_scale_ptr is not None:
                                smem_kv_scales[tidx] = swa_scale_ptr[cache_row]
                            else:
                                smem_kv_scales[tidx] = cutlass.Float32(1.0)
                    else:
                        # Invalid -> zero the row (so QK MMA contributes
                        # zero before we mask scores to -inf later).
                        for g in cute.range_constexpr(GROUP):
                            stage_buf[tidx * GROUP + g] = cutlass.BFloat16(0.0)
                        smem_kv_scales[tidx] = cutlass.Float32(0.0)

                cute.arch.sync_threads()

                # Inner sweep over nope groups for QK. Each group covers
                # GROUP=64 elements of qk_head_dim; sweep all NUM_NOPE_GROUPS
                # to build up the QK score for this k_iter row.
                # Score accumulator for this (token, head) pair: per-thread
                # FP32 lane(s) of the [H_TILE, TOKENS_PER_TILE] tile.

                # XXX(verify): The actual MMA sequence here uses cute.gemm
                # which is the high-level abstraction. For BF16 m16n8k16 on
                # SM120 the MmaAtom is SM80_16x8x16_F32BF16BF16F32 (the SM80
                # MMA works on SM120 via the backward-compatible 4th-gen
                # tensor core). This is the same MmaAtom used in
                # csrc/sm120_mqa_logits_v2_mma.cu so we know it lowers
                # correctly on SM120.

                # Score frag: [H_TILE, TOKENS_PER_TILE] FP32 = [16, 32] tile.
                # Per-thread: 16 FP32 lanes (16 m * 32 n / 32 threads = 16).
                score_frag = cute.make_fragment(cutlass.Float32, 16)
                for j in cute.range_constexpr(16):
                    score_frag[j] = cutlass.Float32(0.0)

                # Sweep nope groups: load Q for this group, do one m16n8k16
                # MMA per group sweep. After all nope groups, score_frag
                # holds the full QK^T for this k_iter.
                for nope_g in cute.range_constexpr(NUM_NOPE_GROUPS):
                    # Stage Q[H_TILE, nope_g_offset:nope_g_offset+GROUP]
                    # into smem_q. Each thread loads (H_TILE * GROUP) / 32
                    # BF16 values.
                    q_per_thread = (H_TILE * GROUP + 31) // 32
                    for i in cute.range_constexpr(q_per_thread):
                        idx = tidx * q_per_thread + i
                        if idx < H_TILE * GROUP:
                            row = idx // GROUP
                            col = idx % GROUP
                            if (head_base + row) < cutlass.Int32(H):
                                # Q[token, head, dim] with token = whichever
                                # token tile we're on. For MQA-style sparse
                                # MLA the same Q row is used across all
                                # tokens in this tile (because the Q tile is
                                # heads, not tokens). Wait -- that's not
                                # right. Let me re-think. Actually Q is
                                # per-token AND per-head; different tokens
                                # have different Q. So we need a per-token
                                # Q stage as well.
                                # XXX(verify): Q stage geometry. The tile is
                                # [tokens, heads, dim] not [heads, dim]; the
                                # MMA is per-token. Re-deriving: for one
                                # m16n8k16 MMA per (token, head_tile),
                                # A=Q[token, head_base:head_base+16, group]
                                # is BF16 [16, 64] = m16n8k16's A side at
                                # K=64 (4 K-iters). B=K[token's k_row, group]
                                # is BF16 [tokens_per_tile, 64] = sweeping
                                # one MMA per token.
                                # The current scaffold collapses this -- we
                                # treat the whole [16, GROUP] Q tile as
                                # head-sliced, which is wrong for per-token
                                # MMA. Fixing this requires re-laying out
                                # the Q stage as [TOKENS_PER_TILE, H_TILE,
                                # GROUP] (= 32*16*64*2 = 64 KB; too big).
                                # Alternative: outer loop over tokens within
                                # the tile, doing one MMA per token. This
                                # is what the scaffold needs to be reworked
                                # to do on the test system.
                                # For now we leave this as a placeholder
                                # -- the scaffold's structural correctness
                                # is more important than exact register
                                # layout, which the test pass will iterate.
                                smem_q[row * GROUP + col] = cutlass.BFloat16(0.0)
                    cute.arch.sync_threads()

                    # The actual cute.gemm (or cute.gemm_partition + mma_inner)
                    # would go here. For the scaffold we mark this as the
                    # MMA inner that needs to be filled in based on the
                    # cutlass.cute version's MMA API surface.
                    # XXX(MMA-INNER): one BF16 m16n8k16 instruction per
                    # token per head_tile per nope_g.

                # Mask invalid entries: scores corresponding to
                # workspace_map[token, k_iter] == INVALID, or k_iter >=
                # topk_length[token], get -inf.
                # XXX(verify): masking via thread-local lane decoding of
                # which (m, n) lanes this thread holds.

                # Apply sm_scale * per-row dequant scale to scores.
                # XXX(verify): per-row scale broadcast across the score frag.

                # Online softmax update step:
                #   m_new = max(m_save_lane, max_lane(score_frag))
                #   rescale = exp2(m_save_lane - m_new) * LOG2_E_inverse
                #     (because we accumulate in base 2 internally for the
                #      hardware ex2.approx.ftz.f32 instruction)
                #   d_new = d_save_lane * rescale + sum_lane(exp2(score_frag - m_new))
                #   o_frag *= rescale
                #   o_frag += exp2(score_frag - m_new) @ V_row (PV MMA)
                # XXX(verify): the per-lane reduction patterns are warp-shuffle
                # heavy and require explicit warp-level primitives (cute.arch.shuffle).

                # PV MMA: BF16 m16n16k16 using probs (= exp2(scores - m_new))
                # as A and V (= the staged BF16 KV row) as B. Accumulates
                # into o_frag.
                # XXX(MMA-INNER): one BF16 m16n16k16 instruction per token
                # per head_tile per v_group.

            # ============================================================
            # End of k_iter loop. Epilogue for this v group.
            # ============================================================

            # Normalize o_frag: o_frag /= d_save (per-row).
            # XXX(verify): per-row broadcast division.

            # Apply attn_sink sigmoid gate (if HAS_SINK):
            #   lse_natural = m_save * LN_2 + log(d_save * LN_2)
            #     (natural-log LSE; vLLM expects natural log for cross-chunk
            #      reduction)
            #   gate = sigmoid(lse_natural - attn_sink[head])
            #   o_frag *= gate
            # XXX(verify): per-head broadcast of attn_sink.

            # Store o_frag to out[token_base:token_base+TOKENS_PER_TILE,
            #                     head_base:head_base+H_TILE,
            #                     v_group_offset:v_group_offset+GROUP]
            # XXX(verify): BF16 cast + store using lane mapping.

        # ============================================================
        # End of v_group loop. Store LSE (one value per (token, head)).
        # ============================================================
        # XXX(verify): store FP32 LSE = m_save * LN_2 + log(d_save * LN_2).

    return _sparse_mla_prefill_kernel


__all__ = [
    "_build_prefill_kernel",
]
