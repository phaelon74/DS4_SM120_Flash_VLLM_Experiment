"""dsl12x.attention.indexer_kernel: SCAFFOLD for SM120 mqa_logits indexer kernel.

STATUS: SCAFFOLD ONLY. Mirrors the architectural pattern of the existing
C4 mqa_logits MMA kernel at csrc/sm120_mqa_logits_v2_mma.cu but in CuTe DSL.
The MMA inner is a TODO marker. Calling
``dsl12x.attention.indexer.run_mqa_logits`` raises NotImplementedError
until a follow-up session fills the inner.

Why a CuTe DSL version of mqa_logits?
======================================

The current C4 mqa_logits MMA (csrc/sm120_mqa_logits_v2_mma.cu) is raw
CUDA + inline PTX, and was the right tool for that work because the
math is single-pass GEMM-like (Q @ K^T, no softmax). It ships a 50-75%
TTFT improvement on the indexer hot path.

For the dsl12x library to "own all attention on SM120" -- the user's
stated end-goal -- we need a CuTe DSL version too, even if the raw-CUDA
version stays as a known-good reference. The CuTe DSL version unlocks:

  * Easier templating across kHeadDim variants without re-writing PTX
    inline assembly.
  * JIT-compiled launcher cache (no per-shape recompile cost from
    Python).
  * Composability with other dsl12x kernels (e.g., a future fused
    indexer-then-prefill path).

DeepSeek V4 Flash mqa_logits contract:

  * Math:      logits = Q @ K^T   (no softmax, just dot products)
  * Q:         BF16 [num_tokens, num_heads, kHeadDim]
  * K:         BF16 [N_kv, num_heads, kHeadDim]
  * logits:    FP32 [num_tokens, num_heads, N_kv]
  * Per-row:   topk_indices is computed downstream from logits (not by
               this kernel).

Per-CTA architecture (from csrc/sm120_mqa_logits_v2_mma.cu):

  M_TILE = 16 tokens
  N_PER_WARP = 8 (one m16n8k16 N tile per warp)
  NUM_WARPS = 4 (parallelism over N)
  kHeadDim varies (64 / 128 / 192); template C parameter

What's COMPLETE in this scaffold:

  * Module structure mirroring decode_kernel.py.
  * Public surface (indexer.py wrapper).
  * Documentation of the kernel contract and per-CTA work.

What's TODO (filled in by follow-up session):

  * The actual @cute.kernel function body, with cute.gemm doing the
    BF16 m16n8k16 MMA and writing FP32 logits to global.
  * Should be SHORTER than prefill_kernel.py because there's no
    softmax / online accumulation / epilogue gating.
"""

from __future__ import annotations

import torch


def _build_indexer_kernel(num_tokens: int, num_heads: int, kHeadDim: int):
    """Build a @cute.kernel for the mqa_logits indexer.

    SCAFFOLD: not implemented. Filling this in requires:

      1. Mirror csrc/sm120_mqa_logits_v2_mma.cu's CTA grid: M_TILE=16,
         N_PER_WARP=8, NUM_WARPS=4.
      2. cute.gemm with MmaAtom.SM80_16x8x16_F32BF16BF16F32 (same as
         hello_mma).
      3. Per-warp accumulator: 4 FP32 lanes per thread of the [16, 8]
         output tile.
      4. Output store: FP32 [num_tokens, num_heads, N_kv] in row-major.
      5. Template on kHeadDim: 64, 128, 192 instantiations.

    Args:
        num_tokens: Q token count.
        num_heads: number of heads.
        kHeadDim: head dimension; must be a multiple of 16 for m16n8k16.

    Returns:
        Would return a @cute.kernel decorated function. Currently raises.
    """
    raise NotImplementedError(
        "dsl12x mqa_logits indexer kernel is a scaffold. The MMA inner "
        "has not been written yet; see dsl12x/README.md and the docstring "
        "of _build_indexer_kernel for the implementation plan. The "
        "raw-CUDA version at csrc/sm120_mqa_logits_v2_mma.cu remains the "
        "production indexer path; the dsl12x version is a future-port "
        "target for ecosystem completeness."
    )
