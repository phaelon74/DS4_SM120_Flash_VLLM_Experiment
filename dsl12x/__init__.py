"""dsl12x: Standalone CuTe DSL kernel library for DeepSeek V4 on NVIDIA SM120.

This package is independent of b12x. b12x is the architectural reference -- a
SM120/SM121 kernel library targeting GLM-5.1 NSA -- but no code in dsl12x
imports from b12x. Where dsl12x reuses architectural patterns (single-warp
per-CTA, per-group Q+KV streaming, double-buffered cp.async, online softmax,
JIT-compiled @cute.kernel with LRU host launcher cache), the patterns are
re-implemented from first principles using cutlass.cute primitives.

Public surface (this session ships only the prefill kernel; decode and indexer
are scaffolds):

    from dsl12x.attention.prefill import run_sparse_mla_prefill, warmup
    from dsl12x.attention.traits import SparseMLATraits

    # Decode and indexer are scaffold-only this session; kernels exist as
    # @cute.kernel skeletons with TODO markers at the MMA inner. Calling them
    # raises NotImplementedError until the MMA inner is filled in.
    from dsl12x.attention.decode import run_sparse_mla_decode  # NotImplementedError
    from dsl12x.attention.indexer import run_mqa_logits        # NotImplementedError

Why standalone (not derived from b12x):

    1. b12x targets the GLM-5.1 NSA contract: page table, packed cache row
       layout (nope|scales|rope), uniform per-page step. DeepSeek V4 Flash
       has a different contract: topk indices + topk_length, dual cache
       (C128 compressed + SWA), sigmoid-gate attn_sink epilogue. Building on
       top of b12x would mean carrying GLM-5.1-isms forever.
    2. Direct FP8 cache reads (no BMM bridge) require kernel ownership of the
       FP8 dequant + workspace map encoding scheme. Hard to do as a thin
       adapter on top of b12x.
    3. Long term, dsl12x can grow to own the full SM120 attention surface for
       DeepSeek V4 (prefill + decode + indexer + others) without any external
       dependency on b12x's evolution.

What's NOT in dsl12x (intentional separation):

    * MoE kernels stay raw CUDA in csrc/. b12x provides MoE via
      b12x.integration.tp_moe; we keep our existing csrc MoE path because it
      works well and dsl12x is attention-only.
    * Dense GEMM kernels (FP8xFP8, FP8xFP4) stay raw CUDA in csrc/.
    * mHC HyperConnection prenorm stays raw CUDA in csrc/sm120_*hc_prenorm*.cu.
      The math is simple enough that raw CUDA + mma.sync.aligned PTX is the
      right tool, and we avoid mixing the dsl12x toolchain with existing C++
      build paths.

Toolchain requirement:

    * cutlass.cute (provided by ``nvidia-cutlass-dsl-libs-cu13==4.4.2`` in the
      base vLLM container).
    * SM120 / sm_120a target (CUTE_DSL_ARCH=sm_120a).
    * BF16 m16n8k16 + m16n16k16 MMA (4th-gen tensor cores).

See dsl12x/README.md for the architecture diagram, kernel-by-kernel scope,
and a description of what's real vs scaffold in this initial release.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
