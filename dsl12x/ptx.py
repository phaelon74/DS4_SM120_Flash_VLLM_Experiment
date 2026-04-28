"""dsl12x.ptx: Inline-PTX wrappers for SM120-specific instructions.

CuTe DSL exposes most common instructions as `cute.copy`, `cute.gemm`, etc.,
but a few SM120-specific paths benefit from raw PTX:

  * cp.async.cg.shared.global       --  asynchronous predicated 16-byte loads
                                        from global to shared (the workhorse
                                        for double-buffered KV staging).
  * ex2.approx.ftz.f32              --  fast hardware exponential, used in
                                        the online softmax inner.
  * lg2.approx.ftz.f32              --  fast hardware log2, used to convert
                                        natural log to log2 in LSE epilogue.

All wrappers are decorated with @dsl_user_op so they can be called from inside
@cute.kernel functions. They generate inline LLVM that lowers to the right
PTX instruction at JIT time.

Why not use cute.cp_async / cute.fast_math? cutlass.cute does provide these,
but with abstractions optimized for general use (predicated copies through
TMA, etc.). For the prefill kernel's double-buffered KV staging, raw
cp.async.cg gives us:

  * Predicated load with a single PTX instruction (cute.copy adds wrapping).
  * Explicit cache hint (.cg = bypass L1, cache in L2 only) which is the
    right hint for KV cache reads (read-once, no spatial locality).
  * Vector width control (16-byte = 8 BF16 = 4 FP32 = 16 FP8) without going
    through cutlass's CopyAtom indirection.

For ex2/lg2, the cute.math.exp/log paths use slower libdevice calls; the
.approx.ftz.f32 hardware variants are 1-2 cycles each on SM120 and are
accuracy-correct for softmax (relative error ~2^-23, well under BF16 noise).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cutlass.cute as cute


def cp_async_load_128b_pred(smem_ptr, gmem_ptr, predicate):
    """Asynchronous predicated 16-byte (128-bit) load from global to shared.

    Lowers to: ``cp.async.cg.shared.global [smem], [gmem], 16, %p`` PTX.

    When ``predicate`` is False the load issues but writes zero to the SMEM
    destination -- this is the canonical predication pattern for boundary
    handling without a conditional branch.

    Args:
        smem_ptr: Pointer in shared memory address space (must be 16-byte
            aligned). Use cute.address_space.shared markers.
        gmem_ptr: Pointer in global memory address space (16-byte aligned).
        predicate: bool. False -> zero-fill the destination.

    Note:
        After all cp.async issues for a stage, the warp must call
        ``cute.cp_async_commit_group()`` and ``cute.cp_async_wait_group(0)``
        (or N for double-buffered) to synchronize. This wrapper does NOT
        commit -- the caller is responsible for batching commits across
        multiple loads, which is the whole point of cp.async (overlap).

    The implementation is currently a stub. The real lowering will use
    ``cutlass.cute.arch.cp_async`` or an inline-asm @dsl_user_op once the
    kernel scaffold is in place. Stubbed here so the import surface and
    type signatures are stable for the kernel files; the kernel files
    intentionally use the dsl12x.ptx surface so we have one place to swap
    the underlying implementation.
    """
    # TODO(dsl12x P2 prefill): replace with @dsl_user_op inline-PTX lowering
    # using cutlass._mlir.dialects.llvm. The b12x reference at
    # https://github.com/lukealonso/b12x/blob/master/b12x/cute/utils.py
    # uses cutlass.cute.arch.cp_async for the equivalent path; we will
    # mirror that approach but with the raw PTX inline so we have control
    # of the cache hint (.cg vs .ca) and predication.
    raise NotImplementedError(
        "cp_async_load_128b_pred is a stub; the prefill kernel will lower "
        "this to inline cp.async.cg.shared.global PTX once the kernel "
        "scaffold is in place. Use cutlass.cute.arch.cp_async in the "
        "interim if you need it."
    )


def exp2_approx_ftz_f32(x):
    """Fast hardware 2^x for FP32 (ex2.approx.ftz.f32 PTX).

    1-2 cycle latency on SM120 (vs. ~10 for libdevice exp2f). Used in online
    softmax inner: instead of computing ``exp(score)`` we compute
    ``2^(score * log2(e))`` so the hot path is one ex2.approx with a constant
    factor folded into sm_scale. Standard FlashAttention/SDPA practice.

    Note:
        ftz = "flush to zero" for denormals. Safe for softmax where any
        denormal score has effectively zero weight in the final output.

    Stubbed; will be replaced with @dsl_user_op inline-PTX in the prefill
    kernel implementation.
    """
    # TODO(dsl12x P2 prefill): @dsl_user_op + cutlass._mlir.dialects.llvm
    # InlineAsmOp with constraint string "=f,f" and the ex2.approx.ftz.f32
    # instruction. Returns one f32.
    raise NotImplementedError(
        "exp2_approx_ftz_f32 is a stub; the prefill kernel will lower "
        "this to inline ex2.approx.ftz.f32 PTX. Use cute.math.exp2 in the "
        "interim if you need it (slower but functional)."
    )


def log2_approx_ftz_f32(x):
    """Fast hardware log2(x) for FP32 (lg2.approx.ftz.f32 PTX).

    Used in LSE epilogue: vLLM expects natural-log LSE for the chunked-prefill
    cross-chunk reduction. Internally the kernel accumulates softmax in
    base 2 (using exp2_approx_ftz_f32), so the final LSE conversion is
    ``lse_natural = lse_base2 * ln(2)``. We do not actually need lg2 in the
    standard path; this wrapper exists for symmetry with b12x's helper set
    and as a forward-compat hook if a future kernel needs base-2 log
    conversion in the inner.

    Stubbed; rarely used on the hot path.
    """
    # TODO(dsl12x P2 prefill): @dsl_user_op + cutlass._mlir.dialects.llvm
    # InlineAsmOp with constraint "=f,f" and lg2.approx.ftz.f32. Returns
    # one f32.
    raise NotImplementedError(
        "log2_approx_ftz_f32 is a stub; rarely used on the kernel hot path."
    )
