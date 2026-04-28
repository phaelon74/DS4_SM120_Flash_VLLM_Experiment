"""dsl12x.hello_mma.kernel: One @cute.kernel doing one BF16 m16n8k16 MMA.

Math (per-CTA, single warp = 32 threads):
    A: BF16 [16, 16] row-major
    B: BF16 [8,  16] row-major
    C: FP32 [16, 8]  row-major   (output)
    C = A @ B^T  (one m16n8k16 MMA instruction, accumulator = 0)

This is the smallest interesting CuTe DSL kernel:

  * Exercises the cute.compile -> JIT -> launcher path end-to-end.
  * Validates that nvidia-cutlass-dsl-libs-cu13 + sm_120a produces runnable
    PTX for the BF16 -> F32 m16n8k16 warp tensor-core MMA.
  * Provides a compile-once / replay-many target for jit_cache validation.

If this fails on a given system, none of the production dsl12x kernels will
work either; the smoke test gates everything else.

API references (all verified against nvidia-cutlass-dsl-libs-cu13==4.4.2 on
SM120 via scripts/dsl12x_discover.py + scripts/dsl12x_discover_deep.py +
direct read of cute/nvgpu/warp/mma.py and cute/arch/smem.py):

    cute.kernel                                 - device function decorator
    cute.compile(fn, *bound_args)               - returns a launcher
    cute.arch.alloc_smem(dtype, n_elems, align) - static SMEM, returns Pointer
    cute.arch.thread_idx() / sync_threads()     - block-level primitives
    cute.make_layout(shape, stride=...)         - layout builder
    cute.make_tensor(ptr, layout)               - tensor view over a pointer
    cute.make_rmem_tensor_like(src)             - register fragment matching src
    cute.autovec_copy(src, dst)                 - universal vectorized copy
    cute.gemm(atom, d, a, b, c)                 - executes D = A*B + C
    cute.make_mma_atom(op)                      - wrap an op as an MmaAtom
    cute.make_tiled_mma(atom, atom_layout=(1,1,1))
    tiled_mma.get_slice(tidx).partition_A/B/C(tensor)
    cute.nvgpu.warp.MmaF16BF16Op(ab_dtype, acc_dtype, shape_mnk)
        - constructor-validated: (BFloat16, Float32, (16, 8, 16))
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import torch

from .. import jit_cache, runtime

logger = logging.getLogger(__name__)


def _build_kernel():
    """Lazily build the @cute.kernel function.

    cutlass.cute is imported here, not at module level, so dsl12x can be
    imported on systems without cutlass-dsl (the smoke test will then fail
    cleanly at first call rather than on import).
    """
    import cutlass
    from cutlass import BFloat16, Float32
    import cutlass.cute as cute
    from cutlass.cute.nvgpu.warp import MmaF16BF16Op

    @cute.kernel
    def hello_mma_kernel(
        a_gmem_ptr: cute.Pointer,  # BF16 elements; logical shape [16,16] row-major
        b_gmem_ptr: cute.Pointer,  # BF16 elements; logical shape [8, 16] row-major
        c_gmem_ptr: cute.Pointer,  # F32  elements; logical shape [16, 8] row-major
    ):
        tid = cute.arch.thread_idx()[0]

        # Static SMEM allocations (alignment defaults to dtype.width//8 = 2B
        # for BF16; we bump to 128B for tensor-core friendliness even though
        # the smoke test's scalar copy does not benefit from it).
        smem_a_ptr = cute.arch.alloc_smem(BFloat16, 16 * 16, alignment=128)
        smem_b_ptr = cute.arch.alloc_smem(BFloat16, 8 * 16, alignment=128)

        # Cooperative scalar copy gmem -> smem. 32 threads / 256 BF16 = 8/thread for A,
        # 32 threads / 128 BF16 = 4/thread for B. Use the raw pointers (flat 1D) for
        # the move; we'll wrap them in a 2D layout for the MMA partitioning below.
        for i in cute.range_constexpr(8):
            smem_a_ptr[tid * 8 + i] = a_gmem_ptr[tid * 8 + i]
        for i in cute.range_constexpr(4):
            smem_b_ptr[tid * 4 + i] = b_gmem_ptr[tid * 4 + i]
        cute.arch.sync_threads()

        # Build 2D tensor views with EXPLICIT row-major strides so the layout
        # matches the PyTorch contiguous tensors on the host side.
        #   PyTorch a[16,16].contiguous() has stride (16, 1)  -> K-minor (row-major)
        #   PyTorch b[8, 16].contiguous() has stride (16, 1)  -> K-minor
        #   PyTorch c[16, 8].contiguous() has stride ( 8, 1)  -> N-minor
        sA_layout = cute.make_layout((16, 16), stride=(16, 1))
        sB_layout = cute.make_layout((8, 16),  stride=(16, 1))
        gC_layout = cute.make_layout((16, 8),  stride=( 8, 1))

        sA = cute.make_tensor(smem_a_ptr, sA_layout)
        sB = cute.make_tensor(smem_b_ptr, sB_layout)
        gC = cute.make_tensor(c_gmem_ptr, gC_layout)

        # Build the MMA atom for the m16n8k16 BF16 -> F32 warp instruction.
        # Constructor signature verified against cute/nvgpu/warp/mma.py:
        #   @dataclass(frozen=True)
        #   class MmaF16BF16Op(WarpMmaOp):
        #       ab_dtype: Type[Numeric]
        #       acc_dtype: Type[Numeric]
        #       shape_mnk: Shape
        # __post_init__ validates: (BFloat16, Float32, (16, 8, 16)) is OK.
        mma_op = MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        mma_atom = cute.make_mma_atom(mma_op)
        tiled_mma = cute.make_tiled_mma(mma_atom)

        # Per-thread partitioning of the SMEM/GMEM tensors according to the
        # m16n8k16 thread-value layout.
        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)   # this thread's view of A in SMEM
        tCsB = thr_mma.partition_B(sB)   # this thread's view of B in SMEM
        tCgC = thr_mma.partition_C(gC)   # this thread's view of C in GMEM

        # Register fragments for the MMA inputs/output.
        rA = cute.make_rmem_tensor_like(tCsA)   # BF16 frag (matches src dtype)
        rB = cute.make_rmem_tensor_like(tCsB)   # BF16 frag
        rC = cute.make_rmem_tensor_like(tCgC)   # F32  frag (matches src dtype)

        # Move SMEM -> registers. autovec_copy picks the right vector width;
        # we don't need an explicit ldmatrix atom for the smoke test.
        cute.autovec_copy(tCsA, rA)
        cute.autovec_copy(tCsB, rB)

        # Zero the accumulator. The fragment is small (4 FP32 per thread for
        # m16n8k16) so a constexpr loop is fine.
        for i in cute.range_constexpr(4):
            rC[i] = Float32(0.0)

        # Execute the MMA: rC = rA * rB + rC.
        # cute.gemm signature: gemm(atom, d, a, b, c). The PTX m16n8k16 op
        # expects A as M-K and B as N-K (same K-major orientation), which is
        # what our row-major (16,16) and (8,16) layouts give us.
        cute.gemm(mma_atom, rC, rA, rB, rC)

        # Write the result back to GMEM.
        cute.autovec_copy(rC, tCgC)

    return hello_mma_kernel


def _invoke_launcher(launcher, args, *, grid, block, stream):
    """Try the few plausible launcher invocation patterns for cutlass-dsl 4.x.

    The cute.compile() return value's call signature varies slightly across
    cutlass-dsl versions and is not directly inspectable (the launcher is a
    compiled MLIR pass-through, not a Python callable with __signature__).
    Rather than hardcode one guess and fail mysteriously, this helper tries
    the patterns in order and remembers which one worked for subsequent
    calls. The successful pattern is printed once to stderr so the operator
    sees which path the toolchain wants.

    Patterns tried (in order):
        1. launcher(*args, grid=, block=, stream=)
        2. launcher(*args)  (kernel decorator may carry launch params)
        3. launcher(LaunchConfig(...), *args)
        4. launcher(grid=, block=, stream=)(*args)  (curried)
    """
    cached = getattr(_invoke_launcher, "_pattern", None)
    if cached is not None:
        return cached(launcher, args, grid=grid, block=block, stream=stream)

    errors = []

    def _p1(lr, ag, *, grid, block, stream):
        return lr(*ag, grid=grid, block=block, stream=stream)

    def _p2(lr, ag, *, grid, block, stream):
        del grid, block, stream
        return lr(*ag)

    def _p3(lr, ag, *, grid, block, stream):
        import cutlass
        cfg = cutlass.LaunchConfig(grid=grid, block=block, stream=stream)
        return lr(cfg, *ag)

    def _p4(lr, ag, *, grid, block, stream):
        return lr(grid=grid, block=block, stream=stream)(*ag)

    for name, fn in (("kwargs", _p1), ("plain", _p2), ("LaunchConfig", _p3), ("curried", _p4)):
        try:
            result = fn(launcher, args, grid=grid, block=block, stream=stream)
        except (TypeError, AttributeError) as exc:
            errors.append(f"  pattern {name!r}: {type(exc).__name__}: {exc}")
            continue
        import sys
        print(
            f"[dsl12x] launcher invocation pattern detected: {name!r}",
            file=sys.stderr,
            flush=True,
        )
        _invoke_launcher._pattern = fn
        return result

    raise RuntimeError(
        "dsl12x: cute.compile launcher could not be invoked with any known "
        "pattern. Errors:\n" + "\n".join(errors)
    )


def hello_mma_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute the smoke MMA on tensors a [16,16] BF16, b [8,16] BF16.

    Returns C = A @ B^T as FP32 [16, 8].

    Uses dsl12x.jit_cache so the second call with the same shape reuses the
    compiled launcher (compile cost amortized).

    Raises:
        RuntimeError: if dsl12x is not running on SM120, or if cutlass.cute
            cannot compile the kernel, or if the launcher cannot be invoked.
    """
    runtime.assert_sm120("hello_mma_run")
    if a.shape != (16, 16) or b.shape != (8, 16):
        raise ValueError(
            f"hello_mma expects a=[16,16] b=[8,16], got a={tuple(a.shape)} b={tuple(b.shape)}"
        )
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError(
            f"hello_mma expects BF16 inputs, got a.dtype={a.dtype} b.dtype={b.dtype}"
        )
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("hello_mma expects CUDA tensors")
    a = a.contiguous()
    b = b.contiguous()
    c = torch.zeros((16, 8), dtype=torch.float32, device=a.device)

    def compile_fn():
        import cutlass.cute as cute

        kernel_fn = _build_kernel()
        return cute.compile(kernel_fn, a, b, c)

    launcher = jit_cache.get_or_compile(
        kernel_fn=_build_kernel,
        metadata_key=("hello_mma", str(a.dtype), str(b.dtype)),
        compile_fn=compile_fn,
    )

    stream = runtime.current_cuda_stream()
    _invoke_launcher(
        launcher,
        (a, b, c),
        grid=(1, 1, 1),
        block=(32, 1, 1),
        stream=stream,
    )
    return c


def hello_mma_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: C = A @ B^T in FP32, used to validate hello_mma_run."""
    return (a.float() @ b.float().T).contiguous()
