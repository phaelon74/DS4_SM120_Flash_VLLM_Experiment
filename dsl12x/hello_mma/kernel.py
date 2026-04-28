"""dsl12x.hello_mma.kernel: One @cute.kernel doing one BF16 m16n8k16 MMA.

Math (per-CTA, single warp = 32 threads):
    A: BF16 [16, 16]
    B: BF16 [8, 16]
    C: FP32 [16, 8]   (output)
    C = A @ B^T       (one m16n8k16 MMA instruction)

The CTA is a single warp; one MMA fills the whole output tile. This is the
smallest interesting CuTe DSL kernel:

  * Exercises the cute.compile -> JIT path.
  * Validates that nvidia-cutlass-dsl-libs-cu13 + sm_120a is producing
    runnable PTX for BF16 tensor-core MMA on RTX PRO 6000.
  * Provides a compile-once-replay-many target for jit_cache validation.

If this fails on a given system, none of the production dsl12x kernels will
work either; the smoke test gates everything else.

The implementation deliberately mirrors b12x's coding patterns (single-warp,
explicit cute.tensor + cute.layout, @cute.kernel decoration) so the
production kernels feel familiar.
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
    import cutlass.cute as cute

    @cute.kernel
    def hello_mma_kernel(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
    ):
        # One warp = one CTA. Use a 32-thread grid with one CTA per launch.
        tidx = cute.arch.thread_idx()[0]

        # Tile shapes for one m16n8k16 MMA.
        # A is [M=16, K=16] BF16, B is [N=8, K=16] BF16, C is [M=16, N=8] FP32.
        # We organize SMEM as straight row-major BF16/FP32 for clarity;
        # this is not optimized for bank-conflict avoidance because it's a
        # toolchain smoke test, not a perf path.
        smem_a = cute.make_shared_memory(cutlass.BFloat16, 16 * 16)
        smem_b = cute.make_shared_memory(cutlass.BFloat16, 8 * 16)

        # Cooperative SMEM load: 32 threads loading 16*16 = 256 BF16 values
        # for A is 8 BF16 per thread; for B 8*16 = 128 / 32 = 4 BF16 per
        # thread. Use scalar loads (the smoke test does not need cp.async).
        for i in cute.range_constexpr(8):
            idx = tidx * 8 + i
            if idx < 16 * 16:
                smem_a[idx] = a_ptr[idx]
        for i in cute.range_constexpr(4):
            idx = tidx * 4 + i
            if idx < 8 * 16:
                smem_b[idx] = b_ptr[idx]
        cute.arch.sync_threads()

        # The actual m16n8k16 MMA: one instruction does the whole tile.
        # cute.gemm with MmaAtom::SM80_16x8x16 (which works on SM120 via the
        # 4th-gen tensor core's backward-compatible m16n8k16 instruction).
        #
        # The accumulator lives in registers distributed across the 32
        # threads of the warp; each thread holds 4 FP32 lanes of C.
        c_frag = cute.make_fragment(cutlass.Float32, 4)
        for i in cute.range_constexpr(4):
            c_frag[i] = cutlass.Float32(0.0)

        # cute.gemm executes one MMA on (smem_a, smem_b, c_frag).
        # The exact API surface depends on cutlass.cute version; this is the
        # canonical pattern from cutlass docs. If this doesn't compile on a
        # given cutlass-dsl version the smoke test will fail loudly with a
        # clear error pointing at this line.
        cute.gemm(
            smem_a,  # A SMEM tile
            smem_b,  # B SMEM tile
            c_frag,  # C register fragment (in-place accumulate)
            mma_atom=cute.MmaAtom.SM80_16x8x16_F32BF16BF16F32,
        )

        # Each thread writes its 4 FP32 lanes of C back to global. The lane
        # mapping for m16n8k16 is the standard PTX layout:
        # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
        #   #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
        # Thread t holds (row, col) pairs:
        #   element 0: (t/4,        (t%4)*2 + 0)
        #   element 1: (t/4,        (t%4)*2 + 1)
        #   element 2: (t/4 + 8,    (t%4)*2 + 0)
        #   element 3: (t/4 + 8,    (t%4)*2 + 1)
        for i in cute.range_constexpr(4):
            row = (tidx // 4) + (8 if i >= 2 else 0)
            col = (tidx % 4) * 2 + (i % 2)
            if row < 16 and col < 8:
                c_ptr[row * 8 + col] = c_frag[i]

    return hello_mma_kernel


def hello_mma_run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute the smoke MMA on tensors a [16, 16] BF16, b [8, 16] BF16.

    Returns C = A @ B^T as FP32 [16, 8].

    Uses dsl12x.jit_cache so the second call with the same shape reuses the
    compiled launcher (compile cost amortized).

    Raises:
        RuntimeError: if dsl12x is not running on SM120, or if cutlass.cute
            cannot compile the kernel.
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
    # Single CTA, one warp.
    launcher(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)(a, b, c)
    return c


def hello_mma_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: C = A @ B^T in FP32, used to validate hello_mma_run."""
    return (a.float() @ b.float().T).contiguous()
