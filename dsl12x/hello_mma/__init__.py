"""dsl12x.hello_mma: minimal toolchain validation kernel.

A trivial @cute.kernel that performs one BF16 m16n8k16 MMA and writes the
FP32 accumulator to global memory. Used to:

  1. Confirm the cutlass-dsl toolchain is installed and SM120-targeted.
  2. Confirm the JIT compile path produces a working launcher.
  3. Confirm the dsl12x.jit_cache LRU cache hits on re-call (compile once,
     replay many).

If hello_mma cannot compile or run on a given system, no production dsl12x
kernel will work either -- this is the gating smoke test before attempting
the much bigger sparse MLA prefill kernel.
"""

from .kernel import hello_mma_run, hello_mma_reference

__all__ = ["hello_mma_run", "hello_mma_reference"]
