"""dsl12x.attention: SM120 DeepSeek V4 attention kernels.

Three kernels in this subpackage:

  * prefill: standalone CuTe DSL sparse MLA prefill kernel (the deep one
    this session).
  * decode: SCAFFOLD for sparse MLA decode (skeleton + TODOs).
  * indexer: SCAFFOLD for mqa_logits indexer (skeleton + TODOs).

Common contract types live in traits.py (SparseMLATraits, etc.).
"""

from .traits import SparseMLATraits

__all__ = ["SparseMLATraits"]
