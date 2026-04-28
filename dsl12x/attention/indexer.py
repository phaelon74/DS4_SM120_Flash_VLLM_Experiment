"""dsl12x.attention.indexer: Host wrapper for the mqa_logits indexer kernel.

STATUS: SCAFFOLD. The indexer kernel itself is not implemented in this
initial release. Calling run_mqa_logits raises NotImplementedError; the
existing C4 mqa_logits MMA at csrc/sm120_mqa_logits_v2_mma.cu remains
the production indexer path.

When the kernel is filled in (follow-up session), this wrapper handles:

  * Shape validation.
  * JIT cache lookup + compile + warmup.
  * Output allocation.
  * Stream selection.

Public surface (will work once the kernel is implemented):

    from dsl12x.attention.indexer import run_mqa_logits, warmup
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def run_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the dsl12x mqa_logits indexer kernel.

    SCAFFOLD: NotImplementedError until the kernel inner is written. The
    patcher's existing path that calls
    csrc/sm120_mqa_logits_v2_mma.cu remains the production indexer
    dispatcher unchanged.

    Args:
        q: BF16 [num_tokens, num_heads, kHeadDim].
        k: BF16 [N_kv, num_heads, kHeadDim].
        out: optional pre-allocated FP32 [num_tokens, num_heads, N_kv].

    Returns:
        FP32 [num_tokens, num_heads, N_kv] = Q @ K^T.
    """
    raise NotImplementedError(
        "dsl12x mqa_logits indexer is a scaffold. Set "
        "DG_SM120_DSL12X_INDEXER=0 (production default) to use the "
        "existing C4 mqa_logits MMA at csrc/sm120_mqa_logits_v2_mma.cu. "
        "The dsl12x version will be filled in in a follow-up session; "
        "see dsl12x/README.md."
    )


def warmup(shapes_list, device: Optional[torch.device] = None) -> None:
    """No-op warmup until the indexer kernel is implemented."""
    return None


def env_enabled() -> bool:
    """True if DG_SM120_DSL12X_INDEXER is set to enable the (future) indexer path."""
    return os.environ.get("DG_SM120_DSL12X_INDEXER", "0").lower() in (
        "1", "true", "yes", "on",
    )


__all__ = [
    "run_mqa_logits",
    "warmup",
    "env_enabled",
]
