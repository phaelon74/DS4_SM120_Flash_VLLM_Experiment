"""dsl12x.attention.decode: Host wrapper for the sparse MLA decode kernel.

STATUS: SCAFFOLD. The decode kernel itself is not implemented in this
initial release. Calling run_sparse_mla_decode raises NotImplementedError;
the patcher's existing decode path (from docker/patch_vllm_deepseekv4.py)
remains the production decode dispatcher unchanged.

When the kernel is filled in (follow-up session), this wrapper handles:

  * Shape validation (1 token per call).
  * Traits inference.
  * JIT cache lookup + compile + warmup.
  * Workspace map construction (single-cache or dual-cache).
  * Output + LSE allocation.
  * Stream selection.

Public surface (will work once the kernel is implemented):

    from dsl12x.attention.decode import run_sparse_mla_decode, warmup
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch


def run_sparse_mla_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the dsl12x sparse MLA decode kernel.

    SCAFFOLD: NotImplementedError until the kernel inner is written. The
    patcher's DG_SM120_DSL12X_DECODE branch (when added) wraps this in
    try/except and falls back to the existing decode path on failure.

    Same semantics as the FlashMLA decode contract:
        q:    BF16 [1, num_heads, qk_head_dim]
        kv:   BF16 [N_kv, 1, qk_head_dim]
        indices: int32 [1, 1, topk_max]
        topk_length: int32 [1]
        attn_sink: optional FP32 [num_heads]

    Returns:
        (out, lse) tuple.
    """
    raise NotImplementedError(
        "dsl12x sparse MLA decode is a scaffold. Set "
        "DG_SM120_DSL12X_DECODE=0 (production default) to use the "
        "patcher's existing decode path. The kernel will be filled in "
        "in a follow-up session; see dsl12x/README.md."
    )


def warmup(traits_list, device: Optional[torch.device] = None) -> None:
    """No-op warmup until the decode kernel is implemented."""
    return None


def env_enabled() -> bool:
    """True if DG_SM120_DSL12X_DECODE is set to enable the (future) decode path."""
    return os.environ.get("DG_SM120_DSL12X_DECODE", "0").lower() in (
        "1", "true", "yes", "on",
    )


__all__ = [
    "run_sparse_mla_decode",
    "warmup",
    "env_enabled",
]
