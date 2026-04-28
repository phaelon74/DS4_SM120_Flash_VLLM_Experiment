"""dsl12x.attention.decode_kernel: SCAFFOLD for the SM120 sparse MLA decode kernel.

STATUS: SCAFFOLD ONLY. The MMA inner is a TODO marker. Calling
``dsl12x.attention.decode.run_sparse_mla_decode`` raises NotImplementedError
until a follow-up session fills the inner. The architectural skeleton
(CTA grid, SMEM layout, online softmax, sigmoid-gate epilogue, dual-cache
workspace map decoding) is in place so the follow-up session has a clear
shape to fill in.

Why a separate kernel from prefill?
====================================

Decode and prefill have different optimal tile geometry:

  Prefill (this session, prefill_kernel.py):
    Per-CTA: 16 Q heads x 32 KV tokens x qk_head_dim (group-swept).
    Why: prefill processes O(seq_len) tokens per call, so the CTA grid
    is sized for many-token tiles to amortize per-CTA fixed overhead.

  Decode (this scaffold):
    Per-CTA: 32 Q heads x 1 token x qk_head_dim (group-swept).
    Why: decode processes one token per request, so the CTA grid is just
    NUM_HEAD_TILES (e.g. 1 CTA total for 32 heads with 32 heads/tile).
    Single-buffer KV staging is fine (no double-buffer benefit when
    only one token's K/V is needed per CTA).

DeepSeek V4 Flash decode contract:

  q:              BF16 [1, num_heads, qk_head_dim]
  kv:             BF16 [N_kv, 1, qk_head_dim]  (already-merged C128+SWA)
  indices:        int32 [1, 1, topk_max]
  topk_length:    int32 [1]
  attn_sink:      FP32 [num_heads]  (optional; sigmoid-gate epilogue)
  out:            BF16 [1, num_heads, v_head_dim]
  lse:            FP32 [1, num_heads]

Per-CTA work:
  * Stage Q for this head_tile (~32 heads x qk_head_dim BF16 = small SMEM).
  * For each topk slot k = 0..topk_max-1:
    * Decode workspace_map[k], early-exit if invalid or k >= topk_length.
    * Stage K and V row from cache.
    * QK MMA (head_tile heads x 1 token x qk_head_dim).
    * Online softmax update (m, d, o frags).
    * PV MMA.
  * attn_sink sigmoid gate.
  * Store out, lse.

What's COMPLETE in this scaffold:

  * Module structure mirroring prefill_kernel.py.
  * Public surface (decode.py wrapper).
  * Traits (decode-specific tile dims).
  * SMEM layout calculation.
  * Documentation of the kernel contract and per-CTA work.

What's TODO (filled in by follow-up session):

  * The actual @cute.kernel function body, with cute.gemm + online softmax
    + epilogue. Should be SHORTER than prefill_kernel.py because the
    outer "tokens per tile" loop becomes a single iteration.
"""

from __future__ import annotations

from typing import Optional

import torch

from .traits import SparseMLATraits


def _build_decode_kernel(traits: SparseMLATraits):
    """Build a @cute.kernel for sparse MLA decode at these traits.

    SCAFFOLD: not implemented. Filling this in is the second deliverable
    of the dsl12x attention work.

    Args:
        traits: SparseMLATraits describing the static configuration. For
            decode, the relevant fields are num_heads, qk_head_dim,
            v_head_dim, topk_max, dtype, has_attn_sink. (chunk_size and
            tokens_per_tile are not used for decode -- always 1 token.)

    Returns:
        Would return a @cute.kernel decorated function. Currently raises.

    To implement (follow-up session):

      1. Mirror prefill_kernel.py's structure but with TOKENS_PER_TILE=1.
      2. Drop the outer-tokens-per-tile loop; the kernel is single-token.
      3. KV stages are single-buffer (no double-buffer benefit at 1 token).
      4. Q stage is larger per CTA: H_TILE=32 heads x qk_head_dim group-swept.
      5. Online softmax is per-head, single-token; m and d are scalars per
         head (not per-token-per-head as in prefill).
      6. Output store: BF16 [1, num_heads, v_head_dim].
    """
    raise NotImplementedError(
        "dsl12x sparse MLA decode kernel is a scaffold. The MMA inner has "
        "not been written yet; see dsl12x/README.md and the docstring of "
        "_build_decode_kernel for the implementation plan."
    )
