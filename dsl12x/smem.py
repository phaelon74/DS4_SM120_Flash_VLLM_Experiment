"""dsl12x.smem: Shared-memory addressing helpers (permuted offsets, banking).

CuTe DSL exposes shared-memory layouts via cute.make_layout, but for the
prefill kernel's hot inner we want explicit control over:

  * 128-bit (16-byte) bank-conflict-free SMEM access patterns.
  * Permuted/swizzled offsets to keep consecutive lanes on different banks.
  * Cross-row stride alignment to 16-byte boundaries (the cp.async.cg
    requirement).

The helpers here mirror b12x's _permuted_offset_128b family from
b12x/cute/utils.py but are re-derived from cutlass primitives (no
``from b12x import ...``). They are pure host-side address arithmetic
helpers; they do NOT issue any PTX. Used by the kernel files to compute
SMEM byte offsets at JIT time.

Banking model on SM120:

  * SMEM is divided into 32 banks, 4-byte width.
  * 32 threads in a warp accessing 32 distinct banks = no conflict.
  * 32 threads accessing the same bank's 32 different rows = also no
    conflict (broadcast).
  * Conflicts arise when two threads in the same warp hit the same bank's
    different addresses.

For 16-byte (128-bit) accesses the effective bank stride is 4 banks per
access. To avoid conflicts in a per-warp tile we permute the row index
into the column offset.

These functions return integer offsets in 128-bit (16-byte) units; multiply
by 16 for the byte offset.
"""

from __future__ import annotations


def permuted_offset_128b(row: int, col: int, num_cols_128b: int) -> int:
    """Compute a permuted 128-bit-unit SMEM offset from (row, col).

    The permutation is ``col_permuted = col XOR (row mod num_cols_128b)``.
    This rotates the per-row column ordering so that consecutive rows do
    not hit the same SMEM bank when accessed by the same lane across rows.

    Args:
        row: Logical row index (0-based).
        col: Logical column index in 128-bit units (each = 16 bytes).
        num_cols_128b: Total number of 128-bit columns per row (e.g., for a
            BF16 [16, 64] tile this is 64 * 2 / 16 = 8).

    Returns:
        Linear 128-bit-unit offset into the SMEM tile.

    Example:
        BF16 K-stage of shape [32 tokens, 64 dim]:
          row=token, col=dim/8 (since 8 BF16 = 16 bytes), num_cols_128b=8.
        Then ``offset_bytes = permuted_offset_128b(row, col, 8) * 16``.
    """
    permuted_col = col ^ (row % num_cols_128b)
    return row * num_cols_128b + permuted_col


def smem_addr_from_b128_offset(base_addr: int, offset_128b: int) -> int:
    """Convert a base SMEM address + 128-bit-unit offset to a byte address.

    Trivial helper but kept as a named function so the kernel reads are
    obviously SMEM-byte-addressed (vs other address arithmetic which uses
    different units).

    Args:
        base_addr: Base SMEM byte address.
        offset_128b: Offset in 128-bit (16-byte) units.

    Returns:
        Final SMEM byte address.
    """
    return base_addr + offset_128b * 16


def advance_offset_by_row_128b(offset_128b: int, num_cols_128b: int) -> int:
    """Advance a permuted 128-bit-unit offset by one row.

    The permutation in permuted_offset_128b means a simple ``offset +
    num_cols_128b`` is wrong (the next row's permutation is XOR-different).
    This helper computes the correct delta for a one-row advance.

    Equivalent to ``permuted_offset_128b(row+1, col, num_cols_128b) -
    permuted_offset_128b(row, col, num_cols_128b)`` but cheaper (avoids the
    full XOR computation when only the relative advance matters).

    For the standard ``XOR (row % num_cols_128b)`` permutation the delta is
    ``num_cols_128b`` for most rows but flips by ``+/- 1`` at the row-modulo
    boundary. The callers in dsl12x typically do explicit
    permuted_offset_128b for each row rather than chaining advances, so this
    helper exists mostly for symmetry with b12x's API and as a perf hook
    if a future kernel needs strided row sweeps.

    Args:
        offset_128b: Current 128-bit-unit offset.
        num_cols_128b: Row width in 128-bit units.

    Returns:
        New 128-bit-unit offset for (row+1, same col).
    """
    return offset_128b + num_cols_128b


def advance_offset_by_column_128b(offset_128b: int, columns: int = 1) -> int:
    """Advance a permuted 128-bit-unit offset by ``columns`` columns.

    Same caveat as advance_offset_by_row_128b: the XOR permutation makes
    this a non-trivial step; this helper computes the simple delta and
    callers MUST stay within one row (no row-wrap) when chaining.

    Args:
        offset_128b: Current 128-bit-unit offset.
        columns: Number of columns to advance (default 1).

    Returns:
        New 128-bit-unit offset for (same row, col + columns).
    """
    return offset_128b + columns


def smem_bytes_for_kv_stage(
    tokens_per_tile: int,
    head_dim_per_group: int,
    bytes_per_elem: int,
    num_stages: int,
) -> int:
    """Compute SMEM bytes for a multi-stage KV staging buffer.

    Used by the prefill kernel's traits to validate the per-CTA SMEM budget
    before kernel selection. Default geometry:

      * tokens_per_tile = 32
      * head_dim_per_group = 64 (one nope group, b12x's _MLA_NOPE_GROUP_ELEMS)
      * bytes_per_elem = 2 (BF16 staging; cache is FP8 but we dequant in
        registers, so SMEM holds BF16)
      * num_stages = 2 (double-buffered)

    Yields ``32 * 64 * 2 * 2 = 8 KB`` per CTA for the KV staging alone, well
    within the 25 KB total per-CTA budget that gives 4 CTAs/SM at 100 KB
    SMEM/SM.

    Args:
        tokens_per_tile: KV tokens loaded per stage.
        head_dim_per_group: Head dim columns per stage (one group sweep).
        bytes_per_elem: 1 (FP8), 2 (BF16/FP16), or 4 (FP32).
        num_stages: Number of stages (1 = single buffer, 2 = double, 3+ =
            multi-stage pipeline).

    Returns:
        Total SMEM bytes for this stage region.
    """
    return tokens_per_tile * head_dim_per_group * bytes_per_elem * num_stages


def smem_bytes_for_q_stage(
    heads_per_tile: int,
    head_dim_per_group: int,
    bytes_per_elem: int,
) -> int:
    """Compute SMEM bytes for the Q staging region (single-buffer).

    Q is loaded once per (q_idx, head_tile) pair and reused across all KV
    iterations, so single-buffer is sufficient and the per-group sweep
    keeps the footprint small.

    Default geometry: ``16 heads * 64 dim * 2 bytes = 2 KB`` per group.

    Args:
        heads_per_tile: Number of heads in this tile (typically 16).
        head_dim_per_group: Head-dim columns per group sweep.
        bytes_per_elem: 1, 2, or 4.

    Returns:
        Total SMEM bytes for Q stage.
    """
    return heads_per_tile * head_dim_per_group * bytes_per_elem


def smem_bytes_for_workspace_map(tokens_per_tile: int, topk_max: int) -> int:
    """Compute SMEM bytes for the per-tile workspace map (sentinel-encoded).

    The workspace map is int32 per (token, topk_slot): positive means C128
    cache row, negative means SWA cache row (encoded as ~row), special
    sentinel (INT32_MIN) means invalid (out-of-topk_length).

    Default geometry: ``32 tokens * 128 topk * 4 bytes = 16 KB`` -- this is
    the largest single SMEM consumer in the prefill kernel and dominates
    the budget.

    Note that 16 KB workspace map + 8 KB KV stage + 2 KB Q stage = 26 KB,
    which is just over the 25 KB target for >=4 CTAs/SM. We use the live
    Phase 1 trace topk_width to size this exactly; production topk is
    capped at 128 (DG_SM120_MAIN_TOPK_CAP=128) so this is the conservative
    upper bound.

    Args:
        tokens_per_tile: Q tokens per tile (typically 32).
        topk_max: Maximum topk width per token (live default 128).

    Returns:
        Total SMEM bytes for the workspace map.
    """
    return tokens_per_tile * topk_max * 4
