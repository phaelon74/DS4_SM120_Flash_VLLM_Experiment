"""dsl12x.attention.traits: shape-keyed kernel selection traits.

A SparseMLATraits dataclass captures all the static (compile-time) shape
parameters of a sparse MLA prefill / decode call. Two calls with the same
traits hash share the same JIT-compiled kernel launcher; two calls with
different traits compile separate launchers.

The traits are intentionally narrow:

  * num_heads, qk_head_dim, v_head_dim       static across a model load
  * topk_max, chunk_size                     static across a serving config
  * dtype, has_attn_sink, has_swa            static across a request type

But NOT:

  * seq_len (per-call dynamic)
  * actual topk_length (per-token, dynamic)
  * specific attn_sink values (data, not shape)

So a given DeepSeek V4 Flash deployment with a fixed model + serving config
typically uses 1-3 distinct SparseMLATraits values across all live traffic
(differing only in chunk_size if the patcher is doing variable-size chunks).
The JIT cache size of 32 (b12x's default) leaves plenty of headroom.

The dataclass is frozen and hashable so it can be used as a dict key /
metadata_key directly in dsl12x.jit_cache.get_or_compile.

Mirrors b12x's SparseMLATraits but without GLM-5.1-isms (no page table size,
no per-page step, no NSA grouping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


# Default tile geometry. Mirrors b12x's per-CTA dimensions but tuned for the
# DeepSeek V4 Flash shape (qk_head_dim=576, v_head_dim=512). The constants
# can be overridden per-traits if a future shape needs different geometry.
DEFAULT_HEADS_PER_TILE = 16            # Q heads loaded per tile
DEFAULT_TOKENS_PER_TILE = 32           # KV tokens loaded per tile
DEFAULT_NOPE_GROUP_ELEMS = 64          # head-dim columns per group sweep
DEFAULT_KV_STAGES = 2                  # double-buffered cp.async stages


@dataclass(frozen=True)
class SparseMLATraits:
    """Static traits describing a sparse MLA prefill or decode call.

    All shape parameters are static (per model load + serving config). Per-call
    dynamic shapes (seq_len, per-token topk_length values) are passed at
    launch time, NOT at compile time, so they do not increase the cache size.

    Attributes:
        num_heads: Number of Q heads computed in attention. For DeepSeek V4
            Flash with TP, this is the per-device active head count
            (DG_SM120_ACTIVE_HEADS=32 in production compose). Drives the
            CTA grid (one CTA per head_tile = 16 heads).
        qk_head_dim: Q and K head dimension (== q.shape[-1] == kv.shape[-1]).
            For DeepSeek V3-family this is 576 = 512 latent + 64 RoPE.
        v_head_dim: V head dimension (the slice of kv used for V). Output
            head dimension. For DeepSeek V3-family this is 512. May equal
            qk_head_dim in some configurations (the BMM bridge requires this).
        topk_max: Maximum sparse topk width (DG_SM120_MAIN_TOPK_CAP=128 in
            production). Drives the per-tile workspace map SMEM size.
        chunk_size: Tokens per chunked-prefill outer iteration
            (DG_SM120_PREFILL_WORKSPACE_CHUNK=256). The kernel's outer loop
            over tokens iterates ``chunk_size / DEFAULT_TOKENS_PER_TILE``
            times per call.
        dtype: BF16 (production default for DeepSeek V4 Flash). Passed to
            the kernel for the staging buffer dtype.
        has_attn_sink: Whether the call passes attn_sink. Off path is a
            simpler epilogue (no sigmoid gate). On path multiplies output
            by ``sigmoid(lse - sink)`` per head.
        has_swa: Whether the workspace map encodes both C128 and SWA cache
            references (sentinel-encoded). False path does C128-only
            (workspace map entries are all C128 row indices).
        heads_per_tile: Number of heads per CTA tile. Default 16. Lower for
            cache-pressure-bound shapes; higher for compute-bound shapes.
        tokens_per_tile: KV tokens per CTA tile. Default 32. Drives KV stage
            SMEM size: ``tokens_per_tile * head_dim_per_group * 2 bytes``.
        nope_group_elems: Head-dim columns per group sweep. Default 64. The
            kernel's per-tile QK MMA inner sweeps ``qk_head_dim /
            nope_group_elems`` groups. Lower = smaller SMEM footprint per
            group, more outer iterations.
        kv_stages: Double-buffer count for the KV staging buffer. Default 2.
            Drives KV stage SMEM size: ``kv_stage_bytes * kv_stages``.

    Methods:
        smem_bytes_per_cta: Compute total per-CTA SMEM footprint to validate
            occupancy. Should be <= 25 KB for >=4 CTAs/SM at 100 KB SMEM/SM.

        cache_key: Stable hashable summary suitable for jit_cache metadata_key.

    Construction:
        Use SparseMLATraits.from_call(q, kv, indices, ...) to build from the
        actual tensors (validates shape compatibility and infers dtype).
    """

    num_heads: int
    qk_head_dim: int
    v_head_dim: int
    topk_max: int
    chunk_size: int
    dtype: torch.dtype = torch.bfloat16
    has_attn_sink: bool = False
    has_swa: bool = False

    heads_per_tile: int = DEFAULT_HEADS_PER_TILE
    tokens_per_tile: int = DEFAULT_TOKENS_PER_TILE
    nope_group_elems: int = DEFAULT_NOPE_GROUP_ELEMS
    kv_stages: int = DEFAULT_KV_STAGES

    def __post_init__(self) -> None:
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.qk_head_dim <= 0 or self.qk_head_dim % self.nope_group_elems != 0:
            raise ValueError(
                f"qk_head_dim={self.qk_head_dim} must be > 0 and a multiple "
                f"of nope_group_elems={self.nope_group_elems}"
            )
        if self.v_head_dim <= 0 or self.v_head_dim > self.qk_head_dim:
            raise ValueError(
                f"v_head_dim={self.v_head_dim} must be > 0 and <= "
                f"qk_head_dim={self.qk_head_dim}"
            )
        if self.topk_max <= 0:
            raise ValueError(f"topk_max must be > 0, got {self.topk_max}")
        if self.chunk_size <= 0 or self.chunk_size % self.tokens_per_tile != 0:
            raise ValueError(
                f"chunk_size={self.chunk_size} must be > 0 and a multiple "
                f"of tokens_per_tile={self.tokens_per_tile}"
            )
        if self.heads_per_tile <= 0 or self.num_heads % self.heads_per_tile != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be a multiple of "
                f"heads_per_tile={self.heads_per_tile}"
            )
        if self.kv_stages < 1 or self.kv_stages > 4:
            raise ValueError(
                f"kv_stages={self.kv_stages} must be in [1, 4]"
            )

    @classmethod
    def from_call(
        cls,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        v_head_dim: int,
        attn_sink: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor],
        chunk_size: int,
        active_heads: Optional[int] = None,
        has_swa: bool = False,
        topk_max: Optional[int] = None,
    ) -> "SparseMLATraits":
        """Infer traits from the actual call tensors.

        Args:
            q: [num_tokens, num_heads, qk_head_dim] BF16 query tensor.
            kv: [N_kv, 1, qk_head_dim] BF16 KV tensor (single-head MLA).
            indices: [num_tokens, 1, topk_width] int32 sparse index tensor.
            v_head_dim: int, the V dimension (output head dim).
            attn_sink: Optional [num_heads] FP32 sink bias.
            topk_length: Optional [num_tokens] int32 valid-prefix lengths.
            chunk_size: Outer-loop chunk size (DG_SM120_PREFILL_WORKSPACE_CHUNK).
            active_heads: Override of num_heads to use the first N heads only.
                If None, uses q.shape[1] (all heads).
            has_swa: True if the workspace map will encode SWA rows.
            topk_max: Override of topk_max. If None, uses indices.shape[-1].

        Returns:
            SparseMLATraits suitable for kernel selection.
        """
        if q.dim() != 3:
            raise ValueError(f"q must be 3D [tokens, heads, dim]; got {tuple(q.shape)}")
        if kv.dim() != 3 or kv.shape[1] != 1:
            raise ValueError(
                f"kv must be 3D [N_kv, 1, dim] (single-head MLA); got {tuple(kv.shape)}"
            )
        if indices.dim() != 3 or indices.shape[1] != 1:
            raise ValueError(
                f"indices must be 3D [tokens, 1, topk]; got {tuple(indices.shape)}"
            )
        if q.shape[-1] != kv.shape[-1]:
            raise ValueError(
                f"q.shape[-1]={q.shape[-1]} != kv.shape[-1]={kv.shape[-1]}"
            )
        num_heads = active_heads if active_heads is not None else q.shape[1]
        return cls(
            num_heads=int(num_heads),
            qk_head_dim=int(q.shape[-1]),
            v_head_dim=int(v_head_dim),
            topk_max=int(topk_max if topk_max is not None else indices.shape[-1]),
            chunk_size=int(chunk_size),
            dtype=q.dtype,
            has_attn_sink=attn_sink is not None,
            has_swa=bool(has_swa),
        )

    def smem_bytes_per_cta(self) -> int:
        """Compute per-CTA SMEM footprint at this traits configuration.

        Used to validate occupancy before kernel selection. The kernel will
        not work if this exceeds SMEM/CTA opt-in (99 KB on SM120 with the
        default carveout). For >=4 CTAs/SM at 100 KB SMEM/SM the budget is
        ~25 KB per CTA.

        Components:
            Q stage:        heads_per_tile * nope_group_elems * 2 B (BF16)
            KV stages:      tokens_per_tile * nope_group_elems * 2 B * kv_stages
            Workspace map:  tokens_per_tile * topk_max * 4 B (int32)
            Scale buf:      ~4 KB (estimate, FP8 dequant scales per group)
            Idx buf:        ~256 B
        """
        q_stage = self.heads_per_tile * self.nope_group_elems * 2
        kv_stages = (
            self.tokens_per_tile * self.nope_group_elems * 2 * self.kv_stages
        )
        workspace_map = self.tokens_per_tile * self.topk_max * 4
        scale_buf = 4096
        idx_buf = 256
        return q_stage + kv_stages + workspace_map + scale_buf + idx_buf

    def num_nope_groups(self) -> int:
        """Number of group sweeps per (q_idx, head_tile) tile.

        The qk_head_dim is processed in groups of nope_group_elems to keep
        SMEM bounded. For qk_head_dim=576 with nope_group_elems=64 this is
        9 groups per tile.
        """
        return self.qk_head_dim // self.nope_group_elems

    def num_v_groups(self) -> int:
        """Number of group sweeps for V (PV MMA inner).

        Often equal to num_nope_groups but may differ if v_head_dim !=
        qk_head_dim. The kernel's PV MMA inner sweeps these groups.
        """
        return (self.v_head_dim + self.nope_group_elems - 1) // self.nope_group_elems

    def num_head_tiles(self) -> int:
        """Number of head tiles per token (CTA grid Y dimension typically)."""
        return self.num_heads // self.heads_per_tile

    def num_token_tiles_per_chunk(self) -> int:
        """Number of token tiles per chunk."""
        return self.chunk_size // self.tokens_per_tile

    def cache_key(self) -> Tuple:
        """Stable hashable summary suitable for jit_cache metadata_key.

        Returns a tuple of basic types (int, str, bool) so it can be used
        across processes / pickled / logged.
        """
        return (
            "dsl12x_sparse_mla",
            self.num_heads,
            self.qk_head_dim,
            self.v_head_dim,
            self.topk_max,
            self.chunk_size,
            str(self.dtype),
            self.has_attn_sink,
            self.has_swa,
            self.heads_per_tile,
            self.tokens_per_tile,
            self.nope_group_elems,
            self.kv_stages,
        )

    def describe(self) -> str:
        """Human-readable one-line description for logs."""
        return (
            f"SparseMLATraits("
            f"H={self.num_heads},"
            f"QK={self.qk_head_dim},"
            f"V={self.v_head_dim},"
            f"topk={self.topk_max},"
            f"chunk={self.chunk_size},"
            f"sink={self.has_attn_sink},"
            f"swa={self.has_swa},"
            f"smem={self.smem_bytes_per_cta()//1024}KB)"
        )


# Sentinel value used in the dual-cache workspace map to mark invalid
# (out-of-topk_length) entries. The kernel checks for this exact value and
# zeros the corresponding score before softmax. Chosen as a value that
# cannot collide with a valid C128 row index (always non-negative) or a
# valid SWA row index (encoded as ~row, so the most-negative valid SWA
# encoding is INT32_MIN+1 corresponding to row INT32_MAX).
INVALID_WORKSPACE_MAP_SENTINEL = -0x7FFFFFFF - 1  # INT32_MIN


def encode_c128_row(row: int) -> int:
    """Encode a C128 cache row as a workspace map entry. Result >= 0."""
    if row < 0:
        raise ValueError(f"C128 row must be non-negative, got {row}")
    return row


def encode_swa_row(row: int) -> int:
    """Encode a SWA cache row as a workspace map entry. Result < 0 (~row)."""
    if row < 0:
        raise ValueError(f"SWA row must be non-negative, got {row}")
    encoded = ~row
    if encoded == INVALID_WORKSPACE_MAP_SENTINEL:
        raise ValueError(
            f"SWA row {row} encodes to INVALID_WORKSPACE_MAP_SENTINEL, which "
            f"is reserved for invalid entries. Maximum supported SWA row is "
            f"{(~INVALID_WORKSPACE_MAP_SENTINEL) - 1}."
        )
    return encoded


def is_c128_entry(workspace_map_entry: int) -> bool:
    """True if entry encodes a C128 cache row (non-negative)."""
    return workspace_map_entry >= 0


def is_swa_entry(workspace_map_entry: int) -> bool:
    """True if entry encodes a SWA cache row (negative but not sentinel)."""
    return workspace_map_entry < 0 and workspace_map_entry != INVALID_WORKSPACE_MAP_SENTINEL


def is_invalid_entry(workspace_map_entry: int) -> bool:
    """True if entry is the sentinel for invalid (out-of-topk_length)."""
    return workspace_map_entry == INVALID_WORKSPACE_MAP_SENTINEL


def decode_swa_row(workspace_map_entry: int) -> int:
    """Decode a SWA workspace map entry back to its row index."""
    if not is_swa_entry(workspace_map_entry):
        raise ValueError(f"not a SWA entry: {workspace_map_entry}")
    return ~workspace_map_entry
