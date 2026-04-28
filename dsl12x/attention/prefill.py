"""dsl12x.attention.prefill: Host wrapper for the sparse MLA prefill kernel.

Public surface:

    run_sparse_mla_prefill(q, kv, indices, sm_scale, ...)
        Match the existing _sm120_flash_mla_sparse_prefill_fwd signature
        from docker/patch_vllm_deepseekv4.py so the patcher branch is a
        drop-in. Builds traits, looks up/compiles the kernel, constructs
        the sentinel-encoded workspace map (single-cache for first ship),
        dispatches the kernel.

    warmup(traits_list)
        Pre-compile the kernel for each traits in the list, populating
        the JIT cache. Called from patcher init with shapes from the
        Phase 1 trace before vllm serve accepts traffic. Adds ~1 minute
        to startup, eliminates per-shape cold-start latency on user
        prompts.

What this implements vs SCAFFOLDS:

    IMPLEMENTED:
        * Shape validation matching the FlashMLA sparse_prefill_fwd contract.
        * Traits inference from the actual call tensors.
        * JIT cache lookup + compile + warmup.
        * Sentinel-encoded workspace map construction (currently single-cache;
          dual-cache encoding is a follow-up that depends on patcher-level
          access to the C128/SWA cache split).
        * Output and LSE allocation.
        * Stream selection.

    SCAFFOLDED (will need iteration on the test system):
        * The actual cute.compile call -- relies on the kernel scaffold in
          prefill_kernel.py compiling successfully under the live cutlass-dsl
          version.
        * Direct FP8 cache reads -- requires patcher-level exposure of the
          FP8 cache pointers + dequant scales, not done in this initial
          ship. The current contract takes BF16 kv input (matches the BMM
          bridge contract); the kernel reads BF16 and skips dequant.

Once the kernel compiles + runs on the test system, this wrapper is the
right place to add direct-FP8 dispatch (separate code path keyed on the
patcher's exposed FP8 cache pointer being non-null).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Iterable, List, Optional, Tuple

import torch

from .. import jit_cache, runtime
from .traits import (
    INVALID_WORKSPACE_MAP_SENTINEL,
    SparseMLATraits,
)

logger = logging.getLogger(__name__)


_warmup_done_lock = threading.Lock()
_warmup_done_for: set[Tuple] = set()


def _build_workspace_map(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    kv_total_tokens: int,
) -> torch.Tensor:
    """Build the sentinel-encoded workspace map for the dsl12x kernel.

    Encoding (single-cache first ship):

      workspace_map[m, k] = indices[m, 0, k]      if 0 <= k < topk_length[m]
                                                  AND indices[m, 0, k] >= 0
                                                  AND indices[m, 0, k] < kv_total_tokens
      workspace_map[m, k] = INVALID_SENTINEL      otherwise

    Args:
        indices: int32 [num_tokens, 1, topk_max] from the patcher.
        topk_length: optional int32 [num_tokens] valid prefix length per token.
        kv_total_tokens: number of rows in the kv cache, for bounds checking.

    Returns:
        int32 [num_tokens, topk_max] workspace map suitable for the kernel.

    For the dual-cache (C128 + SWA) extension this is where we'd encode
    SWA rows as ~swa_row. The patcher would pass the cache split at call
    time. Not in scope for the initial ship -- the kernel scaffold in
    prefill_kernel.py already handles the encoding via has_swa=True; only
    this host-side construction needs the additional logic.
    """
    if indices.dim() != 3 or indices.shape[1] != 1:
        raise ValueError(
            f"indices must be [num_tokens, 1, topk_max]; got {tuple(indices.shape)}"
        )
    indices_2d = indices.squeeze(1)  # [num_tokens, topk_max]

    # Validity mask: index in [0, kv_total_tokens) AND index < topk_length[m].
    valid = (indices_2d >= 0) & (indices_2d < kv_total_tokens)
    if topk_length is not None:
        # topk_length is per-token; broadcast to [num_tokens, topk_max].
        positions = torch.arange(
            indices_2d.shape[1], device=indices_2d.device, dtype=indices_2d.dtype
        ).reshape(1, -1)
        valid = valid & (positions < topk_length.reshape(-1, 1))

    workspace_map = torch.where(
        valid,
        indices_2d.to(torch.int32),
        torch.full_like(indices_2d, INVALID_WORKSPACE_MAP_SENTINEL, dtype=torch.int32),
    ).to(torch.int32).contiguous()
    return workspace_map


def run_sparse_mla_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
    active_heads: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the dsl12x sparse MLA prefill kernel.

    Drop-in replacement for the patcher's BMM-bridge path inside
    _sm120_flash_mla_sparse_prefill_fwd. Same input/output contract.

    Args:
        q: BF16 [num_tokens, num_heads, qk_head_dim]. Query tensor for
            this prefill chunk.
        kv: BF16 [N_kv, 1, qk_head_dim]. Pre-merged C128+SWA cache
            (vLLM's combined kv tensor; future enhancement: take raw FP8
            cache + dequant in kernel).
        indices: int32 [num_tokens, 1, topk_max]. Sparse indices into kv.
        sm_scale: float, the softmax temperature.
        d_v: int, V head dimension. Default 512 (DeepSeek V3-family).
        attn_sink: optional FP32 [num_heads], sigmoid-gate bias per head.
        topk_length: optional int32 [num_tokens], per-token valid topk
            prefix length.
        out: optional pre-allocated BF16 [num_tokens, num_heads, d_v].
        lse: optional pre-allocated FP32 [num_tokens, num_heads].
        chunk_size: int. Defaults to num_tokens (whole call is one chunk).
            If smaller, the kernel processes the input in chunks of this
            size; useful only if num_tokens > what fits in a single CTA
            grid (typically not the case at production chunk_size=256).
        active_heads: int. Use only the first N heads. Defaults to
            num_heads. In production DG_SM120_ACTIVE_HEADS=32 caps this
            to 32.

    Returns:
        (out, max_logits, lse) tuple matching FlashMLA sparse_prefill_fwd.

    Raises:
        RuntimeError: if dsl12x is not running on SM120.
        RuntimeError: if cute.compile fails for these traits.
        ValueError: if input shapes are inconsistent.

    The patcher branch wraps this call in try/except (RuntimeError,
    cuda.CudaError) so a kernel failure falls back to the BMM bridge
    without breaking serving.
    """
    runtime.assert_sm120("run_sparse_mla_prefill")

    # Default arguments and shape validation.
    if not q.is_cuda or not kv.is_cuda or not indices.is_cuda:
        raise ValueError("q, kv, indices must all be CUDA tensors")
    if q.dtype != kv.dtype:
        raise ValueError(
            f"q.dtype={q.dtype} must equal kv.dtype={kv.dtype}; "
            f"the kernel currently expects matching staging dtypes"
        )

    num_tokens = q.shape[0]
    if active_heads is None:
        active_heads = q.shape[1]
    if chunk_size is None:
        chunk_size = num_tokens

    # Build traits (compile-time signature).
    traits = SparseMLATraits.from_call(
        q=q,
        kv=kv,
        indices=indices,
        v_head_dim=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        chunk_size=chunk_size,
        active_heads=active_heads,
        has_swa=False,                 # single-cache first ship
        topk_max=indices.shape[-1],
    )

    # Validate SMEM budget.
    smem_per_cta = traits.smem_bytes_per_cta()
    smem_per_sm = runtime.get_smem_per_sm(q.device.index or 0)
    if smem_per_cta > smem_per_sm:
        raise RuntimeError(
            f"dsl12x sparse MLA prefill: traits {traits.describe()} require "
            f"{smem_per_cta} B SMEM/CTA, exceeds device limit {smem_per_sm} B. "
            f"Consider reducing nope_group_elems or topk_max."
        )

    # Allocate outputs if not provided.
    if out is None:
        out = torch.empty(
            (num_tokens, q.shape[1], d_v), device=q.device, dtype=q.dtype
        )
    if lse is None:
        lse = torch.empty(
            (num_tokens, q.shape[1]), device=q.device, dtype=torch.float32
        )

    # Build the workspace map (host-side; this is data-dependent so per
    # call rather than per traits).
    workspace_map = _build_workspace_map(
        indices=indices,
        topk_length=topk_length,
        kv_total_tokens=kv.shape[0],
    )

    # JIT-compile or fetch the cached launcher.
    def _compile_fn():
        from .prefill_kernel import _build_prefill_kernel
        import cutlass.cute as cute

        kernel_fn = _build_prefill_kernel(traits)
        # Pass dummy tensors of the right shape/dtype to cute.compile so
        # it can specialize the launcher to the input layouts. The actual
        # data is passed at launch time via the launcher call below.
        # XXX(verify): the cute.compile signature surface depends on the
        # cutlass-dsl version. The b12x reference at b12x/cute/utils.py
        # uses the pattern below; if cutlass.cute.compile changes, this
        # is one place that needs updating.
        return cute.compile(kernel_fn)

    launcher = jit_cache.get_or_compile(
        kernel_fn=run_sparse_mla_prefill,    # cache namespace key
        metadata_key=traits.cache_key(),
        compile_fn=_compile_fn,
    )

    # Build kernel arguments.
    # Pointers: cutlass.cute uses cute.make_ptr; for tensor-typed args we
    # pass the tensor directly and cute.compile generates the right
    # bindings.
    seq_len = num_tokens
    c128_cache_n = kv.shape[0]
    c128_cache = kv.squeeze(1)            # [N_kv, qk_head_dim]
    swa_cache_dummy = torch.empty(0, device=q.device, dtype=kv.dtype)
    sink_or_dummy = (
        attn_sink.to(torch.float32) if attn_sink is not None
        else torch.empty(0, device=q.device, dtype=torch.float32)
    )
    topk_length_or_default = (
        topk_length if topk_length is not None
        else torch.full(
            (num_tokens,), indices.shape[-1],
            device=q.device, dtype=torch.int32,
        )
    )
    # Per-row dequant scales: not provided in this contract (kv is already
    # dequantized BF16). Pass empty placeholders; the kernel checks the
    # pointer for null and skips the multiply.
    c128_scale_dummy = torch.empty(0, device=q.device, dtype=torch.float32)
    swa_scale_dummy = torch.empty(0, device=q.device, dtype=torch.float32)

    # Grid: (NUM_TOKEN_TILES, NUM_HEAD_TILES, 1).
    grid = (
        traits.num_token_tiles_per_chunk(),
        traits.num_head_tiles(),
        1,
    )
    block = (32, 1, 1)
    stream = runtime.current_cuda_stream()

    # Launch.
    # XXX(verify): the launcher invocation pattern depends on the
    # cute.compile output. b12x's pattern is `launcher(grid=..., block=...,
    # stream=...)(args)`. If cutlass.cute changes this is the second place
    # to update.
    launcher(grid=grid, block=block, stream=stream)(
        q,
        seq_len,
        c128_cache,
        c128_cache_n,
        swa_cache_dummy,
        0,
        workspace_map,
        topk_length_or_default,
        sink_or_dummy,
        c128_scale_dummy,
        swa_scale_dummy,
        out,
        lse,
        float(sm_scale),
    )

    # max_logits is part of the FlashMLA contract but not used by the
    # patcher's downstream consumers; fill with NaN to match BMM bridge.
    max_logits = torch.full_like(lse, float("nan"))

    return out, max_logits, lse


def warmup(traits_list: Iterable[SparseMLATraits], device: Optional[torch.device] = None) -> None:
    """Pre-compile the kernel for each traits in the list.

    Called from the patcher init (in patch_vllm_deepseekv4.py via the
    DG_SM120_DSL12X_PREFILL branch) with shapes from the Phase 1 trace so
    the first user request does not pay the 1-3 s JIT compile cost per
    shape.

    Idempotent: the same traits is only compiled once per process.

    Args:
        traits_list: iterable of SparseMLATraits to pre-compile.
        device: optional CUDA device to issue dummy launches on. Defaults
            to the current device.

    The function issues a dummy launch with each traits to trigger
    cute.compile. The dummy tensors are GPU-resident and small; the
    overhead is dominated by cutlass.cute compilation, not kernel
    execution.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    runtime.assert_sm120("dsl12x.attention.prefill.warmup")

    for traits in traits_list:
        with _warmup_done_lock:
            key = traits.cache_key()
            if key in _warmup_done_for:
                continue
            _warmup_done_for.add(key)

        logger.info(
            "dsl12x prefill warmup: compiling kernel for %s", traits.describe()
        )

        # Construct dummy tensors that match the traits.
        num_tokens = traits.chunk_size
        n_kv = max(traits.topk_max * 2, 16)  # any value >= max index works
        q = torch.zeros(
            (num_tokens, traits.num_heads, traits.qk_head_dim),
            device=device, dtype=traits.dtype,
        )
        kv = torch.zeros(
            (n_kv, 1, traits.qk_head_dim),
            device=device, dtype=traits.dtype,
        )
        indices = torch.zeros(
            (num_tokens, 1, traits.topk_max),
            device=device, dtype=torch.int32,
        )
        topk_length = torch.full(
            (num_tokens,), traits.topk_max,
            device=device, dtype=torch.int32,
        )
        attn_sink = (
            torch.zeros((traits.num_heads,), device=device, dtype=torch.float32)
            if traits.has_attn_sink else None
        )

        try:
            run_sparse_mla_prefill(
                q=q,
                kv=kv,
                indices=indices,
                sm_scale=1.0,
                d_v=traits.v_head_dim,
                attn_sink=attn_sink,
                topk_length=topk_length,
                chunk_size=traits.chunk_size,
                active_heads=traits.num_heads,
            )
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001
            # Warmup failure is logged but does not raise -- the patcher's
            # try/except in the hot path will still catch failures at
            # serving time and fall back to the BMM bridge. We log so the
            # operator can see WHICH traits failed during startup.
            logger.warning(
                "dsl12x prefill warmup: traits %s FAILED to compile: %s. "
                "First user request matching this shape will fall back to "
                "the BMM bridge.",
                traits.describe(), exc,
            )


def env_enabled() -> bool:
    """True if DG_SM120_DSL12X_PREFILL is set to enable the dsl12x path."""
    return os.environ.get("DG_SM120_DSL12X_PREFILL", "0").lower() in (
        "1", "true", "yes", "on",
    )


def env_strict() -> bool:
    """True if DG_SM120_DSL12X_PREFILL_STRICT requires no fallback to BMM."""
    return os.environ.get("DG_SM120_DSL12X_PREFILL_STRICT", "0").lower() in (
        "1", "true", "yes", "on",
    )


__all__ = [
    "run_sparse_mla_prefill",
    "warmup",
    "env_enabled",
    "env_strict",
]
