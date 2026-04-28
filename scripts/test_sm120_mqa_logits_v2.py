#!/usr/bin/env python3
# scripts/test_sm120_mqa_logits_v2.py
#
# Isolated correctness check for the SM120 MQA logits v2 entry point.
#
# Two inner paths are exercised by default:
#
#   * "scalar" (C1):  bit-exact to deep_gemm::sm120_fallback::mqa_logits_kernel.
#                     Selected by setting DG_SM120_MQA_LOGITS_V2_MMA=0 around
#                     the call (the v2 entry point is default-on for MMA after
#                     C3, so we must explicitly opt out here to test scalar).
#   * "mma"    (C2a): BF16 m16n8k16 mma.sync tensor-core path. Selected by
#                     setting DG_SM120_MQA_LOGITS_V2_MMA=1 around the call.
#
# Both paths reduce FP32 internally and cast to the requested output dtype
# at the end, so the per-dtype tolerances are the same:
#
#   * FP32 output: max_diff < 5e-3 * sqrt(num_heads*head_dim / 2048)
#                  OR max_rel < 5e-4. The absolute gate scales with the
#                  per-output reduction depth because FP8-input + FP32
#                  reduction noise grows roughly as sqrt(reduction).
#                  At the C2a synthetic shape (H=32, D=64) the scale
#                  factor is 1.0; at the live C4 shape (H=64, D=128) it
#                  is 2.0.
#   * BF16 output: max_rel < 1.6e-2 (one BF16 ULP at the result magnitude).
#
# Use ``--paths scalar`` or ``--paths mma`` to test only one inner. Use
# ``--bench`` to add per-path microbench numbers (the headline speedup of
# C2a vs the C1 scalar / apis fallback).
#
# Usage:
#   docker compose exec -T vllm bash -lc \
#       'cd /workspace/DeepGEMM && python3 scripts/test_sm120_mqa_logits_v2.py'
#
# This script does NOT load vLLM or DeepSeek V4. It only loads the deep_gemm
# extension (must be rebuilt with the C1 sources) and performs a single
# kernel comparison per shape.

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from typing import Tuple

import torch

import deep_gemm
import deep_gemm._C as _C


# ---------------------------------------------------------------------------
# Env helpers: flip ``DG_SM120_MQA_LOGITS_V2_MMA`` per-call so we can A/B
# the scalar inner against the C2a MMA inner without restarting the process.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _env_var(name: str, value: str | None):
    """Temporarily set ``name=value`` in os.environ; restore on exit."""
    prev = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


# ---------------------------------------------------------------------------
# Synthetic input generator.
# ---------------------------------------------------------------------------


def _synthesize_inputs(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    *,
    compressed: bool,
    causal: bool,
    device: torch.device,
    seed: int,
) -> dict:
    """Build inputs that match the existing FP8 non-paged dispatch.

    Layouts mirror the production call site (vLLM DeepSeek V4 sparse
    indexer): Q is [seq_len, num_heads, head_dim] FP8 e4m3, KV is
    [seq_len_kv, head_dim] FP8 e4m3, kv_sf is [seq_len_kv] float, weights
    is [seq_len, num_heads] float.
    """
    torch.manual_seed(seed)

    # FP8 e4m3 directly from BF16 saturation cast keeps the values in a
    # representable range for the e4m3 dynamic range.
    q_bf16 = torch.randn(
        seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16
    ) * 0.5
    kv_bf16 = torch.randn(
        seq_len_kv, head_dim, device=device, dtype=torch.bfloat16
    ) * 0.5

    q_fp8 = q_bf16.to(torch.float8_e4m3fn).contiguous()
    kv_fp8 = kv_bf16.to(torch.float8_e4m3fn).contiguous()

    # Per-K-position scale: small positive values, like the sparse indexer
    # produces from per-token RMSNorm * dequant scale.
    kv_sf = (
        torch.rand(seq_len_kv, device=device, dtype=torch.float32) * 0.5 + 0.5
    ).contiguous()
    weights = (
        torch.rand(seq_len, num_heads, device=device, dtype=torch.float32) * 0.4
        + 0.1
    ).contiguous()

    # cu_seq_len_k_start / end define the per-query valid K range.
    if causal:
        # Each query t attends to K positions [0, min(t+1, seq_len_kv)).
        starts = torch.zeros(seq_len, device=device, dtype=torch.int32)
        ends = torch.arange(
            1, seq_len + 1, device=device, dtype=torch.int32
        ).clamp(max=seq_len_kv)
    else:
        # All queries see the same full K range.
        starts = torch.zeros(seq_len, device=device, dtype=torch.int32)
        ends = torch.full(
            (seq_len,), seq_len_kv, device=device, dtype=torch.int32
        )

    if compressed:
        # max_seqlen_k=0 means non-compressed (each row covers full kv).
        # Compressed mode means each row is packed at the start; need
        # max_seqlen_k = max(end - start).
        max_seqlen_k = int((ends - starts).max().item())
    else:
        max_seqlen_k = 0

    return {
        "q": q_fp8,
        "kv": kv_fp8,
        "kv_sf": kv_sf,
        "weights": weights,
        "cu_seq_len_k_start": starts,
        "cu_seq_len_k_end": ends,
        "max_seqlen_k": max_seqlen_k,
        "seq_len": seq_len,
        "seq_len_kv": seq_len_kv,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }


def _allocate_logits(
    *,
    seq_len: int,
    seq_len_kv: int,
    max_seqlen_k: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, int]:
    """Mirror the allocation in apis/attention.hpp::fp8_mqa_logits.

    The apis fallback tile is ``block_qh = 128`` queries-x-heads, so the
    aligned seq_len depends on ``num_heads``: the per-CTA Q tile holds
    ``block_q = block_qh / num_heads`` query rows. Hardcoding
    ``block_qh / 32`` worked for the C2a synthetic case (num_heads=32) but
    misaligned the buffer for the C4 live shape (num_heads=64).
    """

    def _align(x: int, a: int) -> int:
        return ((x + a - 1) // a) * a

    block_qh = 128
    block_kv = 256
    block_q = max(1, block_qh // num_heads)
    aligned_seq_len = _align(seq_len, block_q)
    if max_seqlen_k == 0:
        stride_logits = _align(seq_len_kv + block_kv, 8)
        full = torch.empty(
            (aligned_seq_len, stride_logits), device=device, dtype=dtype
        )
        view = full[:seq_len, :seq_len_kv]
    else:
        stride_logits = _align(max_seqlen_k, block_kv)
        full = torch.empty(
            (aligned_seq_len, stride_logits), device=device, dtype=dtype
        )
        view = full[:seq_len, :max_seqlen_k]
    return full, view, stride_logits


# ---------------------------------------------------------------------------
# Reference CPU compute (FP32, scalar). Used to bound both implementations.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _torch_reference(inputs: dict, *, dtype: torch.dtype) -> torch.Tensor:
    """Pure-torch reference matching the scalar fallback math."""
    q = inputs["q"].to(torch.float32)  # [S, H, D]
    kv = inputs["kv"].to(torch.float32)  # [Skv, D]
    kv_sf = inputs["kv_sf"]  # [Skv]
    weights = inputs["weights"]  # [S, H]
    starts = inputs["cu_seq_len_k_start"].to(torch.int64)
    ends = inputs["cu_seq_len_k_end"].to(torch.int64)
    seq_len = inputs["seq_len"]
    seq_len_kv = inputs["seq_len_kv"]
    max_seqlen_k = inputs["max_seqlen_k"]

    out_cols = max_seqlen_k if max_seqlen_k > 0 else seq_len_kv
    compressed = max_seqlen_k > 0

    # Scaled K: kv_scaled[n, d] = kv[n, d] * kv_sf[n]
    kv_scaled = kv * kv_sf.unsqueeze(-1)  # [Skv, D]

    # Per-head dot: dot[m, h, n] = sum_d q[m, h, d] * kv_scaled[n, d]
    # Equivalent to einsum 'mhd,nd->mhn'.
    dot = torch.einsum("mhd,nd->mhn", q, kv_scaled)  # [S, H, Skv]
    contrib = torch.clamp(dot, min=0.0) * weights.unsqueeze(-1)  # [S, H, Skv]
    full_logits = contrib.sum(dim=1)  # [S, Skv]

    out = torch.full(
        (seq_len, out_cols), float("-inf"), device=q.device, dtype=torch.float32
    )
    starts_cpu = starts.clamp(min=0).clamp(max=seq_len_kv)
    ends_cpu = ends.clamp(min=0).clamp(max=seq_len_kv)
    for m in range(seq_len):
        s = int(starts_cpu[m].item())
        e = int(ends_cpu[m].item())
        if compressed:
            for c in range(out_cols):
                n = s + c
                if n >= s and n < e and n < seq_len_kv:
                    out[m, c] = full_logits[m, n]
        else:
            for n in range(out_cols):
                if n >= s and n < e:
                    out[m, n] = full_logits[m, n]

    return out.to(dtype)


# ---------------------------------------------------------------------------
# Kernel invocations.
# ---------------------------------------------------------------------------


def _call_v2(inputs: dict, *, dtype: torch.dtype, device: torch.device,
             path: str = "scalar"):
    """Call the v2 entry point with the inner selected by ``path``.

    path = "scalar"  ->  set DG_SM120_MQA_LOGITS_V2_MMA=0 (C1 scalar inner).
                         Required because the v2 entry point is default-on for
                         MMA after C3; an unset env now selects MMA, so we
                         must explicitly opt out to exercise the scalar inner.
    path = "mma"     ->  set DG_SM120_MQA_LOGITS_V2_MMA=1 (C2a MMA inner).
                         Equivalent to leaving the env unset post-C3, but
                         explicit makes the test self-documenting.
    """
    if path not in ("scalar", "mma"):
        raise ValueError(f"path must be 'scalar' or 'mma', got {path!r}")
    full, view, stride_logits = _allocate_logits(
        seq_len=inputs["seq_len"],
        seq_len_kv=inputs["seq_len_kv"],
        max_seqlen_k=inputs["max_seqlen_k"],
        num_heads=inputs["num_heads"],
        device=device,
        dtype=dtype,
    )
    env_value = "1" if path == "mma" else "0"
    with _env_var("DG_SM120_MQA_LOGITS_V2_MMA", env_value):
        _C.sm120_fp8_mqa_logits_v2(
            q=inputs["q"],
            kv=inputs["kv"],
            kv_sf=inputs["kv_sf"],
            weights=inputs["weights"],
            cu_seq_len_k_start=inputs["cu_seq_len_k_start"],
            cu_seq_len_k_end=inputs["cu_seq_len_k_end"],
            logits=view,
            logits_dtype=dtype,
            seq_len=inputs["seq_len"],
            seq_len_kv=inputs["seq_len_kv"],
            max_seqlen_k=inputs["max_seqlen_k"],
            logits_stride=stride_logits,
            num_heads=inputs["num_heads"],
            head_dim=inputs["head_dim"],
        )
    return view.clone()


def _call_fallback_via_apis(inputs: dict):
    """Dispatch through deep_gemm._C.fp8_mqa_logits which routes to the
    SM120 fallback on this hardware. Output dtype is chosen by the api;
    return whatever it gives so the caller can compare appropriately.

    Pybind signature (verified):
        fp8_mqa_logits(q: Tensor,
                       kv: tuple[Tensor, Tensor],
                       weights: Tensor,
                       cu_seq_len_k_start: Tensor,
                       cu_seq_len_k_end: Tensor,
                       clean_logits: bool = True,
                       max_seqlen_k: int = 0) -> Tensor
    """
    return _C.fp8_mqa_logits(
        inputs["q"],
        (inputs["kv"], inputs["kv_sf"]),
        inputs["weights"],
        inputs["cu_seq_len_k_start"],
        inputs["cu_seq_len_k_end"],
        False,  # clean_logits
        inputs["max_seqlen_k"],
    )


# ---------------------------------------------------------------------------
# Per-shape comparison.
# ---------------------------------------------------------------------------


def _bench_us(fn, *, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _run_case(
    *,
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    compressed: bool,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    bench: bool,
    path: str = "scalar",
):
    inputs = _synthesize_inputs(
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        compressed=compressed,
        causal=causal,
        device=device,
        seed=seed,
    )

    # v2 kernel under test (path = "scalar" or "mma").
    out_v2 = _call_v2(inputs, dtype=dtype, device=device, path=path)

    # Try to also call the existing apis-bound dispatch as an additional
    # cross-check. On SM120 this hits sm120_fp8_mqa_logits_fallback. The
    # api returns a tensor in its own chosen dtype (FP32 in current
    # bindings); cast to ``dtype`` for diff. If the symbol or signature
    # is not what we expect, skip it cleanly: the torch reference below
    # is the authoritative comparison.
    out_fallback = None
    try:
        out_fallback_raw = _call_fallback_via_apis(inputs)
        # The api may return more columns than ``out_v2`` if
        # ``max_seqlen_k`` is 0 (non-compressed) and the api pads to
        # the next 256 boundary. Slice to the v2 view for comparison.
        if out_fallback_raw.shape != out_v2.shape:
            s, c = out_v2.shape
            if (
                out_fallback_raw.dim() == 2
                and out_fallback_raw.shape[0] >= s
                and out_fallback_raw.shape[1] >= c
            ):
                out_fallback_raw = out_fallback_raw[:s, :c]
        out_fallback = out_fallback_raw.to(dtype)
    except Exception as exc:  # noqa: BLE001
        # Truncate the pybind error tail — its signature dump is huge.
        msg = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"  [info] apis fp8_mqa_logits comparison skipped: {msg}")

    # Torch reference (FP32 in, cast to dtype). This is the math the v2
    # scalar inner implements directly.
    out_ref = _torch_reference(inputs, dtype=dtype)

    # Compare to torch reference.
    finite_mask = torch.isfinite(out_v2) & torch.isfinite(out_ref)
    if finite_mask.any():
        diff = (out_v2[finite_mask].to(torch.float32)
                - out_ref[finite_mask].to(torch.float32)).abs()
        ref_abs = out_ref[finite_mask].to(torch.float32).abs()
        v2_max = float(diff.max().item())
        v2_mean = float(diff.mean().item())
        v2_rel = float((diff / ref_abs.clamp(min=1e-6)).max().item())
    else:
        v2_max = v2_mean = v2_rel = 0.0

    # Compare to apis fallback if available.
    if out_fallback is not None and out_fallback.shape == out_v2.shape:
        finite_mask_fb = torch.isfinite(out_v2) & torch.isfinite(out_fallback)
        if finite_mask_fb.any():
            diff_fb = (out_v2[finite_mask_fb].to(torch.float32)
                       - out_fallback[finite_mask_fb].to(torch.float32)).abs()
            fb_max = float(diff_fb.max().item())
            fb_mean = float(diff_fb.mean().item())
        else:
            fb_max = fb_mean = 0.0
    else:
        fb_max = fb_mean = float("nan")

    # Mask alignment: -inf positions must match.
    mask_v2 = ~torch.isfinite(out_v2)
    mask_ref = ~torch.isfinite(out_ref)
    mask_match = torch.equal(mask_v2, mask_ref)

    print(
        f"  [{path}] shape S={seq_len}, Skv={seq_len_kv}, H={num_heads}, "
        f"D={head_dim}, compressed={compressed}, causal={causal}, dtype={dtype}"
    )
    print(
        f"    vs torch ref:     max_diff={v2_max:.3e}  mean_diff={v2_mean:.3e}  "
        f"max_rel={v2_rel:.3e}  mask_match={mask_match}"
    )
    if out_fallback is not None:
        print(
            f"    vs apis fallback: max_diff={fb_max:.3e}  mean_diff={fb_mean:.3e}"
        )

    # C1 pass criteria are gated on the output dtype because the kernel
    # downcasts the FP32 reduction result to ``dtype`` at the very end:
    #
    #   * FP32 output:  diff is FP8-input-quant + FP32-reduction-order
    #                   noise. The per-(m, n) output is a sum over
    #                   ``num_heads * head_dim`` terms (each a Q*K product
    #                   plus a head-weighted ReLU). For independent FP8
    #                   quant errors the absolute spread vs the BF16/FP32
    #                   torch reference grows roughly as
    #                   ``sqrt(num_heads * head_dim)``. The 5e-3 baseline
    #                   was calibrated for the C2a synthetic shape
    #                   (H=32, D=64, reduction=2048); for the C4 live
    #                   shape (H=64, D=128, reduction=8192) the same
    #                   per-element noise produces ~2x larger absolute
    #                   spread. Scale the absolute gate accordingly,
    #                   keeping it identical for the C2a synthetic shape.
    #                   We additionally pass if ``max_rel`` is tiny
    #                   (<= 5e-4); empirically the FP32-output cases land
    #                   at max_rel ~= 3e-4 even when the absolute spread
    #                   is at the per-element noise floor.
    #
    #   * BF16 output:  diff is dominated by the final
    #                   ``__float2bfloat16`` rounding (1 ULP at the
    #                   value's magnitude). For result magnitudes up to
    #                   ~16, 1 BF16 ULP is up to ~0.125 absolute.
    #                   Gate on relative error: BF16 machine epsilon is
    #                   ``2^-7 ~ 7.81e-3``; we accept a small headroom
    #                   for ULP rounding ties.
    #
    # mask_match must always hold: mismatched -inf positions are a real
    # bug regardless of output dtype.
    if dtype == torch.float32:
        # Scale the absolute gate with the per-output reduction depth.
        # baseline=2048 corresponds to the C2a synthetic shape
        # (H=32, D=64); for H=64, D=128 the scale factor is 2.0.
        reduction_baseline = 32 * 64  # 2048
        reduction = max(1, num_heads * head_dim)
        scale = math.sqrt(reduction / reduction_baseline)
        abs_gate = 5e-3 * max(1.0, scale)
        rel_gate = 5e-4
        abs_ok = v2_max < abs_gate
        rel_ok = v2_rel < rel_gate
        c1_ok = (abs_ok or rel_ok) and mask_match
        why = (
            f"max_diff={v2_max:.3e} < {abs_gate:.3e} "
            f"(scale={scale:.2f}) "
            f"OR max_rel={v2_rel:.3e} < {rel_gate:.0e}, "
            f"mask_match={mask_match}"
        )
    elif dtype == torch.bfloat16:
        # 2x BF16 epsilon caps gives headroom for one extra rounding
        # tie that the v2 path may resolve differently than the ref.
        c1_ok = (v2_rel < 1.6e-2) and mask_match and v2_mean < 5e-3
        why = (
            f"max_rel={v2_rel:.3e} < 1.6e-2, "
            f"mean_diff={v2_mean:.3e} < 5e-3, mask_match={mask_match}"
        )
    else:
        c1_ok = False
        why = f"unsupported dtype {dtype}"
    print(f"    C1 pass: {c1_ok} ({why})")

    if bench:
        v2_us = _bench_us(
            lambda: _call_v2(inputs, dtype=dtype, device=device, path=path)
        )
        print(f"    v2 [{path}] bench: {v2_us:.1f} us / call")
        if out_fallback is not None:
            try:
                fb_us = _bench_us(lambda: _call_fallback_via_apis(inputs))
                print(f"    apis fallback bench: {fb_us:.1f} us / call")
            except Exception:  # noqa: BLE001
                pass
    return c1_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true",
                        help="microbench v2 vs fallback (slow)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--paths",
        default="scalar,mma",
        help="comma-separated list of v2 inner paths to test "
             "(scalar | mma | scalar,mma). Default: both.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping")
        return 0

    device = torch.device("cuda")
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    for p in paths:
        if p not in ("scalar", "mma"):
            print(f"[error] unknown --paths value {p!r}; expected scalar/mma")
            return 1

    cases = [
        # (S, Skv, H, D, compressed, causal)
        # --- C2a synthetic shape (H=32, D=64): kept for regression coverage.
        (32, 256, 32, 64, False, False),       # tiny dense
        (32, 256, 32, 64, False, True),        # tiny causal
        (128, 1024, 32, 64, False, True),      # decode-shape causal
        (256, 4096, 32, 64, True, True),       # 4k compressed causal (uniform start)
        (64, 8192, 32, 64, False, False),      # wide kv
        # --- C4 live shape (H=64, D=128): the live DeepSeek V4 sparse
        # indexer dispatches with num_heads=64, head_dim=128. Trace data
        # confirmed prefill is (S=4096, Skv=1024) and decode is
        # (S=4, Skv=1025). The cases below cover both regimes plus a
        # compressed-mode variant; smaller seq_lens are used so the test
        # stays fast (the synthetic torch reference is O(S * Skv * H * D)).
        (32, 64, 64, 128, False, False),       # tiny dense (H=64, D=128)
        (32, 64, 64, 128, False, True),        # tiny causal
        (4, 1024, 64, 128, False, False),      # decode-shape (S=4, Skv=1024)
        (64, 256, 64, 128, False, True),       # short causal prefill
        (128, 256, 64, 128, True, True),       # compressed causal prefill
    ]

    overall_ok = True
    per_path_ok: dict[str, bool] = {}
    for path in paths:
        if path == "mma":
            print(
                "== SM120 MQA logits v2 (C2a MMA, "
                "DG_SM120_MQA_LOGITS_V2_MMA=1) correctness =="
            )
        else:
            print("== SM120 MQA logits v2 (C1 scalar) correctness ==")
        path_ok = True
        for shape in cases:
            S, Skv, H, D, compressed, causal = shape
            for dtype in (torch.float32, torch.bfloat16):
                ok = _run_case(
                    seq_len=S,
                    seq_len_kv=Skv,
                    num_heads=H,
                    head_dim=D,
                    compressed=compressed,
                    causal=causal,
                    dtype=dtype,
                    device=device,
                    seed=args.seed,
                    bench=args.bench,
                    path=path,
                )
                path_ok = path_ok and ok
        per_path_ok[path] = path_ok
        overall_ok = overall_ok and path_ok
        print()

    for path in paths:
        label = "C2a MMA" if path == "mma" else "C1 scalar"
        if per_path_ok[path]:
            print(f"[verdict] {label} PASSES synthetic correctness.")
        else:
            print(f"[verdict] {label} FAILED at least one case. Inspect "
                  "max_diff / max_rel / mask_match per shape above.")

    print(
        "          Per-dtype gates: FP32 (max_diff < 5e-3*sqrt(H*D/2048) "
        "OR max_rel < 5e-4), BF16 (max_rel < 1.6e-2 AND mean_diff < 5e-3), "
        "both with strict mask_match."
    )
    if "mma" in paths and per_path_ok.get("mma", False):
        print(
            "          C2a synthetic correctness validated; next step is the "
            "microbench (scripts/bench_sm120_mqa_logits_v2.py) at 4k/8k/16k "
            "prompt-shaped inputs to measure the kernel-level speedup vs the "
            "apis fallback before moving to live wire-up in C3."
        )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
