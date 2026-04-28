#!/usr/bin/env python3
# scripts/bench_sm120_mqa_logits_v2.py
#
# Microbench the SM120 MQA logits v2 kernel against the apis fallback
# (sm120_fp8_mqa_logits_fallback, the existing scalar non-paged path) and
# against the v2 scalar inner. Reports per-shape per-dtype timings and the
# speedup of the C2a MMA inner over the scalar baseline.
#
# Live profile attribution (rank 0 PyTorch trace, single request):
#   prompt 4k:   961 ms / 42 calls (= 22.9 ms/call) -> 44% of TTFT
#   prompt 8k:  3647 ms / 63 calls (= 57.9 ms/call) -> 62% of TTFT
#   prompt 16k: 13963 ms /105 calls (=133.0 ms/call)-> 76% of TTFT
#
# C2a target (sequential-over-H, BF16 m16n8k16): ~150x kernel speedup,
# enough to drop the 16k call from 133 ms to roughly ~1 ms. C2b adds the
# parallel head split if C2a underperforms.
#
# This script does NOT load vLLM or DeepSeek V4. It only loads the
# deep_gemm extension and runs the kernel against synthetic inputs whose
# shape matches the live sparse indexer dispatch:
#   q       : [seq_len, num_heads=32, head_dim=64] FP8 e4m3
#   kv      : [seq_len_kv, head_dim=64]            FP8 e4m3
#   kv_sf   : [seq_len_kv]                          float32
#   weights : [seq_len, num_heads]                  float32
#   cu_seq_len_k_start / cu_seq_len_k_end           int32
#
# Live shape pattern: seq_len = num_prefill_tokens (= prompt length on the
# rank), seq_len_kv = same, max_seqlen_k = 0 (non-compressed), causal mask
# (cu_seq_len_k_end[m] = m+1, cu_seq_len_k_start[m] = 0).
#
# Usage:
#   docker compose exec -T vllm bash -lc \
#       'cd /workspace/DeepGEMM && python3 scripts/bench_sm120_mqa_logits_v2.py'
#
#   # Add --shapes to bench specific prompt sizes (default 1k/4k/8k/16k):
#   docker compose exec -T vllm bash -lc \
#       'cd /workspace/DeepGEMM && python3 scripts/bench_sm120_mqa_logits_v2.py \
#           --shapes 1024,4096,8192'
#
#   # Add --skip-fallback if the apis call signature mismatches; the v2
#   # scalar already provides a baseline.
#
# Typical output line per (S, dtype):
#   S=4096  dtype=bf16  scalar=22341 us  mma=  198 us  speedup=112.8x

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

import deep_gemm
import deep_gemm._C as _C


# ---------------------------------------------------------------------------
# Env helper (mirrors test_sm120_mqa_logits_v2.py).
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _env_var(name: str, value: str | None):
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
# Live-shape input synthesizer. Mirrors the production sparse indexer call
# pattern exactly (causal full-prefix, max_seqlen_k=0, dtype torch.bfloat16).
# ---------------------------------------------------------------------------


def _synthesize_live_shape(
    seq_len: int,
    *,
    num_heads: int = 32,
    head_dim: int = 64,
    device: torch.device,
    seed: int = 1234,
) -> dict:
    torch.manual_seed(seed)
    seq_len_kv = seq_len  # live: each prompt contributes its own KV

    q_bf16 = torch.randn(
        seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16
    ) * 0.5
    kv_bf16 = torch.randn(
        seq_len_kv, head_dim, device=device, dtype=torch.bfloat16
    ) * 0.5
    q = q_bf16.to(torch.float8_e4m3fn).contiguous()
    kv = kv_bf16.to(torch.float8_e4m3fn).contiguous()
    kv_sf = (
        torch.rand(seq_len_kv, device=device, dtype=torch.float32) * 0.5 + 0.5
    ).contiguous()
    weights = (
        torch.rand(seq_len, num_heads, device=device, dtype=torch.float32) * 0.4
        + 0.1
    ).contiguous()

    starts = torch.zeros(seq_len, device=device, dtype=torch.int32)
    ends = torch.arange(
        1, seq_len + 1, device=device, dtype=torch.int32
    ).clamp(max=seq_len_kv)

    return {
        "q": q,
        "kv": kv,
        "kv_sf": kv_sf,
        "weights": weights,
        "cu_seq_len_k_start": starts,
        "cu_seq_len_k_end": ends,
        "seq_len": seq_len,
        "seq_len_kv": seq_len_kv,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "max_seqlen_k": 0,  # non-compressed
    }


def _allocate_logits(
    *,
    seq_len: int,
    seq_len_kv: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Mirror ``apis/attention.hpp::fp8_mqa_logits`` allocation for the
    non-compressed case (max_seqlen_k == 0)."""

    def _align(x: int, a: int) -> int:
        return ((x + a - 1) // a) * a

    block_qh = 128
    block_kv = 256
    aligned_seq_len = _align(seq_len, block_qh // 32)  # num_heads=32
    stride_logits = _align(seq_len_kv + block_kv, 8)
    full = torch.empty(
        (aligned_seq_len, stride_logits), device=device, dtype=dtype
    )
    view = full[:seq_len, :seq_len_kv]
    return full, view, stride_logits


# ---------------------------------------------------------------------------
# Kernel callers. ``_call_v2(path=...)`` selects the v2 inner via env.
# ---------------------------------------------------------------------------


def _call_v2(inputs: dict, *, dtype: torch.dtype, device: torch.device,
             path: str) -> torch.Tensor:
    if path not in ("scalar", "mma"):
        raise ValueError(path)
    _, view, stride = _allocate_logits(
        seq_len=inputs["seq_len"], seq_len_kv=inputs["seq_len_kv"],
        device=device, dtype=dtype,
    )
    env_value = "1" if path == "mma" else None
    with _env_var("DG_SM120_MQA_LOGITS_V2_MMA", env_value):
        _C.sm120_fp8_mqa_logits_v2(
            q=inputs["q"], kv=inputs["kv"], kv_sf=inputs["kv_sf"],
            weights=inputs["weights"],
            cu_seq_len_k_start=inputs["cu_seq_len_k_start"],
            cu_seq_len_k_end=inputs["cu_seq_len_k_end"],
            logits=view, logits_dtype=dtype,
            seq_len=inputs["seq_len"], seq_len_kv=inputs["seq_len_kv"],
            max_seqlen_k=inputs["max_seqlen_k"], logits_stride=stride,
            num_heads=inputs["num_heads"], head_dim=inputs["head_dim"],
        )
    return view


def _call_apis_fallback(inputs: dict) -> torch.Tensor:
    """Existing apis-bound dispatch -> sm120_fp8_mqa_logits_fallback on SM120.

    Pybind: fp8_mqa_logits(q, (kv, kv_sf), weights, starts, ends,
                           clean_logits=True, max_seqlen_k=0)
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
# Bench loop with CUDA events for accuracy (perf_counter is too noisy at
# the ~100 us level the MMA path is targeting).
# ---------------------------------------------------------------------------


def _bench_us(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(stop)
    return elapsed_ms / iters * 1000.0  # ms -> us


def _format_us(us: float) -> str:
    if us < 1000:
        return f"{us:7.1f} us"
    if us < 1_000_000:
        return f"{us / 1000:7.2f} ms"
    return f"{us / 1_000_000:7.2f} s "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="1024,4096,8192,16384",
        help="comma-separated prompt sizes (= seq_len = seq_len_kv)",
    )
    parser.add_argument(
        "--dtypes",
        default="float32,bfloat16",
        help="comma-separated logits dtypes (float32, bfloat16)",
    )
    parser.add_argument(
        "--paths",
        default="scalar,mma",
        help="v2 inner paths to bench (subset of scalar,mma). "
             "Default: both, plus the apis fallback as a third reference.",
    )
    parser.add_argument(
        "--skip-fallback", action="store_true",
        help="skip the apis fp8_mqa_logits comparison",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping")
        return 0

    device = torch.device("cuda")
    shapes = [int(s) for s in args.shapes.split(",") if s.strip()]
    dtype_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    dtypes = [dtype_map[d.strip()] for d in args.dtypes.split(",") if d.strip()]
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    for p in paths:
        if p not in ("scalar", "mma"):
            print(f"[error] unknown path {p!r}")
            return 1

    print("== SM120 MQA logits v2 microbench ==")
    print(
        f"   shapes (S=Skv): {shapes}\n"
        f"   dtypes:         {[str(d) for d in dtypes]}\n"
        f"   paths:          {paths}"
        + ("" if args.skip_fallback else " + apis fallback")
    )
    print()

    for S in shapes:
        # Pre-synthesize inputs once per shape (reuse across paths/dtypes).
        inputs = _synthesize_live_shape(
            S, device=device, seed=args.seed
        )
        for dtype in dtypes:
            timings: Dict[str, float] = {}
            for path in paths:
                fn = lambda p=path: _call_v2(  # noqa: E731
                    inputs, dtype=dtype, device=device, path=p,
                )
                # Warmup outside the bench loop to avoid double-warm.
                try:
                    timings[path] = _bench_us(
                        fn, warmup=args.warmup, iters=args.iters
                    )
                except RuntimeError as exc:
                    print(f"  S={S}  dtype={dtype}  path={path}  ERROR: {exc}")
                    timings[path] = float("nan")
            if not args.skip_fallback:
                try:
                    fb_us = _bench_us(
                        lambda: _call_apis_fallback(inputs),
                        warmup=args.warmup, iters=args.iters,
                    )
                    timings["apis_fallback"] = fb_us
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc).splitlines()[0]
                    print(f"  [info] apis fallback skipped: {msg}")
                    timings["apis_fallback"] = float("nan")

            # Format the per-shape line.
            parts = [f"S={S:>5d}", f"dtype={str(dtype).split('.')[-1]:>8s}"]
            for label, us in timings.items():
                parts.append(f"{label}={_format_us(us)}")

            # Speedup vs the slowest available baseline (apis fallback if
            # available, otherwise scalar).
            baseline = (
                timings.get("apis_fallback")
                if (not args.skip_fallback
                    and timings.get("apis_fallback") == timings.get("apis_fallback"))  # not nan
                else timings.get("scalar")
            )
            mma_us = timings.get("mma")
            if (
                baseline is not None and mma_us is not None
                and baseline == baseline and mma_us == mma_us  # not nan
                and mma_us > 0
            ):
                parts.append(f"speedup={baseline / mma_us:6.2f}x")
            print("  " + "  ".join(parts))
        print()

    print(
        "[note] CUDA-event timed; warmup={} iters={}. The mma path is the\n"
        "       C2a sequential-over-H BF16 m16n8k16 inner; if it lands\n"
        "       under ~1 ms at S=16k the kernel-level target is hit and\n"
        "       C3 (live wire-up) is the next step. Otherwise C2b adds the\n"
        "       parallel head split.".format(args.warmup, args.iters)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
