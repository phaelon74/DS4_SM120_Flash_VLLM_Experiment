#!/usr/bin/env python3
"""dsl12x sparse MLA prefill performance benchmark.

Compares three implementations on the same shape:

    * dsl12x:   The new dsl12x.attention.prefill.run_sparse_mla_prefill.
    * BMM:      The patcher's existing torch.bmm-based bridge from
                docker/patch_vllm_deepseekv4.py:1216.
    * scalar:   Pure-PyTorch per-token reference (slow, used as ground
                truth and as a floor for speedup measurements).

Each impl is run with cold-start measurement (first call after JIT cache
clear), warmup, then steady-state median over N iters.

Usage:

    python3 scripts/bench_dsl12x_sparse_prefill.py
    python3 scripts/bench_dsl12x_sparse_prefill.py \
        --num-tokens 256 \
        --num-heads 32 \
        --qk-head-dim 576 \
        --v-head-dim 512 \
        --topk 128 \
        --kv-rows 16384 \
        --has-attn-sink \
        --warmup 5 --iters 50

Output (one line per impl):

    dsl12x      cold=1234.5 ms  steady=  82.3 us  ratio_to_bmm=0.45x
    bmm         cold=  12.3 ms  steady= 184.1 us  ratio_to_bmm=1.00x
    scalar      cold=  45.6 ms  steady=12345.6 us  ratio_to_bmm=67.05x
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Callable, Optional, Tuple

# Bootstrap the workspace root (parent of scripts/) onto sys.path so the
# `dsl12x` package is importable regardless of where this script is run from.
_WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-tokens", type=int, default=256,
                        help="Q tokens (chunk size). Default 256 "
                             "(DG_SM120_PREFILL_WORKSPACE_CHUNK).")
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Active Q heads. Default 32.")
    parser.add_argument("--qk-head-dim", type=int, default=576,
                        help="qk_head_dim. Default 576.")
    parser.add_argument("--v-head-dim", type=int, default=512,
                        help="v_head_dim. Default 512.")
    parser.add_argument("--topk", type=int, default=128,
                        help="topk_max. Default 128.")
    parser.add_argument("--kv-rows", type=int, default=16384,
                        help="N_kv. Default 16384 (~16k context).")
    parser.add_argument("--has-attn-sink", action="store_true",
                        help="Pass an attn_sink tensor.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iters before steady-state measurement.")
    parser.add_argument("--iters", type=int, default=50,
                        help="Steady-state iters for median measurement.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed.")
    parser.add_argument("--skip-scalar", action="store_true",
                        help="Skip the scalar reference (slow at large shapes).")
    parser.add_argument("--skip-bmm", action="store_true",
                        help="Skip the BMM bridge.")
    parser.add_argument("--skip-dsl12x", action="store_true",
                        help="Skip dsl12x (useful when testing infra without "
                             "kernel implementation).")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    return parser.parse_args()


def make_inputs(args, device: str = "cuda"):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    torch.manual_seed(args.seed)
    sm_scale = 1.0 / math.sqrt(args.qk_head_dim)
    q = torch.randn(
        (args.num_tokens, args.num_heads, args.qk_head_dim),
        dtype=dtype, device=device,
    ) * 0.1
    kv = torch.randn(
        (args.kv_rows, 1, args.qk_head_dim),
        dtype=dtype, device=device,
    ) * 0.1
    indices = torch.randint(
        low=0, high=args.kv_rows,
        size=(args.num_tokens, 1, args.topk),
        device=device, dtype=torch.int32,
    )
    topk_length = torch.randint(
        low=max(1, args.topk // 4), high=args.topk + 1,
        size=(args.num_tokens,), device=device, dtype=torch.int32,
    )
    attn_sink = (
        torch.randn((args.num_heads,), dtype=torch.float32, device=device) * 0.5
        if args.has_attn_sink else None
    )
    return q, kv, indices, topk_length, attn_sink, sm_scale


def time_call(call: Callable, iters: int) -> float:
    """Median wall-clock per iteration in microseconds."""
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


def cold_call(call: Callable) -> float:
    """One-shot cold call wall-clock in milliseconds (includes JIT compile)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    call()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3


def run_dsl12x(q, kv, indices, sm_scale, d_v, attn_sink, topk_length):
    from dsl12x.attention.prefill import run_sparse_mla_prefill
    return run_sparse_mla_prefill(
        q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=d_v,
        attn_sink=attn_sink, topk_length=topk_length,
        chunk_size=q.shape[0],
    )


def run_bmm(q, kv, indices, sm_scale, d_v, attn_sink, topk_length):
    """BMM bridge equivalent of the patcher's
    _dg_sm120_prefill_torch_bmm at docker/patch_vllm_deepseekv4.py:1216."""
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    qk_dim = q.shape[-1]
    indices_2d = indices.squeeze(1)
    kv_2d = kv.squeeze(1)
    # Build the per-token gathered workspace (this is the gather that
    # dsl12x avoids).
    valid_mask = (indices_2d >= 0) & (indices_2d < kv_2d.shape[0])
    if topk_length is not None:
        positions = torch.arange(
            indices_2d.shape[1], device=indices_2d.device, dtype=indices_2d.dtype
        ).reshape(1, -1)
        valid_mask = valid_mask & (positions < topk_length.reshape(-1, 1))
    safe_indices = torch.where(valid_mask, indices_2d, torch.zeros_like(indices_2d))
    workspace = kv_2d[safe_indices]   # [num_tokens, topk, qk_dim]
    valid_any = valid_mask.any(dim=-1)
    # BMM: scores = Q @ workspace[..., :qk_dim].T
    q_chunk = q.transpose(0, 1).contiguous()  # [num_heads, num_tokens, qk_dim]
    # workspace is [num_tokens, topk, qk_dim]; we need to broadcast over heads.
    # For the simplified bench we do the per-token loop ourselves to mirror
    # the patcher's actual BMM shapes.
    out = torch.zeros((num_tokens, num_heads, d_v), dtype=q.dtype, device=q.device)
    lse = torch.full((num_tokens, num_heads), float("-inf"),
                     dtype=torch.float32, device=q.device)
    for token_idx in range(num_tokens):
        if not valid_any[token_idx].item():
            continue
        valid_for_token = valid_mask[token_idx]
        ws = workspace[token_idx][valid_for_token]
        if ws.shape[0] == 0:
            continue
        scores = (q[token_idx].float() @ ws[:, :qk_dim].float().T) * sm_scale
        per_head_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        token_out = probs @ ws[:, :d_v]
        if attn_sink is not None:
            gate = torch.sigmoid(per_head_lse - attn_sink.to(per_head_lse.dtype))
            token_out = token_out * gate.to(token_out.dtype).unsqueeze(-1)
        out[token_idx] = token_out
        lse[token_idx] = per_head_lse
    return out, lse


def run_scalar(q, kv, indices, sm_scale, d_v, attn_sink, topk_length):
    """Pure-PyTorch per-token scalar reference (matches test_dsl12x_sparse_prefill.py)."""
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    qk_dim = q.shape[-1]
    out = torch.zeros((num_tokens, num_heads, d_v), dtype=q.dtype, device=q.device)
    lse = torch.full((num_tokens, num_heads), float("-inf"),
                     dtype=torch.float32, device=q.device)
    indices_2d = indices.squeeze(1)
    kv_2d = kv.squeeze(1)
    kv_f = kv_2d.to(torch.float32)
    q_f = q.to(torch.float32)
    for token_idx in range(num_tokens):
        if topk_length is not None:
            tk_len = int(topk_length[token_idx].item())
        else:
            tk_len = indices_2d.shape[1]
        token_indices = indices_2d[token_idx, :tk_len]
        valid_mask = (token_indices >= 0) & (token_indices < kv_2d.shape[0])
        token_indices = token_indices[valid_mask]
        if token_indices.numel() == 0:
            continue
        selected = kv_f.index_select(0, token_indices)
        scores = (q_f[token_idx] @ selected[:, :qk_dim].T) * sm_scale
        per_head_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        token_out = probs @ selected[:, :d_v].to(q.dtype)
        if attn_sink is not None:
            gate = torch.sigmoid(per_head_lse - attn_sink.to(per_head_lse.dtype))
            token_out = token_out * gate.to(token_out.dtype).unsqueeze(-1)
        out[token_idx] = token_out
        lse[token_idx] = per_head_lse
    return out, lse


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    cap = torch.cuda.get_device_capability()
    if cap[0] != 12:
        print(f"WARN: dsl12x targets SM120, device is {cap[0]}.{cap[1]}", file=sys.stderr)

    q, kv, indices, topk_length, attn_sink, sm_scale = make_inputs(args)
    print(
        f"shape: tokens={args.num_tokens} heads={args.num_heads} "
        f"qk_dim={args.qk_head_dim} v_dim={args.v_head_dim} "
        f"topk={args.topk} kv_rows={args.kv_rows} "
        f"dtype={args.dtype} has_attn_sink={args.has_attn_sink}\n"
    )

    bmm_steady = None

    impls: list[Tuple[str, Callable[[], None], bool]] = []

    if not args.skip_dsl12x:
        impls.append((
            "dsl12x",
            lambda: run_dsl12x(q, kv, indices, sm_scale, args.v_head_dim, attn_sink, topk_length),
            False,
        ))
    if not args.skip_bmm:
        impls.append((
            "bmm",
            lambda: run_bmm(q, kv, indices, sm_scale, args.v_head_dim, attn_sink, topk_length),
            False,
        ))
    if not args.skip_scalar:
        impls.append((
            "scalar",
            lambda: run_scalar(q, kv, indices, sm_scale, args.v_head_dim, attn_sink, topk_length),
            False,
        ))

    results = []
    for name, fn, _ in impls:
        try:
            cold_ms = cold_call(fn)
            for _ in range(args.warmup):
                fn()
            steady_us = time_call(fn, args.iters)
        except NotImplementedError as e:
            print(f"{name:<10}  SCAFFOLD: {e}")
            results.append((name, None, None))
            continue
        except Exception as e:
            print(f"{name:<10}  ERROR: {type(e).__name__}: {e}")
            results.append((name, None, None))
            continue
        results.append((name, cold_ms, steady_us))
        if name == "bmm":
            bmm_steady = steady_us

    print("\nResults (median over {} steady-state iters):".format(args.iters))
    print(f"{'impl':<10}  {'cold':>12}  {'steady':>12}  {'ratio_to_bmm':>14}")
    for name, cold_ms, steady_us in results:
        if cold_ms is None:
            print(f"{name:<10}  {'(skip)':>12}  {'(skip)':>12}  {'':>14}")
            continue
        if bmm_steady is not None and bmm_steady > 0:
            ratio = steady_us / bmm_steady
            ratio_str = f"{ratio:>10.2f}x"
        else:
            ratio_str = "n/a"
        print(
            f"{name:<10}  {cold_ms:>10.1f} ms  {steady_us:>10.2f} us  {ratio_str:>14}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
