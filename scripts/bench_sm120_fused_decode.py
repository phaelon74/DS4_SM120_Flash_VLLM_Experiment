"""Microbenchmark for SM120 fused sparse MLA decode (v2).

Measures the per-call latency of:
  * v2 fused decode  (FP8 cache direct, online softmax, no BF16 workspace)
  * gather + workspace-split decode (current default path)

so the structural-fusion win can be measured independently of any MMA upgrade.

Usage::

    docker compose exec -T vllm bash -lc \
      'cd /workspace/DeepGEMM && python3 scripts/bench_sm120_fused_decode.py'
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from test_sm120_fused_decode import _make_fp8_ds_mla_cache  # noqa: E402


def _bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1e6  # microseconds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=64)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    args = p.parse_args()

    import deep_gemm

    dg_c = deep_gemm._C

    head_dim = 512
    softmax_scale = 1.0 / math.sqrt(head_dim)

    cache = _make_fp8_ds_mla_cache(
        args.num_blocks, args.block_size, head_dim=head_dim, seed=0
    )

    q = torch.randn(
        (args.batch, args.heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    indices = torch.randint(
        0,
        args.num_blocks * args.block_size,
        (args.batch, 1, args.topk),
        device="cuda",
        dtype=torch.int32,
    )
    topk_length = torch.full(
        (args.batch,), args.topk, device="cuda", dtype=torch.int32
    )
    out = torch.empty(
        (args.batch, args.heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    workspace = torch.empty(
        (args.batch, args.topk, head_dim), device="cuda", dtype=torch.bfloat16
    )
    out_ref = torch.empty_like(out)

    def run_v2():
        dg_c.sm120_sparse_mla_decode_v2(
            q, cache, indices, topk_length, None, head_dim, softmax_scale,
            args.block_size, out,
        )

    def run_ref():
        dg_c.sm120_dequantize_and_gather_indexed_k_cache(
            workspace, cache, indices, topk_length, args.block_size, 0
        )
        dg_c.sm120_sparse_mla_decode_from_bf16_workspace_split(
            q, workspace, topk_length, None, None, args.topk, 0, head_dim,
            softmax_scale, out_ref,
        )

    us_ref = _bench(run_ref, args.iters, args.warmup)
    us_v2 = _bench(run_v2, args.iters, args.warmup)

    print(
        f"batch={args.batch} heads={args.heads} topk={args.topk} block={args.block_size}"
    )
    print(f"  gather+split bridge : {us_ref:8.3f} us")
    print(f"  fused v2            : {us_v2:8.3f} us")
    print(f"  speedup             : {us_ref / max(us_v2, 1e-6):5.2f}x")


if __name__ == "__main__":
    main()
