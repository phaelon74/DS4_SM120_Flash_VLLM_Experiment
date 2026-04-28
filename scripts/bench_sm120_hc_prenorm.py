"""Microbenchmark for the SM120 HyperConnection prenorm GEMM v2.

Compares the new tiled unified-kernel implementation (default-on via
``DG_SM120_HC_PRENORM_V2=1``) against the original scalar fallback
(``DG_SM120_HC_PRENORM_V2=0``). Both should be bit-exact within fp32 reduction
order.

Usage::

    docker compose exec -T vllm bash -lc \
      'cd /workspace/DeepGEMM && python3 scripts/bench_sm120_hc_prenorm.py'
"""

from __future__ import annotations

import argparse
import os
import time

import torch


def _bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--k", type=int, default=7168)
    p.add_argument("--num-splits", type=int, default=1)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    args = p.parse_args()

    import deep_gemm

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((args.n, args.k), dtype=torch.float32, device="cuda")
    if args.num_splits == 1:
        d = torch.empty((args.m, args.n), dtype=torch.float32, device="cuda")
        s = torch.empty((args.m,), dtype=torch.float32, device="cuda")
    else:
        d = torch.empty(
            (args.num_splits, args.m, args.n), dtype=torch.float32, device="cuda"
        )
        s = torch.empty(
            (args.num_splits, args.m), dtype=torch.float32, device="cuda"
        )

    def run():
        deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, args.num_splits)

    print(f"shape m={args.m} n={args.n} k={args.k} num_splits={args.num_splits}")

    # v2 scalar kernel (production default).
    os.environ["DG_SM120_HC_PRENORM_V2"] = "1"
    os.environ["DG_SM120_HC_PRENORM_V2_MMA"] = "0"
    us_v2 = _bench(run, args.iters, args.warmup)
    d_v2 = d.clone()
    s_v2 = s.clone()
    print(f"  v2 scalar  : {us_v2:8.3f} us")

    # v2 MMA path (dsl12x Phase 5 scaffold). When the kernel scaffold
    # returns sentinel -inf (current state), we report the dispatch
    # latency but not bit-exactness; once the kernel inner is filled
    # in, this path should both correctness-match v2 scalar and be
    # faster.
    os.environ["DG_SM120_HC_PRENORM_V2_MMA"] = "1"
    us_v2_mma = _bench(run, args.iters, args.warmup)
    d_v2_mma = d.clone()
    s_v2_mma = s.clone()
    print(f"  v2 MMA     : {us_v2_mma:8.3f} us  (scaffold; -inf if not implemented)")

    # v1 fallback (the original scalar kernel kept for diff comparison).
    os.environ["DG_SM120_HC_PRENORM_V2"] = "0"
    os.environ["DG_SM120_HC_PRENORM_V2_MMA"] = "0"
    us_v1 = _bench(run, args.iters, args.warmup)
    d_v1 = d.clone()
    s_v1 = s.clone()
    print(f"  v1 fallback: {us_v1:8.3f} us")
    print(f"  v1 vs v2 scalar speedup: {us_v1 / max(us_v2, 1e-6):5.2f}x")
    if not bool(torch.isinf(d_v2_mma).any().item()):
        print(f"  v2 scalar vs v2 MMA speedup: {us_v2 / max(us_v2_mma, 1e-6):5.2f}x")

    diff_d = (d_v1 - d_v2).abs().max().item()
    diff_s = (s_v1 - s_v2).abs().max().item()
    print(f"  v1 vs v2 scalar:  d max abs diff = {diff_d:.6e}, s max abs diff = {diff_s:.6e}")
    if not bool(torch.isinf(d_v2_mma).any().item()):
        diff_d_mma = (d_v2 - d_v2_mma).abs().max().item()
        diff_s_mma = (s_v2 - s_v2_mma).abs().max().item()
        print(f"  v2 scalar vs v2 MMA: d max abs diff = {diff_d_mma:.6e}, s max abs diff = {diff_s_mma:.6e}")
    else:
        print(
            "  v2 MMA: scaffold sentinel detected -- kernel inner not yet "
            "implemented (see csrc/sm120_tf32_hc_prenorm_gemm.cu "
            "hc_prenorm_mma_kernel_scaffold)"
        )


if __name__ == "__main__":
    main()
