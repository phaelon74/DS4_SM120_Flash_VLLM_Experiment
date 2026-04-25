#!/usr/bin/env python3
import argparse
import time

import torch

import vllm.model_executor.layers.mhc  # noqa: F401


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(1234)
    residual = torch.randn(
        (args.tokens, args.hc_mult, args.hidden_size),
        device=args.device,
        dtype=torch.bfloat16,
    )
    fn = torch.randn(
        (args.hc_mult * (2 + args.hc_mult), args.hc_mult * args.hidden_size),
        device=args.device,
        dtype=torch.float32,
    )
    hc_scale = torch.ones((3,), device=args.device, dtype=torch.float32)
    hc_base = torch.zeros(
        (args.hc_mult * (2 + args.hc_mult),),
        device=args.device,
        dtype=torch.float32,
    )

    def pre_call():
        return torch.ops.vllm.mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            args.sinkhorn_iters,
        )

    post, comb, x = pre_call()

    def post_call():
        return torch.ops.vllm.mhc_post(x, residual, post, comb)

    pre_us = bench(pre_call, args.warmup, args.iters)
    post_us = bench(post_call, args.warmup, args.iters)
    print(
        f"tokens={args.tokens} hidden={args.hidden_size} hc_mult={args.hc_mult} "
        f"sinkhorn={args.sinkhorn_iters}"
    )
    print(f"mhc_pre_us: {pre_us:.3f}")
    print(f"mhc_post_us: {post_us:.3f}")
    print(f"mhc_total_two_blocks_us: {(2 * (pre_us + post_us)):.3f}")


if __name__ == "__main__":
    main()
