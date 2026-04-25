#!/usr/bin/env python3
import argparse
import os
import time

import torch

import deep_gemm


def make_case(batch: int, heads: int, d: int, r: int, device: str):
    assert d % 128 == 0
    assert r % 128 == 0
    torch.manual_seed(20260425)
    a = (torch.randn((batch, heads, r), device=device, dtype=torch.float32) * 0.25).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn((heads, d, r), device=device, dtype=torch.float32) * 0.25).to(
        torch.float8_e4m3fn
    )
    packed_r = (r // 128 + 3) // 4
    a_scale = torch.full(
        (batch, heads, packed_r), 0x7F7F7F7F, device=device, dtype=torch.int32
    )
    b_scale = torch.full(
        (heads, d // 128, packed_r), 0x7F7F7F7F, device=device, dtype=torch.int32
    )
    return (
        a.contiguous(),
        a_scale.contiguous(),
        b.contiguous(),
        b_scale.contiguous(),
    )


def call_kernel(case, out: torch.Tensor, d_per_block: int) -> None:
    os.environ["DG_SM120_BHR_D_PER_BLOCK"] = str(d_per_block)
    a, a_scale, b, b_scale = case
    deep_gemm._C.sm120_fp8_bhr_hdr_bhd(a, a_scale, b, b_scale, out)


def bench(case, out: torch.Tensor, d_per_block: int, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        call_kernel(case, out, d_per_block)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        call_kernel(case, out, d_per_block)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d", type=int, default=1024)
    parser.add_argument("--r", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", default="1,2,4,8,16")
    args = parser.parse_args()

    case = make_case(args.batch, args.heads, args.d, args.r, args.device)
    ref = torch.empty(
        (args.batch, args.heads, args.d), device=args.device, dtype=torch.bfloat16
    )
    call_kernel(case, ref, 1)
    torch.cuda.synchronize()

    print(
        f"shape: B={args.batch} H={args.heads} D={args.d} R={args.r} "
        f"output={tuple(ref.shape)}"
    )
    for variant in [int(v) for v in args.variants.split(",") if v.strip()]:
        out = torch.empty_like(ref)
        call_kernel(case, out, variant)
        torch.cuda.synchronize()
        diff = (ref.float() - out.float()).abs()
        ref_abs = ref.float().abs().clamp_min(1e-6)
        usec = bench(case, out, variant, args.warmup, args.iters)
        print(
            f"d_per_block={variant:<2d} usec={usec:8.3f} "
            f"max_abs={diff.max().item():.6g} "
            f"mean_abs={diff.mean().item():.6g} "
            f"max_rel={(diff / ref_abs).max().item():.6g}"
        )


if __name__ == "__main__":
    main()
