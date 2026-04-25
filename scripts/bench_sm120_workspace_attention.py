#!/usr/bin/env python3
import argparse
import os
import time

import torch

import deep_gemm


def make_case(batch: int, heads: int, topk: int, dtype: torch.dtype, device: str):
    torch.manual_seed(20260425)
    q = torch.randn((batch, 1, heads, 512), device=device, dtype=dtype) * 0.05
    kv = torch.randn((batch, topk, 512), device=device, dtype=dtype) * 0.05
    # Use int64 because vLLM often carries topk_length as int64 metadata.
    lens = torch.full((batch,), topk, device=device, dtype=torch.int64)
    sink = torch.zeros((heads,), device=device, dtype=torch.float32)
    out = torch.empty((batch, 1, heads, 512), device=device, dtype=dtype)
    return q.contiguous(), kv.contiguous(), lens, sink, out


def call_fused(case, main_topk: int, softmax_scale: float):
    q, kv, lens, sink, out = case
    return deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace(
        q, kv, lens, None, sink, main_topk, 0, 512, softmax_scale, out
    )


def call_split(case, main_topk: int, softmax_scale: float):
    q, kv, lens, sink, out = case
    return deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace_split(
        q, kv, lens, None, sink, main_topk, 0, 512, softmax_scale, out
    )


def call_torch(case, softmax_scale: float):
    q, kv, lens, sink, out = case
    batch, _, heads, _ = q.shape
    q_decode = q[:, 0, :, :]
    scores = torch.matmul(q_decode, kv.transpose(1, 2))
    scores = scores.to(torch.float32).mul_(float(softmax_scale))
    pos = torch.arange(kv.shape[1], device=q.device)
    valid = pos.unsqueeze(0) < lens.reshape(-1, 1)
    scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
    row_lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    attn_out = torch.matmul(probs, kv)
    gate = torch.sigmoid(row_lse - sink.reshape(1, heads))
    attn_out = attn_out * gate.to(attn_out.dtype).unsqueeze(-1)
    out.copy_(attn_out.unsqueeze(1))
    return out, row_lse.unsqueeze(-1)


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
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dtype = torch.bfloat16
    scale = 512.0**-0.5
    case_fused = make_case(args.batch, args.heads, args.topk, dtype, args.device)
    case_torch = (
        case_fused[0],
        case_fused[1],
        case_fused[2],
        case_fused[3],
        torch.empty_like(case_fused[4]),
    )

    out_fused, lse_fused = call_fused(case_fused, args.topk, scale)
    out_split, lse_split = call_split(case_fused, args.topk, scale)
    out_torch, lse_torch = call_torch(case_torch, scale)
    torch.cuda.synchronize()

    out_diff = (out_fused.float() - out_torch.float()).abs()
    lse_diff = (lse_fused.float() - lse_torch.float()).abs()
    split_out_diff = (out_split.float() - out_torch.float()).abs()
    split_lse_diff = (lse_split.float() - lse_torch.float()).abs()
    print(f"shape: batch={args.batch} heads={args.heads} topk={args.topk}")
    print(f"out_max_abs: {out_diff.max().item():.6g}")
    print(f"out_mean_abs: {out_diff.mean().item():.6g}")
    print(f"lse_max_abs: {lse_diff.max().item():.6g}")
    print(f"split_out_max_abs: {split_out_diff.max().item():.6g}")
    print(f"split_out_mean_abs: {split_out_diff.mean().item():.6g}")
    print(f"split_lse_max_abs: {split_lse_diff.max().item():.6g}")
    print(
        "fused_us: "
        f"{bench(lambda: call_fused(case_fused, args.topk, scale), args.warmup, args.iters):.3f}"
    )
    print(
        "split_us: "
        f"{bench(lambda: call_split(case_fused, args.topk, scale), args.warmup, args.iters):.3f}"
    )
    print(
        "torch_us: "
        f"{bench(lambda: call_torch(case_torch, scale), args.warmup, args.iters):.3f}"
    )


if __name__ == "__main__":
    main()
