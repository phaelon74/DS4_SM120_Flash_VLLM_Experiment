#!/usr/bin/env python3
import argparse
import time

import torch

import deep_gemm


K_HEAD_DIM = 512
K_FP8_DIM = 448
K_BF16_DIM = 64
K_TOKEN_BYTES = K_FP8_DIM + K_BF16_DIM * 2
K_SCALE_BYTES = 8


def make_packed_cache(
    num_blocks: int,
    block_size: int,
    device: str,
) -> torch.Tensor:
    torch.manual_seed(20260425)

    fp8_values = (
        torch.randn((num_blocks, block_size, K_FP8_DIM), device=device, dtype=torch.float32)
        * 0.05
    ).to(torch.float8_e4m3fn)
    fp8_bytes = fp8_values.view(torch.uint8)
    rope = (
        torch.randn((num_blocks, block_size, K_BF16_DIM), device=device, dtype=torch.float32)
        * 0.05
    ).to(torch.bfloat16)
    rope_bytes = rope.view(torch.uint8).reshape(num_blocks, block_size, K_BF16_DIM * 2)

    # UE8M0 scale byte 0x7f corresponds to scale 1.0 in the extension.
    scales = torch.full(
        (num_blocks, block_size, K_SCALE_BYTES),
        0x7F,
        device=device,
        dtype=torch.uint8,
    )

    token_data = torch.cat((fp8_bytes, rope_bytes), dim=-1).contiguous()
    raw = torch.empty(
        (num_blocks, block_size * (K_TOKEN_BYTES + K_SCALE_BYTES)),
        device=device,
        dtype=torch.uint8,
    )
    raw[:, : block_size * K_TOKEN_BYTES] = token_data.reshape(
        num_blocks, block_size * K_TOKEN_BYTES
    )
    raw[:, block_size * K_TOKEN_BYTES :] = scales.reshape(
        num_blocks, block_size * K_SCALE_BYTES
    )
    return raw.view(num_blocks, block_size, 1, K_TOKEN_BYTES + K_SCALE_BYTES)


def make_case(
    batch: int,
    heads: int,
    topk: int,
    num_blocks: int,
    block_size: int,
    device: str,
):
    q = (
        torch.randn((batch, 1, heads, K_HEAD_DIM), device=device, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)
    k_cache = make_packed_cache(num_blocks, block_size, device)
    flat_tokens = num_blocks * block_size
    indices = torch.randint(
        0,
        flat_tokens,
        (batch, 1, topk),
        device=device,
        dtype=torch.int32,
    )
    topk_length = torch.full((batch,), topk, device=device, dtype=torch.int64)
    sink = torch.zeros((heads,), device=device, dtype=torch.float32)
    workspace = torch.empty((batch, topk, K_HEAD_DIM), device=device, dtype=torch.bfloat16)
    return q.contiguous(), k_cache, indices.contiguous(), topk_length, sink, workspace


def call_direct(case, softmax_scale: float):
    q, k_cache, indices, topk_length, sink, _workspace = case
    return deep_gemm._C.sm120_sparse_mla_decode(
        q,
        k_cache,
        indices,
        topk_length,
        sink,
        None,
        None,
        None,
        K_HEAD_DIM,
        softmax_scale,
        None,
    )


def call_fused(case, softmax_scale: float):
    q, k_cache, indices, topk_length, sink, _workspace = case
    return deep_gemm._C.sm120_sparse_mla_decode_fused(
        q,
        k_cache,
        indices,
        topk_length,
        sink,
        K_HEAD_DIM,
        softmax_scale,
        None,
    )


def call_workspace(case, block_size: int, softmax_scale: float):
    q, k_cache, indices, topk_length, sink, workspace = case
    deep_gemm._C.sm120_dequantize_and_gather_indexed_k_cache(
        workspace,
        k_cache,
        indices,
        topk_length,
        block_size,
        0,
    )
    return deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace(
        q,
        workspace,
        topk_length,
        None,
        sink,
        int(indices.shape[-1]),
        0,
        K_HEAD_DIM,
        softmax_scale,
        None,
    )


def call_workspace_split(case, block_size: int, softmax_scale: float):
    q, k_cache, indices, topk_length, sink, workspace = case
    deep_gemm._C.sm120_dequantize_and_gather_indexed_k_cache(
        workspace,
        k_cache,
        indices,
        topk_length,
        block_size,
        0,
    )
    return deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace_split(
        q,
        workspace,
        topk_length,
        None,
        sink,
        int(indices.shape[-1]),
        0,
        K_HEAD_DIM,
        softmax_scale,
        None,
    )


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
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    softmax_scale = K_HEAD_DIM**-0.5
    case = make_case(
        args.batch,
        args.heads,
        args.topk,
        args.num_blocks,
        args.block_size,
        args.device,
    )

    out_direct, lse_direct = call_direct(case, softmax_scale)
    out_fused, lse_fused = call_fused(case, softmax_scale)
    out_workspace, lse_workspace = call_workspace(case, args.block_size, softmax_scale)
    out_workspace_split, lse_workspace_split = call_workspace_split(
        case, args.block_size, softmax_scale
    )
    torch.cuda.synchronize()

    out_diff = (out_direct.float() - out_workspace.float()).abs()
    lse_diff = (lse_direct.float() - lse_workspace.float()).abs()
    fused_out_diff = (out_fused.float() - out_workspace.float()).abs()
    fused_lse_diff = (lse_fused.float() - lse_workspace.float()).abs()
    workspace_split_out_diff = (
        out_workspace_split.float() - out_workspace.float()
    ).abs()
    workspace_split_lse_diff = (
        lse_workspace_split.float() - lse_workspace.float()
    ).abs()
    print(
        f"shape: batch={args.batch} heads={args.heads} topk={args.topk} "
        f"blocks={args.num_blocks} block_size={args.block_size}"
    )
    print(f"out_max_abs: {out_diff.max().item():.6g}")
    print(f"out_mean_abs: {out_diff.mean().item():.6g}")
    print(f"lse_max_abs: {lse_diff.max().item():.6g}")
    print(f"fused_out_max_abs: {fused_out_diff.max().item():.6g}")
    print(f"fused_out_mean_abs: {fused_out_diff.mean().item():.6g}")
    print(f"fused_lse_max_abs: {fused_lse_diff.max().item():.6g}")
    print(
        "workspace_split_out_max_abs: "
        f"{workspace_split_out_diff.max().item():.6g}"
    )
    print(
        "workspace_split_out_mean_abs: "
        f"{workspace_split_out_diff.mean().item():.6g}"
    )
    print(
        "workspace_split_lse_max_abs: "
        f"{workspace_split_lse_diff.max().item():.6g}"
    )
    print(
        "direct_us: "
        f"{bench(lambda: call_direct(case, softmax_scale), args.warmup, args.iters):.3f}"
    )
    print(
        "fused_us: "
        f"{bench(lambda: call_fused(case, softmax_scale), args.warmup, args.iters):.3f}"
    )
    print(
        "workspace_us: "
        f"{bench(lambda: call_workspace(case, args.block_size, softmax_scale), args.warmup, args.iters):.3f}"
    )
    print(
        "workspace_split_us: "
        f"{bench(lambda: call_workspace_split(case, args.block_size, softmax_scale), args.warmup, args.iters):.3f}"
    )


if __name__ == "__main__":
    main()
