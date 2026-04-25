#!/usr/bin/env python3
import argparse
import time

import torch

import deep_gemm


K_HEAD_DIM = 512
K_FP8_DIM = 448
K_BF16_DIM = 64
K_SCALE_BYTES = 8


def make_packed_cache(num_blocks: int, block_size: int, device: str) -> torch.Tensor:
    torch.manual_seed(20260426)
    fp8_bytes = torch.randint(
        0,
        256,
        (num_blocks, block_size, K_FP8_DIM),
        device=device,
        dtype=torch.uint8,
    )
    rope = (
        torch.randn((num_blocks, block_size, K_BF16_DIM), device=device, dtype=torch.float32)
        * 0.05
    ).to(torch.bfloat16)
    rope_bytes = rope.view(torch.uint8).reshape(num_blocks, block_size, K_BF16_DIM * 2)
    scales = torch.full(
        (num_blocks, block_size, K_SCALE_BYTES),
        0x7F,
        device=device,
        dtype=torch.uint8,
    )
    return torch.cat((fp8_bytes, rope_bytes, scales), dim=-1).contiguous().unsqueeze(-2)


def make_case(batch: int, heads: int, seq_len: int, block_size: int, device: str):
    torch.manual_seed(20260426)
    q = (
        torch.randn((batch, 1, heads, K_HEAD_DIM), device=device, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)

    blocks_per_req = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_req + 8
    k_cache = make_packed_cache(num_blocks, block_size, device)

    block_table = torch.empty((batch, blocks_per_req), device=device, dtype=torch.int32)
    for req in range(batch):
        start = req * blocks_per_req
        block_table[req] = torch.arange(start, start + blocks_per_req, device=device, dtype=torch.int32)

    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    req_id_per_token = torch.arange(batch, device=device, dtype=torch.int32)
    sink = torch.zeros((heads,), device=device, dtype=torch.float32)
    workspace = torch.empty((batch, seq_len, K_HEAD_DIM), device=device, dtype=torch.bfloat16)
    return q.contiguous(), k_cache, block_table.contiguous(), seq_lens, req_id_per_token, sink, workspace


def call_direct(case, softmax_scale: float):
    q, k_cache, block_table, seq_lens, req_id_per_token, sink, _workspace = case
    return deep_gemm._C.sm120_sparse_mla_decode_full_context(
        q,
        k_cache,
        block_table,
        seq_lens,
        req_id_per_token,
        sink,
        K_HEAD_DIM,
        softmax_scale,
        None,
    )


def call_workspace(case, block_size: int, softmax_scale: float):
    q, k_cache, block_table, seq_lens, _req_id_per_token, sink, workspace = case
    deep_gemm._C.sm120_dequantize_and_gather_k_cache(
        workspace,
        k_cache,
        seq_lens,
        None,
        block_table,
        block_size,
        0,
    )
    return deep_gemm._C.sm120_sparse_mla_decode_from_bf16_workspace(
        q,
        workspace,
        seq_lens.to(torch.int64),
        None,
        sink,
        int(workspace.shape[1]),
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
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    softmax_scale = K_HEAD_DIM**-0.5
    case = make_case(args.batch, args.heads, args.seq_len, args.block_size, args.device)

    out_direct, lse_direct = call_direct(case, softmax_scale)
    out_workspace, lse_workspace = call_workspace(case, args.block_size, softmax_scale)
    torch.cuda.synchronize()

    out_diff = (out_direct.float() - out_workspace.float()).abs()
    lse_diff = (lse_direct.float() - lse_workspace.float()).abs()
    print(
        f"shape: batch={args.batch} heads={args.heads} seq_len={args.seq_len} "
        f"block_size={args.block_size}"
    )
    print(f"out_max_abs: {out_diff.max().item():.6g}")
    print(f"out_mean_abs: {out_diff.mean().item():.6g}")
    print(f"lse_max_abs: {lse_diff.max().item():.6g}")
    print(
        "direct_us: "
        f"{bench(lambda: call_direct(case, softmax_scale), args.warmup, args.iters):.3f}"
    )
    print(
        "workspace_us: "
        f"{bench(lambda: call_workspace(case, args.block_size, softmax_scale), args.warmup, args.iters):.3f}"
    )


if __name__ == "__main__":
    main()
