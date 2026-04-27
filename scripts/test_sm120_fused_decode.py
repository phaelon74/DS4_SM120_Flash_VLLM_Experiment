"""Synthetic correctness harness for SM120 fused sparse MLA decode (v2).

Compares the v2 fused decode (FP8 cache direct, online softmax, no BF16
workspace) against the existing
``sm120_sparse_mla_decode_from_bf16_workspace_split`` pipeline driven by
``sm120_dequantize_and_gather_indexed_k_cache``. Both should produce the same
output within BF16 tolerance.

Run inside the vLLM container after ``setup.py build_ext --inplace``::

    docker compose exec -T vllm bash -lc \
      'cd /workspace/DeepGEMM && python3 scripts/test_sm120_fused_decode.py'
"""

from __future__ import annotations

import argparse
import math

import torch


def _make_fp8_ds_mla_cache(
    num_blocks: int,
    block_size: int,
    head_dim: int = 512,
    fp8_dim: int = 448,
    *,
    seed: int = 0,
    device: str = "cuda",
):
    bf16_dim = head_dim - fp8_dim
    token_data_bytes = fp8_dim + bf16_dim * 2
    scale_bytes = 8
    bytes_per_block = block_size * (token_data_bytes + scale_bytes)

    torch.manual_seed(seed)
    cache = torch.empty(
        (num_blocks, block_size, 1, token_data_bytes + scale_bytes),
        dtype=torch.uint8,
        device=device,
    )

    fp8_part = torch.randn(
        (num_blocks, block_size, fp8_dim), device=device, dtype=torch.float32
    )
    fp8_e4m3 = fp8_part.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    fp8_view = fp8_e4m3.view(torch.uint8)

    bf16_part = torch.randn(
        (num_blocks, block_size, bf16_dim), device=device, dtype=torch.bfloat16
    )

    scales = torch.full(
        (num_blocks, block_size, scale_bytes),
        fill_value=127,
        dtype=torch.uint8,
        device=device,
    )
    cache_view = cache.view(num_blocks, block_size, token_data_bytes + scale_bytes)
    cache_view[..., :fp8_dim] = fp8_view
    cache_view[..., fp8_dim : fp8_dim + bf16_dim * 2] = bf16_part.view(
        torch.uint8
    ).reshape(num_blocks, block_size, bf16_dim * 2)
    cache_view[..., token_data_bytes:] = scales
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-sink", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    import deep_gemm

    dg_c = deep_gemm._C

    head_dim = 512
    softmax_scale = 1.0 / math.sqrt(head_dim)

    cache = _make_fp8_ds_mla_cache(
        args.num_blocks, args.block_size, head_dim=head_dim, seed=args.seed
    )

    torch.manual_seed(args.seed + 1)
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
    sink = (
        torch.randn((args.heads,), device="cuda", dtype=torch.float32)
        if args.with_sink
        else None
    )

    print(
        f"running v2 vs workspace-split reference: batch={args.batch} heads={args.heads}"
        f" topk={args.topk} block_size={args.block_size}"
    )

    # ---- v2 fused decode ---------------------------------------------------
    out_v2 = torch.empty(
        (args.batch, args.heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    out_v2_returned, lse_v2 = dg_c.sm120_sparse_mla_decode_v2(
        q,
        cache,
        indices,
        topk_length,
        sink,
        head_dim,
        softmax_scale,
        args.block_size,
        out_v2,
    )

    # ---- workspace-split reference -----------------------------------------
    workspace = torch.empty(
        (args.batch, args.topk, head_dim), device="cuda", dtype=torch.bfloat16
    )
    dg_c.sm120_dequantize_and_gather_indexed_k_cache(
        workspace,
        cache,
        indices,
        topk_length,
        args.block_size,
        0,
    )
    out_ref = torch.empty(
        (args.batch, args.heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    out_ref_returned, lse_ref = dg_c.sm120_sparse_mla_decode_from_bf16_workspace_split(
        q,
        workspace,
        topk_length,
        None,
        sink,
        args.topk,
        0,
        head_dim,
        softmax_scale,
        out_ref,
    )

    out_diff = (out_v2.float() - out_ref.float()).abs().max().item()
    lse_diff = (lse_v2 - lse_ref).abs().max().item()

    out_norm = out_ref.float().abs().max().item()
    print(f"  out max abs diff = {out_diff:.6f} (out max abs = {out_norm:.4f})")
    print(f"  lse max abs diff = {lse_diff:.6e}")

    # The v2 inner loop is scalar fp32 just like the reference; we expect
    # bit-exact bf16 output up to addition order. Use a generous tolerance for
    # safety since reduction order may differ.
    if out_diff > max(0.02, 0.005 * max(out_norm, 1.0)):
        raise SystemExit(f"v2 vs reference output diff too large: {out_diff}")
    if lse_diff > 1e-3:
        raise SystemExit(f"v2 vs reference LSE diff too large: {lse_diff}")
    print("OK")


if __name__ == "__main__":
    main()
