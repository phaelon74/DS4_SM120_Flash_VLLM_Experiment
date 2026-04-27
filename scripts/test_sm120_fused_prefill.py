"""Synthetic correctness harness for SM120 fused sparse MLA prefill (v2).

Compares ``sm120_sparse_mla_prefill_v2`` (single-cache, FP8 cache direct,
online softmax, no BF16 workspace) against the existing
``sm120_sparse_mla_prefill_from_bf16_workspace_split`` reference using a
gathered BF16 workspace built with
``sm120_dequantize_and_gather_indexed_k_cache``.

Both should produce the same output within BF16 tolerance.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

# Allow running either as ``python3 scripts/test_sm120_fused_prefill.py`` from
# the project root or as ``python3 -m scripts.test_sm120_fused_prefill``.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from test_sm120_fused_decode import _make_fp8_ds_mla_cache  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import deep_gemm

    dg_c = deep_gemm._C

    head_dim = 512
    softmax_scale = 1.0 / math.sqrt(head_dim)

    cache = _make_fp8_ds_mla_cache(
        args.num_blocks, args.block_size, head_dim=head_dim, seed=args.seed
    )

    torch.manual_seed(args.seed + 7)
    q = torch.randn(
        (args.seq, args.heads, head_dim), device="cuda", dtype=torch.bfloat16
    )

    # Each sequence row picks ``topk`` workspace rows. Build a workspace_map
    # that just maps workspace_row==i to physical KV slot index ``slot[i]``.
    nslots = args.num_blocks * args.block_size
    n_workspace = max(args.seq * args.topk, 4096)
    slot = torch.randint(
        0, nslots, (n_workspace,), device="cuda", dtype=torch.int32
    )
    workspace_map = slot.clone()

    indices = torch.randint(
        0, n_workspace, (args.seq, 1, args.topk), device="cuda", dtype=torch.int32
    )
    topk_length = torch.full(
        (args.seq,), args.topk, device="cuda", dtype=torch.int32
    )

    print(
        f"running prefill v2 vs workspace-split reference: seq={args.seq} heads={args.heads}"
        f" topk={args.topk}"
    )

    # ---- v2 fused prefill --------------------------------------------------
    out_v2 = torch.empty(
        (args.seq, args.heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    out_v2_returned, _, lse_v2 = dg_c.sm120_sparse_mla_prefill_v2(
        q,
        cache,
        workspace_map,
        indices,
        topk_length,
        None,                       # attn_sink
        args.block_size,
        head_dim,
        softmax_scale,
        out_v2,
    )

    # ---- workspace-split reference -----------------------------------------
    # Build a physical-cache-indexed view of the same data: indices_phys[s, k]
    # = workspace_map[indices[s, 1, k]], then gather into BF16 and decode.
    physical_indices = workspace_map[indices.squeeze(1)].unsqueeze(1)
    workspace = torch.empty(
        (args.seq, args.topk, head_dim), device="cuda", dtype=torch.bfloat16
    )
    dg_c.sm120_dequantize_and_gather_indexed_k_cache(
        workspace, cache, physical_indices, topk_length, args.block_size, 0
    )
    out_ref = torch.empty_like(out_v2)
    out_ref_returned, _, lse_ref = (
        dg_c.sm120_sparse_mla_prefill_from_bf16_workspace_split(
            q,
            workspace.unsqueeze(1).reshape(args.seq, args.topk, head_dim),
            physical_indices,
            topk_length,
            None,
            head_dim,
            softmax_scale,
            out_ref,
        )
    )

    out_diff = (out_v2.float() - out_ref.float()).abs().max().item()
    lse_diff = (lse_v2 - lse_ref).abs().max().item()
    out_norm = out_ref.float().abs().max().item()
    print(f"  out max abs diff = {out_diff:.6f} (out max abs = {out_norm:.4f})")
    print(f"  lse max abs diff = {lse_diff:.6e}")

    if out_diff > max(0.05, 0.01 * max(out_norm, 1.0)):
        raise SystemExit(f"prefill v2 vs reference diff too large: {out_diff}")
    if lse_diff > 1e-2:
        raise SystemExit(f"prefill v2 vs reference LSE diff too large: {lse_diff}")
    print("OK")


if __name__ == "__main__":
    main()
