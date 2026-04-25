#!/usr/bin/env python3
"""Estimate DeepSeek V4 Flash KV/workspace memory for vLLM FP8 MLA serving."""

from __future__ import annotations

import argparse
import math


def gib(num_bytes: int | float) -> float:
    return num_bytes / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=576)
    parser.add_argument("--fp8-mla-token-bytes", type=int, default=656)
    parser.add_argument("--prefill-workspace-factor", type=int, default=1)
    args = parser.parse_args()

    blocks_per_seq = math.ceil(args.max_model_len / args.block_size)
    allocated_tokens_per_seq = blocks_per_seq * args.block_size
    kv_bytes_per_gpu = (
        args.max_num_seqs
        * allocated_tokens_per_seq
        * args.layers
        * args.fp8_mla_token_bytes
    )

    # WorkspaceManager stores one reusable buffer per worker. For FP8 sparse MLA
    # it must hold q_concat plus the BF16 upconverted prefill workspace.
    q_concat_bytes = (
        args.max_num_batched_tokens * args.local_heads * args.head_dim * 2
    )
    prefill_workspace_bytes = (
        args.max_model_len * args.prefill_workspace_factor * args.head_dim * 2
    )

    total_per_gpu = kv_bytes_per_gpu + q_concat_bytes + prefill_workspace_bytes
    print(f"allocated_tokens_per_seq: {allocated_tokens_per_seq}")
    print(f"fp8_mla_kv_per_gpu_gib: {gib(kv_bytes_per_gpu):.3f}")
    print(f"q_concat_workspace_per_gpu_gib: {gib(q_concat_bytes):.3f}")
    print(f"prefill_workspace_per_gpu_gib: {gib(prefill_workspace_bytes):.3f}")
    print(f"kv_plus_flashmla_workspace_per_gpu_gib: {gib(total_per_gpu):.3f}")
    print(f"cluster_kv_plus_workspace_gib_tp{args.tp}: {gib(total_per_gpu * args.tp):.3f}")


if __name__ == "__main__":
    main()
