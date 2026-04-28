#!/usr/bin/env python3
"""dsl12x sparse MLA prefill correctness test.

Compares ``dsl12x.attention.prefill.run_sparse_mla_prefill`` output against
a PyTorch scalar reference (the same math the BMM bridge implements but
spelled out explicitly so the reference is independent of any kernel).

Usage:

    python3 scripts/test_dsl12x_sparse_prefill.py
    python3 scripts/test_dsl12x_sparse_prefill.py \
        --num-tokens 32 \
        --num-heads 32 \
        --qk-head-dim 576 \
        --v-head-dim 512 \
        --topk 128 \
        --kv-rows 4096 \
        --has-attn-sink \
        --tolerance-multiplier 1.5

Tolerance gate: scaled by ``sqrt((qk_head_dim + v_head_dim) * topk / 2048)``
so wider sums (more contributions) get more headroom. Lesson from the C4
mqa_logits MMA test pass.

Exit codes:
    0 = pass
    1 = correctness fail (max_diff exceeds tolerance)
    2 = NotImplementedError (kernel scaffold, MMA inner not yet written)
    3 = environment error (no CUDA, not SM120)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Bootstrap the workspace root (parent of scripts/) onto sys.path so the
# `dsl12x` package is importable regardless of where this script is run from.
_WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-tokens", type=int, default=32,
                        help="Q tokens (chunk size). Default 32.")
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Active Q heads (DG_SM120_ACTIVE_HEADS). Default 32.")
    parser.add_argument("--qk-head-dim", type=int, default=576,
                        help="qk_head_dim. Default 576 (DeepSeek V3-family).")
    parser.add_argument("--v-head-dim", type=int, default=512,
                        help="v_head_dim. Default 512.")
    parser.add_argument("--topk", type=int, default=128,
                        help="topk_max (DG_SM120_MAIN_TOPK_CAP). Default 128.")
    parser.add_argument("--kv-rows", type=int, default=4096,
                        help="N_kv (cache size). Default 4096.")
    parser.add_argument("--has-attn-sink", action="store_true",
                        help="Pass an attn_sink tensor (sigmoid-gate epilogue).")
    parser.add_argument("--has-topk-length", action="store_true", default=True,
                        help="Pass per-token topk_length (default: True).")
    parser.add_argument("--no-topk-length", dest="has_topk_length",
                        action="store_false")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility.")
    parser.add_argument("--abs-tol", type=float, default=5e-3,
                        help="Absolute tolerance base (scaled by reduction depth).")
    parser.add_argument("--rel-tol", type=float, default=5e-4,
                        help="Relative tolerance fallback.")
    parser.add_argument("--tolerance-multiplier", type=float, default=1.0,
                        help="Multiplier on the absolute tolerance gate.")
    parser.add_argument("--sm-scale", type=float, default=None,
                        help="Override sm_scale. Default 1/sqrt(qk_head_dim).")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Input dtype.")
    return parser.parse_args()


def reference_sparse_mla_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for sparse MLA prefill.

    Mirrors the math in docker/patch_vllm_deepseekv4.py:1685-1693 (the
    scalar fallback inside the BMM bridge), spelled out explicitly so this
    file does not depend on the patcher being importable.

    Returns:
        (out, lse) where out is BF16 and lse is FP32.
    """
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    qk_dim = q.shape[-1]
    out = torch.zeros((num_tokens, num_heads, d_v), dtype=q.dtype, device=q.device)
    lse = torch.full((num_tokens, num_heads), float("-inf"),
                     dtype=torch.float32, device=q.device)

    indices_2d = indices.squeeze(1)  # [num_tokens, topk]
    kv_2d = kv.squeeze(1)            # [N_kv, qk_dim]
    kv_f = kv_2d.to(torch.float32)
    q_f = q.to(torch.float32)

    for token_idx in range(num_tokens):
        # Build the per-token valid index list.
        if topk_length is not None:
            tk_len = int(topk_length[token_idx].item())
        else:
            tk_len = indices_2d.shape[1]
        token_indices = indices_2d[token_idx, :tk_len]
        valid_mask = (token_indices >= 0) & (token_indices < kv_2d.shape[0])
        token_indices = token_indices[valid_mask]
        if token_indices.numel() == 0:
            # No valid entries; output stays zero, LSE stays -inf.
            continue

        selected = kv_f.index_select(0, token_indices)
        # Scores: per-head [num_heads, num_valid].
        scores = (q_f[token_idx] @ selected[:, :qk_dim].T) * sm_scale
        per_head_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        token_out = (probs @ selected[:, :d_v].to(q.dtype))
        if attn_sink is not None:
            gate = torch.sigmoid(per_head_lse - attn_sink.to(per_head_lse.dtype))
            token_out = token_out * gate.to(token_out.dtype).unsqueeze(-1)
        out[token_idx] = token_out
        lse[token_idx] = per_head_lse

    return out, lse


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 3
    cap = torch.cuda.get_device_capability()
    if cap[0] != 12:
        print(f"ERROR: dsl12x requires SM120, got {cap[0]}.{cap[1]}", file=sys.stderr)
        return 3

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = "cuda"

    torch.manual_seed(args.seed)
    sm_scale = args.sm_scale if args.sm_scale is not None else 1.0 / math.sqrt(args.qk_head_dim)

    print(
        f"dsl12x sparse MLA prefill correctness test\n"
        f"  shape: tokens={args.num_tokens} heads={args.num_heads} "
        f"qk_dim={args.qk_head_dim} v_dim={args.v_head_dim} topk={args.topk} "
        f"kv_rows={args.kv_rows}\n"
        f"  dtype={args.dtype} sm_scale={sm_scale:.6f} "
        f"has_attn_sink={args.has_attn_sink} has_topk_length={args.has_topk_length}\n"
    )

    # Inputs.
    q = torch.randn(
        (args.num_tokens, args.num_heads, args.qk_head_dim),
        dtype=dtype, device=device,
    ) * 0.1
    kv = torch.randn(
        (args.kv_rows, 1, args.qk_head_dim),
        dtype=dtype, device=device,
    ) * 0.1
    indices = torch.randint(
        low=0, high=args.kv_rows,
        size=(args.num_tokens, 1, args.topk),
        device=device, dtype=torch.int32,
    )
    if args.has_topk_length:
        topk_length = torch.randint(
            low=max(1, args.topk // 4), high=args.topk + 1,
            size=(args.num_tokens,), device=device, dtype=torch.int32,
        )
    else:
        topk_length = None
    if args.has_attn_sink:
        attn_sink = torch.randn(
            (args.num_heads,), dtype=torch.float32, device=device,
        ) * 0.5
    else:
        attn_sink = None

    # Reference.
    print("computing PyTorch reference...")
    out_ref, lse_ref = reference_sparse_mla_prefill(
        q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=args.v_head_dim,
        attn_sink=attn_sink, topk_length=topk_length,
    )
    torch.cuda.synchronize()

    # dsl12x.
    print("running dsl12x kernel...")
    try:
        from dsl12x.attention.prefill import run_sparse_mla_prefill
        out_dsl, _max_logits, lse_dsl = run_sparse_mla_prefill(
            q=q, kv=kv, indices=indices, sm_scale=sm_scale,
            d_v=args.v_head_dim, attn_sink=attn_sink,
            topk_length=topk_length, chunk_size=args.num_tokens,
        )
        torch.cuda.synchronize()
    except NotImplementedError as e:
        print(f"\nKERNEL SCAFFOLD: {e}", file=sys.stderr)
        print(
            "The dsl12x prefill kernel is currently a scaffold with "
            "explicit XXX(verify) and XXX(MMA-INNER) markers. The MMA "
            "inner has not been filled in yet. This test will pass once "
            "the kernel is implemented (see dsl12x/README.md and the "
            "docstring of dsl12x/attention/prefill_kernel.py).",
            file=sys.stderr,
        )
        return 2
    except Exception as e:
        print(f"\nKERNEL ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # Tolerance gate: scaled by sqrt((qk_dim+v_dim)*topk/2048).
    scale = math.sqrt((args.qk_head_dim + args.v_head_dim) * args.topk / 2048.0)
    abs_gate = args.abs_tol * scale * args.tolerance_multiplier
    print(f"\ntolerance gate: abs={abs_gate:.4e} (base {args.abs_tol:.4e} x scale {scale:.2f})")

    # Output diff (BF16 -> FP32 for comparison).
    out_diff = (out_dsl.float() - out_ref.float()).abs()
    out_max = out_diff.max().item()
    out_mean = out_diff.mean().item()
    out_rel = (out_diff / (out_ref.float().abs() + 1e-6)).max().item()

    # LSE diff (already FP32).
    lse_diff = (lse_dsl - lse_ref).abs()
    # Mask -inf (no valid entries) before computing max.
    valid_lse = (lse_ref > -1e30) & (lse_dsl > -1e30)
    if valid_lse.any():
        lse_max = lse_diff[valid_lse].max().item()
        lse_mean = lse_diff[valid_lse].mean().item()
    else:
        lse_max = 0.0
        lse_mean = 0.0

    print(
        f"output diff: max={out_max:.4e} mean={out_mean:.4e} max_rel={out_rel:.4e}\n"
        f"LSE diff:    max={lse_max:.4e} mean={lse_mean:.4e}"
    )

    fail = False
    if out_max > abs_gate and out_rel > args.rel_tol:
        print(
            f"\nFAIL: output max_diff={out_max:.4e} exceeds gate "
            f"{abs_gate:.4e} AND max_rel={out_rel:.4e} exceeds "
            f"rel_tol={args.rel_tol:.4e}",
            file=sys.stderr,
        )
        fail = True
    if lse_max > abs_gate * 2:  # LSE has wider numerical tolerance
        print(
            f"\nFAIL: LSE max_diff={lse_max:.4e} exceeds gate "
            f"{abs_gate*2:.4e}",
            file=sys.stderr,
        )
        fail = True

    if fail:
        return 1
    print("\nPASS: dsl12x output matches reference within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
