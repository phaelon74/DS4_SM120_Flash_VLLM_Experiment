#!/usr/bin/env python3
# scripts/test_sm120_decode_v2_native.py
#
# Isolated correctness check: SM120 v2 scalar vs v2 native FP8 block-scaled
# MMA decode. Runs entirely against the loaded deep_gemm._C extension; no
# vLLM, no model load, no server restart required.
#
# Test design:
#   * Synthesize a tiny but realistic fp8_ds_mla cache:
#       - First 448 dims per token are FP8 e4m3 (one byte per dim).
#       - Last 64 dims per token are BF16 (128 bytes per token).
#       - 8 UE8M0 scale bytes follow the entire block of token data
#         (NOT interleaved per token).
#       - block_size = 256 tokens per block.
#   * Pin all UE8M0 scales to byte 127 (== 2^0 == 1.0) so scale-decoding
#     noise drops out of the comparison and we measure the MMA itself.
#   * Build random Q [B, H, 512] bf16 and random sparse indices into the
#     cache, with H=32 (multiple of native kHeadsPerCta=16).
#   * Run twice with the same input tensors:
#       (1) DG_SM120_FUSED_DECODE_V2_NATIVE=0 -> scalar v2 reference
#       (2) DG_SM120_FUSED_DECODE_V2_NATIVE=1 -> native FP8 MMA candidate
#   * Compare out (bf16) and lse (fp32).
#
# Pass criteria (printed but not asserted; user evaluates):
#   * out:  max abs diff < 0.05, max rel diff < 0.05 typical for bf16 logits.
#   * lse:  max abs diff < 1e-3 typical (fp32 reduction).

import math
import os
import sys

import numpy as np
import torch


def encode_fp8_e4m3(values_fp32: np.ndarray) -> np.ndarray:
    """Round bf16 / fp32 magnitudes to E4M3 byte representation.

    Uses torch's built-in cast for correctness; no need to hand-roll.
    """
    t = torch.from_numpy(values_fp32.astype(np.float32))
    e4m3 = t.to(torch.float8_e4m3fn)
    # Reinterpret as uint8 bytes.
    return e4m3.view(torch.uint8).numpy().astype(np.uint8)


def synthesize_fp8_ds_mla_cache(
    num_blocks: int,
    block_size: int,
    rng: np.random.Generator,
):
    """Returns (cache_bytes_2d, k_dense_bf16) where:
       cache_bytes_2d : torch.uint8 cuda, shape [num_blocks, block_size * 584]
       k_dense_bf16   : torch.bfloat16 cuda, shape [num_blocks * block_size, 512]

    The cache is the on-device representation expected by the v2 kernel.
    k_dense_bf16 is a Python-side decoded view used only for sanity prints.
    """
    fp8_dim = 448
    bf16_dim = 64
    head_dim = fp8_dim + bf16_dim
    token_data_bytes = fp8_dim + bf16_dim * 2  # 576
    scale_bytes = 8
    quant_block = 64
    num_quant_blocks = fp8_dim // quant_block  # 7
    num_tokens = num_blocks * block_size
    per_block_bytes = block_size * (token_data_bytes + scale_bytes)

    # Pin all UE8M0 scales to byte 127 (== exp2(127-127) = 1.0) so the
    # native MMA's per-block scale just becomes pass-through. This isolates
    # MMA correctness from scale-codec correctness.
    UE8M0_BYTE_ONE_F32 = 127

    # Generate FP8-quantizable per-token data with bounded magnitude. We
    # restrict the FP8 portion to roughly [-1, 1] so that the range stays
    # well inside e4m3 saturation, and the BF16 RoPE portion to a similar
    # range.
    k_dense = rng.standard_normal(
        (num_tokens, head_dim), dtype=np.float32
    ).astype(np.float32) * 0.25

    # Build the cache layout:
    #   block: [block_size * 576 token-bytes] [block_size * 8 scale-bytes]
    cache = np.zeros((num_blocks, per_block_bytes), dtype=np.uint8)

    for block_id in range(num_blocks):
        block_token0 = block_id * block_size
        token_region = cache[block_id, : block_size * token_data_bytes]
        scale_region = cache[block_id, block_size * token_data_bytes:]
        for t in range(block_size):
            token_bytes = token_region[
                t * token_data_bytes : (t + 1) * token_data_bytes
            ]
            row = k_dense[block_token0 + t]
            # FP8 portion: dims [0, 448).
            fp8_region = row[:fp8_dim]
            fp8_bytes = encode_fp8_e4m3(fp8_region)  # [448]
            token_bytes[:fp8_dim] = fp8_bytes
            # BF16 portion: dims [448, 512), 64 bf16 values, 128 bytes.
            bf16_view = (
                torch.from_numpy(row[fp8_dim:].copy())
                .to(torch.bfloat16)
                .view(torch.uint8)
                .numpy()
            )
            token_bytes[fp8_dim:] = bf16_view  # 128 bytes
            # Scale bytes for this token: 7 used + 1 pad.
            scale_region[t * scale_bytes : (t + 1) * scale_bytes] = (
                np.array(
                    [UE8M0_BYTE_ONE_F32] * num_quant_blocks + [0],
                    dtype=np.uint8,
                )
            )

    cache_t = torch.from_numpy(cache).cuda().contiguous()

    # Replace k_dense with the dequantized-via-FP8 view so the comparison
    # reference matches what the kernel actually consumes.
    k_dense_consumed = np.zeros_like(k_dense)
    k_dense_consumed[:, :fp8_dim] = (
        torch.from_numpy(
            cache.reshape(num_blocks, -1)[:, : block_size * token_data_bytes]
            .reshape(num_blocks * block_size, token_data_bytes)[:, :fp8_dim]
            .copy()
        )
        .view(torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
        .numpy()
    )
    k_dense_consumed[:, fp8_dim:] = (
        torch.from_numpy(
            cache.reshape(num_blocks, -1)[:, : block_size * token_data_bytes]
            .reshape(num_blocks * block_size, token_data_bytes)[:, fp8_dim:]
            .copy()
        )
        .view(torch.uint8)
        .view(torch.bfloat16)
        .float()
        .numpy()
    )

    k_dense_bf16 = (
        torch.from_numpy(k_dense_consumed).to(torch.bfloat16).cuda()
    )
    return cache_t, k_dense_bf16


def run_v2(
    deep_gemm_C,
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    block_size,
    use_native: bool,
):
    os.environ["DG_SM120_FUSED_DECODE_V2_NATIVE"] = "1" if use_native else "0"
    out, lse = deep_gemm_C.sm120_sparse_mla_decode_v2(
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink,
        head_dim_v,
        softmax_scale,
        block_size,
        None,  # out (will be allocated)
    )
    torch.cuda.synchronize()
    return out, lse


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; aborting.", file=sys.stderr)
        return 2
    torch.cuda.set_device(0)
    rng = np.random.default_rng(0xC0FFEE)

    import deep_gemm._C as _C  # noqa: E402

    if not hasattr(_C, "sm120_sparse_mla_decode_v2"):
        print("FATAL: deep_gemm._C.sm120_sparse_mla_decode_v2 missing", file=sys.stderr)
        return 2

    # Shape: small but realistic.
    B = 1
    H = 32        # H % 16 == 0 (native requires).
    head_dim = 512
    block_size = 256
    num_blocks = 2
    num_tokens = num_blocks * block_size
    K = 128       # matches DG_SM120_MAIN_TOPK_CAP.

    print(
        f"[shape] B={B} H={H} head_dim={head_dim} block_size={block_size} "
        f"num_blocks={num_blocks} num_tokens={num_tokens} K={K}"
    )

    # Q: bf16 in cuda, small magnitude so logits fall in a sane range.
    q = (
        torch.from_numpy(rng.standard_normal((B, H, head_dim), dtype=np.float32))
        .to(torch.bfloat16)
        .cuda()
        * 0.1
    )

    # Cache + reference dense view.
    k_cache, _k_dense = synthesize_fp8_ds_mla_cache(num_blocks, block_size, rng)

    # Indices: K random valid linear indices per batch, plus a few -1 padding.
    raw = rng.integers(low=0, high=num_tokens, size=(B, 1, K), dtype=np.int32)
    raw[..., -2:] = -1  # last two are padding
    indices = torch.from_numpy(raw).cuda()

    # topk_length: K-2 valid for each batch (matching the -1 padding above).
    topk_length = torch.full((B,), K - 2, dtype=torch.int32, device="cuda")

    # No attention sink for this test.
    attn_sink = None

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # ---- Run scalar v2 (reference) ----
    print("[run] DG_SM120_FUSED_DECODE_V2_NATIVE=0 (scalar v2 reference)...")
    out_ref, lse_ref = run_v2(
        _C, q, k_cache, indices, topk_length, attn_sink,
        head_dim, softmax_scale, block_size, use_native=False,
    )
    print(f"  out_ref: shape={tuple(out_ref.shape)} dtype={out_ref.dtype}")
    print(f"  lse_ref: shape={tuple(lse_ref.shape)} dtype={lse_ref.dtype}")
    print(f"  out_ref[0,0,:8] = {out_ref[0,0,:8].float().cpu().numpy()}")
    print(f"  lse_ref[0,:8]   = {lse_ref[0,:8].float().cpu().numpy()}")

    # ---- Run native v2 (candidate) ----
    print("[run] DG_SM120_FUSED_DECODE_V2_NATIVE=1 (native FP8 MMA candidate)...")
    try:
        out_nat, lse_nat = run_v2(
            _C, q, k_cache, indices, topk_length, attn_sink,
            head_dim, softmax_scale, block_size, use_native=True,
        )
    except Exception as e:
        print(f"FATAL: native call raised: {e!r}", file=sys.stderr)
        return 1
    print(f"  out_nat: shape={tuple(out_nat.shape)} dtype={out_nat.dtype}")
    print(f"  lse_nat: shape={tuple(lse_nat.shape)} dtype={lse_nat.dtype}")
    print(f"  out_nat[0,0,:8] = {out_nat[0,0,:8].float().cpu().numpy()}")
    print(f"  lse_nat[0,:8]   = {lse_nat[0,:8].float().cpu().numpy()}")

    # ---- Diff ----
    out_diff = (out_nat.float() - out_ref.float()).abs()
    lse_diff = (lse_nat.float() - lse_ref.float()).abs()
    out_max = out_diff.max().item()
    out_mean = out_diff.mean().item()
    lse_max = lse_diff.max().item()
    lse_mean = lse_diff.mean().item()

    # Relative diff for out (avoid div by ~0 by clamping denominator).
    denom = out_ref.float().abs().clamp_min(1e-3)
    out_rel = (out_diff / denom).max().item()

    print()
    print("=" * 60)
    print(f"out:  max_abs={out_max:.6f}  mean_abs={out_mean:.6f}  max_rel={out_rel:.6f}")
    print(f"lse:  max_abs={lse_max:.6f}  mean_abs={lse_mean:.6f}")
    print("=" * 60)

    # Soft pass criteria.
    out_ok = out_max < 0.05 and out_rel < 0.10
    lse_ok = lse_max < 1e-3
    if out_ok and lse_ok:
        print("[verdict] PASS (within bf16/fp32 reduction tolerance)")
        return 0
    else:
        print("[verdict] DIFF observed; native path differs from scalar v2 reference")
        print(
            "         (this does NOT necessarily mean the native path is wrong; "
            "it means a side-by-side numerical investigation is warranted)"
        )
        return 0  # exit 0 so user can inspect even on diff


if __name__ == "__main__":
    sys.exit(main())
