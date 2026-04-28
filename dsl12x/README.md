# dsl12x

Standalone CuTe DSL kernel library for DeepSeek V4 Flash on NVIDIA SM120
(RTX PRO 6000 Blackwell workstation).

dsl12x is **independent of b12x**. b12x is the architectural reference (it
shipped first and proved that CuTe DSL kernels run well on SM120/SM121),
but no code in dsl12x imports from b12x. Where dsl12x reuses architectural
patterns (single-warp per-CTA, per-group Q+KV streaming, double-buffered
`cp.async`, online softmax, JIT-compiled `@cute.kernel` with LRU host
launcher cache), the patterns are re-implemented from first principles
using `cutlass.cute` primitives.

## Why standalone

1. **Different model contract.** b12x targets the GLM-5.1 NSA contract
   (page table, packed cache row layout `nope|scales|rope`, uniform
   per-page step). DeepSeek V4 Flash has a different contract: top-k
   indices + per-token `topk_length`, dual cache (C128 compressed + SWA),
   sigmoid-gate `attn_sink` epilogue. Building on top of b12x would mean
   carrying GLM-5.1-isms forever.
2. **Direct FP8 cache reads (eventual goal).** Removing the BMM bridge
   requires kernel ownership of the FP8 dequant + workspace map encoding
   scheme. Hard to do as a thin adapter on top of b12x.
3. **Long-term clarity.** dsl12x can grow to own the full SM120
   attention surface for DeepSeek V4 (prefill + decode + indexer + future
   kernels) without any external dependency on b12x's evolution.

## What's NOT in dsl12x (intentional separation)

- MoE kernels stay raw CUDA in `csrc/`. b12x provides MoE via
  `b12x.integration.tp_moe`; we keep the existing `csrc` MoE path
  because it works and dsl12x is attention-only.
- Dense GEMM (FP8xFP8, FP8xFP4) stays raw CUDA in `csrc/`.
- mHC HyperConnection prenorm stays raw CUDA in `csrc/sm120_*hc_prenorm*.cu`
  (the math is simple enough that raw CUDA + `mma.sync.aligned` PTX is
  the right tool).

## What is implemented vs scaffold (this initial release)

| Component                              | Status     | File                                           |
| -------------------------------------- | ---------- | ---------------------------------------------- |
| Package + runtime helpers              | Complete   | `dsl12x/__init__.py`, `runtime.py`, `ptx.py`, `smem.py`, `jit_cache.py` |
| Hello-MMA toolchain smoke test         | Complete   | `dsl12x/hello_mma/kernel.py` + `scripts/test_dsl12x_smoke.py` |
| Sparse MLA traits (shape-keyed)        | Complete   | `dsl12x/attention/traits.py`                   |
| Sparse MLA prefill kernel              | **Scaffold** | `dsl12x/attention/prefill_kernel.py` (XXX(verify) and XXX(MMA-INNER) markers in body) |
| Sparse MLA prefill host wrapper        | Complete   | `dsl12x/attention/prefill.py`                  |
| Sparse MLA decode kernel               | **Scaffold** | `dsl12x/attention/decode_kernel.py`            |
| Sparse MLA decode host wrapper         | Complete (NotImplementedError) | `dsl12x/attention/decode.py` |
| mqa_logits indexer kernel              | **Scaffold** | `dsl12x/attention/indexer_kernel.py`           |
| mqa_logits indexer host wrapper        | Complete (NotImplementedError) | `dsl12x/attention/indexer.py` |
| Patcher integration (try/except + warmup) | Complete | `docker/patch_vllm_deepseekv4.py` `_sm120_flash_mla_sparse_prefill_fwd` body |
| Sibling Docker container               | Complete   | `Dockerfile.vllm-nightly-sm120-dsl12x` + `docker-compose.dsl12x.yml` |
| Tests + bench                          | Complete   | `scripts/test_dsl12x_*.py` + `scripts/bench_dsl12x_*.py` |

## Scaffold semantics

A "scaffold" file contains:
- Complete module structure (imports, decorators, function signatures).
- Complete docstrings explaining the kernel architecture.
- Complete SMEM layout calculation, CTA grid sizing, fragment register
  layout, online-softmax math, epilogue gating semantics.
- An explicit `XXX(verify)` or `XXX(MMA-INNER)` marker at the line where
  the cutlass.cute API surface or PTX-level work needs to be filled in
  by a follow-up session that has access to the test system.

The scaffolds compile (assuming `cutlass.cute` is available) and the host
wrappers route to them when the corresponding env var is set
(`DG_SM120_DSL12X_PREFILL=1`, `DG_SM120_DSL12X_DECODE=1`, etc.). On
runtime failure the patcher's `try/except` falls back to the existing
production path (BMM bridge for prefill; existing `csrc/` paths for
decode and mqa_logits).

This means **the scaffolds are safe to merge to main**. Production
behavior is unchanged because the env vars default OFF in
`docker-compose.yml`. They flip ON only in the sibling
`docker-compose.dsl12x.yml` service on host port 8081.

## Architecture

```
+-------------------------------------------------------+
|              vLLM serve (DeepSeek V4 Flash)           |
+-------------------------------------------------------+
                          |
                          v
        flash_mla_sparse_fwd (patched)
                          |
              +-----------+-----------+
              v                       v
   DG_SM120_DSL12X_PREFILL=1    (default path)
              |                       |
              v                       v
   dsl12x.attention.prefill    BMM bridge (existing)
              |                       |
              v                       v
   prefill_kernel.py           torch.bmm + softmax + ...
   @cute.kernel
   (online softmax + sigmoid-gate)
              |
              v
   BF16 output + FP32 LSE
```

## Per-CTA design (sparse MLA prefill)

```
Threads:    32 (one warp).
SMEM:       ~14-22 KB depending on traits.
Tile:       16 Q heads x 32 KV tokens x qk_head_dim
            (swept in groups of nope_group_elems=64).
MMA:        BF16 m16n8k16 for QK and PV (4th-gen tensor cores).
Output:     BF16 [num_tokens, num_heads, v_head_dim].
LSE:        FP32 [num_tokens, num_heads]
            (for chunked-prefill cross-chunk reduction by the patcher).
```

Hot loop:

```
for each token tile (chunk_size / 32 tiles):
  reset online-softmax frags: m=-inf, d=0, o=0
  for each topk slot k = 0..topk_max-1:
    decode workspace_map[token, k] -> (cache_id, row, valid)
    cp.async stage K row from cache (FP8) into SMEM
    cp.async stage V row from cache (FP8) into SMEM
    sync + cp.async.commit
    dequant FP8 -> BF16 in registers
    QK MMA (BF16 m16n8k16); scores += sm_scale * (Q @ K_row^T)
    mask invalid scores to -inf
    online softmax update: m_new, rescale, d_new, o *= rescale
    PV MMA (BF16 m16n16k16): o += probs @ V_row
  epilogue:
    out = o / d
    lse = m * ln(2) + log(d * ln(2))
    if has_attn_sink:
      out *= sigmoid(lse - attn_sink[head])
    store BF16 out, FP32 LSE
```

## Dual-cache workspace map encoding

The patcher provides `workspace_map: int32 [num_tokens, topk_max]`:

| `workspace_map[m, k]` | Meaning                                                       |
| --------------------- | ------------------------------------------------------------- |
| `>= 0`                | C128 cache row index `workspace_map[m, k]`.                   |
| `< 0` (not sentinel)  | SWA cache row index `~workspace_map[m, k]` (bitwise-NOT).     |
| `INT32_MIN` (sentinel)| Invalid: out-of-`topk_length` or original cache row was -1.   |

For the **first ship** (this initial release), the workspace map is
single-cache only (all entries are non-negative C128 row indices, or
the invalid sentinel). The dual-cache encoding is documented and the
kernel scaffold supports it via `has_swa=True`, but the host wrapper
currently builds single-cache maps (the patcher contract gives a single
pre-merged kv tensor, and decoding the C128/SWA split requires deeper
patcher integration). Direct FP8 cache reads + dual cache will land
in a follow-up patch.

## attn_sink semantic

DeepSeek V4 Flash's `attn_sink` is a **post-attention sigmoid gate**, not
a softmax-denominator sink token. Confirmed by reading the BMM bridge at
`docker/patch_vllm_deepseekv4.py:1613-1622` and the scalar reference at
`:1692-1693`:

```python
out = softmax(QK^T * sm_scale) @ V          # standard attention
lse = logsumexp(QK^T * sm_scale, dim=-1)    # natural log

if attn_sink is not None:
    gate = sigmoid(lse - attn_sink[head])   # per-head, scalar per token
    out *= gate.unsqueeze(-1)               # broadcast over v_head_dim
```

NOT a softmax-denominator sink (which would be
`softmax_denom += exp(sink); out = exp(scores) / softmax_denom @ V`).

The kernel's epilogue computes lse from the online-softmax `m` and `d`
frags then applies the sigmoid gate before storing.

## SMEM budget (default DeepSeek V4 Flash shape)

Default traits at `qk_head_dim=576, v_head_dim=512, topk_max=128, num_heads=32, chunk_size=256`:

| Component       | Size              |
| --------------- | ----------------- |
| Q stage         | 16 heads * 64 group * 2 B = 2 KB  |
| KV stages (x2)  | 32 tokens * 64 * 2 B * 2 = 8 KB   |
| Workspace map   | 32 tokens * 128 topk * 4 B = 16 KB |
| Scale buffer    | ~4 KB             |
| Idx buffer      | ~256 B            |
| **Total**       | **~30 KB / CTA**  |

Note: this exceeds the 25 KB target for >=4 CTAs/SM at 100 KB SMEM/SM.
Two mitigations:

- Reduce `nope_group_elems` from 64 to 32 (kernel sweeps more groups
  per tile but each KV stage halves to 4 KB; total drops to ~22 KB).
- Reduce `topk_max` to live distribution (Phase 1 trace will show what
  the actual distribution is; if median is 64 then the workspace map
  drops to 8 KB; total ~22 KB).

The traits constructor validates SMEM/CTA <= device limit (99 KB on SM120
opt-in) and the host wrapper raises `RuntimeError` if the budget is
exceeded.

## Deployment workflow

```bash
# 1. Capture live shape from trace (run during normal serving):
docker compose down vllm
DG_SM120_PREFILL_V2_TRACE=1 docker compose up -d vllm
# (run any benchmark; trace prints to stderr; reach into logs:)
docker compose logs vllm 2>&1 | grep sm120_prefill_v2_trace

# 2. Stop production and bring up dsl12x sibling:
docker compose down vllm
docker compose -f docker-compose.dsl12x.yml up -d vllm-dsl12x

# 3. Hit the sibling on port 8081 to exercise dsl12x prefill:
curl -sS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"Reply exactly OK."}],"max_tokens":8,"temperature":0}'

# 4. (Once dsl12x prefill is proven, edit docker-compose.yml to flip
#     DG_SM120_DSL12X_PREFILL=1 production-default; retire the sibling.)
```

## Toolchain dependency

dsl12x requires:
- `cutlass.cute` (provided by `nvidia-cutlass-dsl-libs-cu13==4.4.2`).
- SM120 / sm_120a target (`CUTE_DSL_ARCH=sm_120a`).

Both are already installed in the production base container
(`vllm/vllm-openai:deepseekv4-cu130`); no new pip dependencies needed.

## Roadmap (follow-up sessions)

| # | Work item                                                          | Estimated session length |
| - | ------------------------------------------------------------------ | ------------------------ |
| 1 | Fill in prefill_kernel.py MMA inner (BF16 m16n8k16 QK + PV)        | 1-2 sessions, requires test feedback |
| 2 | Validate prefill correctness vs scalar; tune chunk size            | 1 session                |
| 3 | Patcher work: expose vLLM FP8 cache ptrs + dequant scales          | 1 session                |
| 4 | Update kernel + host wrapper to use direct FP8 cache reads         | 1-2 sessions             |
| 5 | Implement decode_kernel.py MMA inner                               | 1-2 sessions             |
| 6 | Implement indexer_kernel.py MMA inner (port C4 raw-CUDA to dsl12x) | 1 session                |
| 7 | mHC MMA kernel inner (csrc/sm120_tf32_hc_prenorm_gemm.cu)          | 1 session                |

## File map

```
dsl12x/
  __init__.py                  Package surface.
  runtime.py                   Stream + hardware-info helpers.
  ptx.py                       Inline-PTX wrappers (cp.async, ex2, lg2).
  smem.py                      Permuted-offset SMEM helpers.
  jit_cache.py                 LRU host launcher cache.
  hello_mma/
    __init__.py
    kernel.py                  Toolchain smoke test kernel.
  attention/
    __init__.py
    traits.py                  SparseMLATraits + workspace map encoding.
    prefill_kernel.py          [SCAFFOLD] sparse MLA prefill kernel.
    prefill.py                 Host wrapper + warmup.
    decode_kernel.py           [SCAFFOLD] sparse MLA decode kernel.
    decode.py                  Host wrapper.
    indexer_kernel.py          [SCAFFOLD] mqa_logits indexer kernel.
    indexer.py                 Host wrapper.
  README.md                    This file.
```

Tests + bench live outside `dsl12x/`:

```
scripts/test_dsl12x_smoke.py            Toolchain smoke test runner.
scripts/test_dsl12x_sparse_prefill.py   Correctness vs scalar reference.
scripts/bench_dsl12x_sparse_prefill.py  dsl12x vs BMM vs scalar.
```

Sibling Docker:

```
Dockerfile.vllm-nightly-sm120-dsl12x     Sibling vLLM image (port 8081).
docker-compose.dsl12x.yml                Sibling vllm-dsl12x service.
```
