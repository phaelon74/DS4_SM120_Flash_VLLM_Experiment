# DeepGEMM SM120 / DeepSeek V4 Flash Handoff

This repository has been modified to make `deepseek-ai/DeepSeek-V4-Flash`
load and serve under vLLM on 2x NVIDIA RTX PRO 6000 Blackwell workstation
GPUs, which report compute capability SM120.

## Current Functional State

- Target runtime is `vllm/vllm-openai:deepseekv4-cu130`.
- Serving path is configured through `docker-compose.yml`.
- External API is exposed on host port `8080`.
- Model: `deepseek-ai/DeepSeek-V4-Flash`.
- vLLM serves the model with:
  - `--attention-backend FLASHMLA_SPARSE`
  - `--moe-backend deep_gemm`
  - `--kv-cache-dtype fp8_ds_mla`
  - tensor parallel size `2`
  - expert parallel enabled
  - max model length `131072`
- The model loads and returns valid chat completions.
- First-request OOM was fixed by capping the KV cache reservation.

## Current Default Runtime Profile

`docker-compose.yml` now defaults to a balance between 128k context and usable
throughput:

- `VLLM_MAX_MODEL_LEN=131072`
- `VLLM_KV_CACHE_MEMORY_BYTES=8589934592`
- `VLLM_MAX_NUM_BATCHED_TOKENS=4096`
- `VLLM_MAX_NUM_SEQS=4`
- `VLLM_ENFORCE_EAGER=0`
- `VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=4`
- `VLLM_PERFORMANCE_MODE=balanced`
- `VLLM_OPTIMIZATION_LEVEL=2`
- `DG_SM120_SEQUENTIAL_COMPRESSOR=1`
- `DG_SM120_FLASHMLA_PREFILL_WORKSPACE_FACTOR=1`
- `DG_SM120_ENABLE_B12X_MOE=0`

Observed after startup:

- CUDA graph capture completes.
- vLLM reports maximum concurrency for 131,072 token requests at about `4.07x`.
- Each RTX PRO 6000 sits around `88.9 GiB used / 8.4 GiB free`.

## Observed Benchmarks

Using `scripts/bench_deepseek_v4_flash_api.sh` against
`http://192.168.2.216:8080` with `MAX_TOKENS=512`, `MIN_TOKENS=512`,
`ignore_eos=true`, and unique prompts:

- `CONCURRENCY=1`: about `44.7 tok/s`
- `CONCURRENCY=2`: about `81.7 tok/s` aggregate, about `40.9 tok/s/request`
- `CONCURRENCY=4`: about `117.6-119.4 tok/s` aggregate, about `29.4-29.9 tok/s/request`

The earlier memory-safe eager profile produced only about `21-22 tok/s` and
queued all parallel requests because `max_num_seqs=1`.

## Major Code Changes

### Build and Container

- Added `Dockerfile.vllm-nightly-sm120`.
- Added `docker-compose.yml`.
- The Dockerfile is based on `vllm/vllm-openai:deepseekv4-cu130`.
- Installs CUDA 13 development libraries needed for extension builds.
- Installs `b12x==0.7.0` and `nvidia-cutlass-dsl-libs-cu13==4.4.2`.
- Runs `docker/patch_vllm_deepseekv4.py` to patch the installed vLLM package.

### vLLM Runtime Patcher

Added `docker/patch_vllm_deepseekv4.py`. It patches the installed vLLM package
inside the container at startup. Important patch categories:

- Treat SM120 as DeepGEMM-capable in vLLM CUDA platform checks.
- Allow DeepGEMM MXFP4/FP8 paths on SM120.
- Prepack MXFP4 scales for the SM120 DeepGEMM path.
- Patch DeepSeek V4 FP8 einsum paths to use DeepGEMM SM120 kernels where
  available.
- Add SM120 sparse MLA decode fallbacks and direct paths.
- Patch sparse-indexer behavior for SM120 and smaller/full-context shapes.
- Patch DeepSeek V4 compressor scheduling to avoid first-request peak-memory
  OOM.
- Reduce FlashMLA sparse prefill workspace reservation with
  `DG_SM120_FLASHMLA_PREFILL_WORKSPACE_FACTOR`.
- Add optional vLLM memory-breakdown logging.
- Add optional layer-level profiling hooks.
- Add optional b12x MXFP4 MoE path behind `DG_SM120_ENABLE_B12X_MOE=1`.

The b12x MoE path is currently disabled by default because it was not yet proven
stable end-to-end in this setup.

### DeepGEMM C++/CUDA Extension

Added SM120-specific extension sources:

- `csrc/sm120_fp8_fp8_cutlass.cu`
- `csrc/sm120_fp8_fp4_cutlass.cu`
- `csrc/sm120_fp8_gemm_fallback.cu`
- `csrc/sm120_mqa_logits_fallback.cu`
- `csrc/sm120_hc_prenorm_fallback.cu`
- `csrc/sm120_sparse_mla_decode.cu`
- `csrc/sm120_profile.hpp`

Added matching JIT header entry points:

- `csrc/jit_kernels/impls/sm120_fp8_fp8_cutlass.hpp`
- `csrc/jit_kernels/impls/sm120_fp8_fp4_cutlass.hpp`
- `csrc/jit_kernels/impls/sm120_fp8_gemm_fallback.hpp`
- `csrc/jit_kernels/impls/sm120_mqa_logits_fallback.hpp`
- `csrc/jit_kernels/impls/sm120_hc_prenorm_fallback.hpp`
- `csrc/jit_kernels/impls/sm120_sparse_mla_decode.hpp`

Modified API registration and dispatch files:

- `setup.py`
- `deep_gemm/__init__.py`
- `csrc/python_api.cpp`
- `csrc/apis/gemm.hpp`
- `csrc/apis/attention.hpp`
- `csrc/apis/hyperconnection.hpp`
- `csrc/apis/layout.hpp`
- `csrc/utils/layout.hpp`
- `csrc/jit/device_runtime.hpp`

### Scripts

Added operational and benchmark scripts:

- `scripts/test_deepseek_v4_flash_api.sh`
- `scripts/bench_deepseek_v4_flash_api.sh`
- `scripts/bench_deepseek_v4_flash_parallel_sweep.sh`
- `scripts/estimate_deepseek_v4_flash_memory.py`
- `scripts/bench_sm120_fp8_bhr_hdr_bhd.py`
- `scripts/bench_sm120_fp8_m1.py`
- `scripts/bench_sm120_fp8_projection_shapes.py`
- `scripts/bench_sm120_mhc.py`
- `scripts/bench_sm120_moe_small_m.py`
- `scripts/bench_sm120_workspace_attention.py`

## Key Findings

- The model requires both FP8 and FP4/MXFP4 paths to load on 2x RTX PRO 6000.
- The original failure mode was DeepGEMM/vLLM support checking for SM90/SM100
  but not SM120.
- A simple SM100-as-is path is not sufficient because DeepGEMM SM100 kernels use
  TMEM assumptions that do not map directly to SM120.
- The main functional remap needed was C128/DeepGEMM, not C4/Marlin.
- With vLLM auto KV reservation, the process left too little runtime scratch
  memory and failed at first request inside the DeepSeek compressor Triton
  kernel.
- Explicit `--kv-cache-memory-bytes 8589934592` provides enough scratch while
  retaining 128k context support.
- Enabling CUDA graphs and allowing `max_num_seqs=4` is required for usable
  aggregate throughput.
- Single-request throughput remains much lower than the target. The current
  system reaches 100+ tok/s aggregate only with multiple concurrent requests.

## Most Promising Research Directions

1. Implement a true production SM120 C128 FP8 decode kernel.
   The current path still appears bottlenecked by sparse MLA decode and/or MoE
   decode kernels. The goal is to remove fallback-style dequantization and
   workspace-heavy paths.

2. Replace any FP8 KV cache dequantize-to-BF16 workspace path with direct SM120
   tensor-core computation.
   The current workspace path is functional but likely too expensive for the
   desired single-request decode rate.

3. Revisit the MXFP4 MoE path using b12x as reference.
   There is optional b12x integration behind `DG_SM120_ENABLE_B12X_MOE=1`, but
   it needs correctness and layout validation. The likely useful pieces are the
   b12x SM120 FP8 activation x packed FP4 expert-weight flow and scale swizzles.

4. Profile per-layer and per-kernel time with the existing hooks.
   Set:
   - `DG_SM120_PROFILE_LAYER=1`
   - `DG_SM120_KERNEL_PROFILE=1`
   - `DG_SM120_KERNEL_PROFILE_EVERY=64`
   Kernel profile output defaults to `/tmp/dg_sm120_kernel_profile.tsv` inside
   the container.

5. Audit all remaining SM120 fallback paths.
   Files with `fallback` in the name should be treated as suspects for the
   single-request throughput ceiling.

6. Look for excessive synchronization in profiling or fallback paths.
   `csrc/sm120_profile.hpp` intentionally synchronizes only when profiling is
   enabled. Keep `DG_SM120_KERNEL_PROFILE=0` for real benchmarks.

## Useful Commands

Start the service:

```bash
docker compose up -d --force-recreate vllm
```

Health check:

```bash
curl -sS http://192.168.2.216:8080/health
```

Tiny correctness request:

```bash
curl -sS http://192.168.2.216:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"Reply exactly OK."}],"max_tokens":8,"temperature":0}'
```

512-token benchmark:

```bash
BASE_URL=http://192.168.2.216:8080 \
MAX_TOKENS=512 \
MIN_TOKENS=512 \
CONCURRENCY=4 \
scripts/bench_deepseek_v4_flash_api.sh
```

Parallel sweep:

```bash
BASE_URL=http://192.168.2.216:8080 \
MAX_TOKENS=512 \
MIN_TOKENS=512 \
CONCURRENCY_LIST="1 2 4" \
scripts/bench_deepseek_v4_flash_parallel_sweep.sh
```

Memory estimator:

```bash
python3 scripts/estimate_deepseek_v4_flash_memory.py
```

## Known Caveats

- The code is a working research prototype, not an upstream-quality patch.
- vLLM is patched in-place at container startup rather than rebuilt from source.
- `vllm_src/` is present for reference, but the running container patches the
  installed vLLM package in the image.
- `VLLM_*` local config variables produce "unknown vLLM environment variable"
  warnings. These are noisy but harmless because the variables are consumed by
  the compose startup shell before vLLM starts.
- The optional b12x MoE path is off by default.
- The single-request target of 100+ tok/s has not been achieved.
