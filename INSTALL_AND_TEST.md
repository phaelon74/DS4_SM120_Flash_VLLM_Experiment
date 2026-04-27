# Install + Test on a 4x RTX PRO 6000 Blackwell host

This guide walks through installing this fork, building the SM120 DeepGEMM
extension (including the new fused v2 kernels), starting vLLM with DeepSeek
V4 Flash, and validating the new `*_V2` paths against the regression gate.

It assumes a Linux host with **4x NVIDIA RTX PRO 6000 Blackwell** (compute
capability SM120), ~768 GB system RAM, NVMe scratch, and a working Docker
+ NVIDIA Container Toolkit setup.

> **Important**: this fork is a research prototype, not an upstream-quality
> patch. The new v2 fused kernels were authored without iterative testing on
> the dev box and need on-host validation before being relied on. Follow the
> recommended order below and only flip the v2 flags after the regression
> gate is green on the v1 baseline.

---

## 0. Prerequisites

1. NVIDIA driver 580.x or newer (CUDA 13 capable).
2. Docker Engine 24.x+ and `docker compose` plugin.
3. NVIDIA Container Toolkit installed; verify:
   ```bash
   docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
   ```
   Expected output: 4 GPUs reported, compute capability 12.0.
4. Hugging Face token cached at `~/.cache/huggingface/token` so DeepSeek V4
   Flash weights can download (model is `deepseek-ai/DeepSeek-V4-Flash`).
5. **Power cap**: leave whatever cap the platform has — the fork is tuned
   for the user's 300 W cap. Do not raise.

Disk: ~600 GB free for the model weights + caches.

---

## 1. Clone and choose your GPU layout

```bash
git clone https://your.repo.url/DS4_SM120_Flash_VLLM_Experiment.git
cd DS4_SM120_Flash_VLLM_Experiment
```

The shipped `docker-compose.yml` uses `device_ids: ["0", "2"]` and `-tp 2`
(matching the dev workstation's 2-GPU layout). On a **4x RTX PRO 6000** host,
two valid options:

### Option A — keep `tp=2`, run two services

Edit nothing. You can launch a second compose stack on the other 2 GPUs by
copying `docker-compose.yml` to `docker-compose.4plus5.yml`, changing
`container_name`, port, and `device_ids` to `["1", "3"]`, and `docker compose
-f docker-compose.4plus5.yml up -d vllm`.

### Option B — `tp=4` for higher single-request decode throughput

This is the layout the user almost certainly wants for "running DeepSeek V4
Flash on a 4x RTX PRO 6000 box". Edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["0", "1", "2", "3"]
          capabilities: [gpu]
```

and change `-tp 2` to `-tp 4` in the `exec vllm serve` command (around
line 132). Also bump:

```yaml
environment:
  CUDA_VISIBLE_DEVICES: "0,1,2,3"
```

(around line 158). Leave `--enable-expert-parallel` alone; with `tp=4` the
expert parallel layout maps to 4 ranks automatically.

> If you change `tp`, also re-check `VLLM_KV_CACHE_MEMORY_BYTES`. The default
> 8 GiB is per-rank, and 4 ranks can hold a larger total KV cache; bump if
> long-context concurrency is the goal.

For all subsequent steps this guide assumes Option B (tp=4).

---

## 2. Build the container image

```bash
docker compose build vllm
```

This bakes:
- the upstream `vllm/vllm-openai:deepseekv4-cu130` image,
- CUDA 13 dev libraries (`cuda-libraries-dev-13-0`,
  `cuda-nsight-compute-13-0`),
- `b12x==0.7.0` and `nvidia-cutlass-dsl-libs-cu13==4.4.2` (CUTE DSL runtime),
- `docker/patch_vllm_deepseekv4.py` for upfront vLLM patching.

The image is tagged `vllm-deepseekv4-sm120-deepgemm:latest`.

---

## 3. Start the service (will compile the extension on first run)

```bash
docker compose up -d vllm
docker compose logs -f vllm
```

The startup script does, in order:

1. `pip install -e .` for the DeepGEMM source tree (editable install, JIT-able).
2. Run `docker/patch_vllm_deepseekv4.py` against the installed vLLM in the
   image. This is the patcher that integrates SM120 fast paths and the new
   v2 fused decode/prefill fast-paths.
3. Export every `DG_SM120_*` knob with the AGENTS.md defaults.
4. `vllm serve deepseek-ai/DeepSeek-V4-Flash …` with FlashMLA sparse
   attention, DeepGEMM MoE backend, FP8 ds_mla KV cache, and MTP1
   speculative decoding using
   `docker/vllm_speculative_mtp1_local_argmax.json`.

Watch for these signs of a healthy startup:

- `Starting vLLM API server` followed by no traceback.
- `SpecDecoding metrics` appears in the logs (confirms MTP1 was wired).
- `INFO: Application startup complete.`

> **First-run extension build**: the editable install will JIT-build the C++
> extension on first import. This compiles all `csrc/sm120_*.cu` files,
> including the new `sm120_tf32_hc_prenorm_gemm.cu`,
> `sm120_sparse_mla_decode_v2.cu`, and `sm120_sparse_mla_prefill_v2.cu`. Expect
> ~2-3 minutes for the first compile. If you change `csrc/*` later, force a
> rebuild with:
> ```bash
> docker compose exec -T vllm bash -lc \
>   'cd /workspace/DeepGEMM && DG_FORCE_BUILD=1 MAX_JOBS=8 \
>    python3 setup.py build_ext --inplace 2>&1 | tail -50'
> ```

---

## 4. Health + tiny correctness check

```bash
curl -fsS http://127.0.0.1:8080/health
# expected: 200 OK (no body)

curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-ai/DeepSeek-V4-Flash",
    "messages":[{"role":"user","content":"Reply exactly OK."}],
    "max_tokens":8,"temperature":0
  }' | jq -r '.choices[0].message.content'
# expected: OK.
```

If `/health` 200s but the chat 4xx/5xx, check `docker compose logs vllm`
for tracebacks; almost always a patcher mismatch (unexpected vLLM source
shape) — open an issue with the offending log line.

---

## 5. Run the regression gate (Phase 0 baseline)

This runs the deterministic c1/c4/4k/8k/16k matrix described in
`scripts/_regression_gate_runner.py`:

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && \
   scripts/regression_gate.sh --baseline --label phase0_v1baseline'
```

What you should see (numbers depend on host; AGENTS.md notes representative
ranges from the user's 2-GPU dev box at 300 W cap):

- c1 256-token streaming steady decode  — typically **85-95 tok/s** on tp=2
  300 W; expect a similar or higher floor on tp=4 because the per-rank GEMM
  body shrinks.
- c4 256-token streaming aggregate      — typically **175-200 tok/s** on tp=2.
- 4k-token TTFT                         — typically **1.9-2.1 s**.
- 8k-token TTFT                         — typically **4.7-5.4 s**.
- 16k-token TTFT                        — typically **15-17 s**.

The gate exit code is **0** when all floor/cap thresholds hold. The summary
JSON is also written to `profiles/baseline_phase0_v1baseline.txt` so later
phases can diff against it.

> If the gate fails on the **v1 baseline** (no v2 flags enabled), STOP. The
> v2 flags must only be evaluated against a healthy v1 baseline. The most
> common cause is a version skew between the cached image and the patched
> vLLM source; rebuild with `docker compose build --no-cache vllm` and
> retry.

---

## 6. Validate the new HC prenorm v2 (default-on)

`DG_SM120_HC_PRENORM_V2=1` is on by default in `docker-compose.yml`.
Validate it didn't break anything:

```bash
# A) Synthetic vs scalar fallback bench
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && python3 scripts/bench_sm120_hc_prenorm.py \
     --m 16 --n 32 --k 7168 --num-splits 1'
# Expected: bit-exact (max_abs_diff < 1e-5); v2 should be at least as fast as v1.

# B) Run the regression gate again with v2 on (default).
scripts/regression_gate.sh --label phase1_hc_prenorm_v2
```

If decoder output now mismatches the v1 baseline by more than the floors,
disable v2: `DG_SM120_HC_PRENORM_V2=0 docker compose up -d --force-recreate
vllm`.

---

## 7. Validate the fused decode v2 (opt-in, single-cache only)

Synthetic correctness first:

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && python3 scripts/test_sm120_fused_decode.py \
     --batch 4 --heads 32 --topk 128 --block-size 256'
# Expected: "OK" with out_diff well within BF16 tolerance.
# The harness compares against the existing
# sm120_dequantize_and_gather_indexed_k_cache + workspace-split decode.
```

Microbench:

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && python3 scripts/bench_sm120_fused_decode.py \
     --batch 1 --heads 32 --topk 128 --block-size 256 --iters 200'
# Expected: prints v2 vs reference us; v2 should be at least neutral. The
# headline win comes after the MMA upgrade marked in csrc/sm120_sparse_mla_decode_v2.cu.
```

Now flip the live flag and re-run the regression gate. Use a temporary
override file so you can revert quickly:

```bash
cat > docker-compose.override.yml <<'YAML'
services:
  vllm:
    environment:
      DG_SM120_FUSED_DECODE_V2: "1"
      DG_SM120_FUSED_DECODE_V2_STRICT: "0"   # fall back silently on errors
YAML
docker compose up -d --force-recreate vllm
docker compose logs -f vllm   # watch for clean startup
```

After `/health` 200s and tiny correctness still returns `OK.`, run:

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && \
   scripts/regression_gate.sh --label phase2_fused_decode_v2'
```

Acceptance:
- The gate exit code is **0**.
- c1 256-token steady decode improves vs the v1 baseline (the synthetic
  harness numbers will look more dramatic than live numbers).
- No `illegal memory access` / `RuntimeError` / `OOM` lines in
  `docker compose logs vllm`.

If anything is wrong, set the strict flag to surface the underlying error:

```bash
yq -y -i '.services.vllm.environment.DG_SM120_FUSED_DECODE_V2_STRICT = "1"' \
   docker-compose.override.yml
docker compose up -d --force-recreate vllm
# repeat the failing request; the traceback will now propagate.
```

To disable v2 decode entirely, delete the override file (or set the flag to
"0") and re-up.

---

## 8. Validate the fused prefill v2 (opt-in, compressed-only)

Synthetic correctness first:

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && python3 scripts/test_sm120_fused_prefill.py \
     --tokens 16 --heads 32 --topk 512 --block-size 256'
# Expected: max output diff within BF16 tolerance vs the BF16
# workspace-split reference.
```

Now wire it up live:

```bash
cat > docker-compose.override.yml <<'YAML'
services:
  vllm:
    environment:
      DG_SM120_FUSED_DECODE_V2: "1"   # keep from step 7
      DG_SM120_FUSED_PREFILL_V2: "1"
      DG_SM120_FUSED_PREFILL_V2_STRICT: "0"
YAML
docker compose up -d --force-recreate vllm
```

Stress test with **20 unique long prompts** to flush out the kind of illegal
memory access that bit prior compiled-bridge attempts (per AGENTS.md):

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && for i in $(seq 1 20); do
     python3 - <<PY
import os, random, requests, json
random.seed(4001 + $i)
words = " ".join(
    "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(3,10)))
    for _ in range(8200))
r = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json={"model":"deepseek-ai/DeepSeek-V4-Flash",
          "messages":[{"role":"user","content":words}],
          "max_tokens":16,"temperature":0,"min_tokens":16})
r.raise_for_status()
print($i, r.json()["choices"][0]["finish_reason"])
PY
   done'
```

If any iteration fails, switch to strict mode and re-run to capture the
underlying error.

After clean stress: run the regression gate:

```bash
scripts/regression_gate.sh --label phase3_fused_prefill_v2
```

Acceptance: gate green, 8k TTFT cap held, no `illegal memory access`.

---

## 9. (Optional) Push toward 100 tok/s — Phase 4+

Once Phase 2 + Phase 3 are validated, the next bottleneck is the MoE
FP8xFP4 GEMM body. Two paths:

- **Phase 4.1**: rebase CUTLASS to a commit that includes PR #3121 and add
  the K=64 dense FP8 tile variant. See `docs/CUTLASS_REBASE.md` for the
  full procedure and risk register.
- **MMA upgrade for v2 kernels**: the `// TODO(SM120-MMA):` markers in
  `csrc/sm120_sparse_mla_decode_v2.cu` and
  `csrc/sm120_sparse_mla_prefill_v2.cu` indicate where to slot in
  `mma.sync.aligned.kind::f8f6f4.block_scale` warp-level instructions. This
  is the change that turns "structurally fused" into "tensor-core fused"
  and is expected to deliver the headline 2-4x kernel-level win called for
  in the plan (`docs/SM120_FUSED_KERNELS.md`).

Both are deliberately outside the scope of this initial integration.

---

## 10. Tearing down

```bash
docker compose down
```

`-v` to drop volumes; the `~/.cache/huggingface` and `~/.cache/pip` mounts
are deliberately bind-mounted so model weights survive across runs.

---

## Troubleshooting

| Symptom                                              | Likely cause + fix                                                                                                                          |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia-smi` shows compute_cap 8.x or 9.x, not 12.0  | These aren't RTX PRO 6000s. The build will fail; abort.                                                                                     |
| First compile fails with `unrecognized arch sm_120f` | CUDA toolkit too old (need 13.0+). The shipped image bakes 13.0; if you customized it, restore.                                             |
| `/health` 200s but chat returns garbage              | Make sure `DG_SM120_BYPASS_TP_ALLREDUCE=0`. The bypass is invalid for serving and only used for diagnostics.                                |
| Patcher fails on startup with "Could not patch"      | vLLM source shape changed in the image. Open an issue with the failing source line.                                                         |
| `DG_SM120_FUSED_*_V2` regresses TTFT/decode          | Disable the flag in `docker-compose.override.yml` and re-up. Run the synthetic harness to confirm the v2 kernel is still bit-exact.         |
| `illegal memory access` after enabling fused prefill | Set `DG_SM120_FUSED_PREFILL_V2_STRICT=1` and reproduce; capture the traceback. Likely a workspace_map shape mismatch in a corner case.      |
| 4-GPU run produces NaN / wrong tokens                | Check the override didn't break `--enable-expert-parallel`; with `tp=4` the EP layout is recomputed.                                        |
| Regression gate fails on c4 only                     | c4 is sensitive to the first-after-cold path; warm with two short c1 runs first, or rerun after a few minutes of idle.                     |

---

## Reference of files added / changed for this milestone

| File                                                       | Status   | Purpose                                              |
| ---------------------------------------------------------- | -------- | ---------------------------------------------------- |
| `csrc/sm120_tf32_hc_prenorm_gemm.cu`                       | NEW      | Tiled HC prenorm v2 (Phase 1.1)                      |
| `csrc/jit_kernels/impls/sm120_tf32_hc_prenorm_gemm.hpp`    | NEW      | Header for above                                     |
| `csrc/sm120_sparse_mla_decode_v2.cu`                       | NEW      | Fused sparse MLA decode v2 (Phase 2)                 |
| `csrc/jit_kernels/impls/sm120_sparse_mla_decode_v2.hpp`    | NEW      | Header for above                                     |
| `csrc/sm120_sparse_mla_prefill_v2.cu`                      | NEW      | Fused sparse MLA prefill v2 (Phase 3)                |
| `csrc/jit_kernels/impls/sm120_sparse_mla_prefill_v2.hpp`   | NEW      | Header for above + register_apis entry              |
| `csrc/python_api.cpp`                                      | edit     | Wire `sm120_mla_v2::register_apis(m)`                |
| `csrc/apis/hyperconnection.hpp`                            | edit     | Dispatch SM120 HC prenorm to v2 when `n <= 64`       |
| `setup.py`                                                 | edit     | Include the new .cu files in the build              |
| `docker-compose.yml`                                       | edit     | Default-on `_HC_PRENORM_V2`, opt-in `_FUSED_*_V2`    |
| `docker/patch_vllm_deepseekv4.py`                          | edit     | Fast-path branches for fused decode/prefill v2; remove unused `patch_flashinfer_mxfp4_sm120` |
| `scripts/regression_gate.sh`                               | NEW      | Phase 0 acceptance gate driver                       |
| `scripts/_regression_gate_runner.py`                       | NEW      | OpenAI-style chat completions matrix runner          |
| `scripts/test_sm120_fused_decode.py`                       | NEW      | Synthetic correctness harness for decode v2          |
| `scripts/bench_sm120_fused_decode.py`                      | NEW      | Microbench for decode v2 vs reference                |
| `scripts/test_sm120_fused_prefill.py`                      | NEW      | Synthetic correctness harness for prefill v2         |
| `scripts/bench_sm120_hc_prenorm.py`                        | NEW      | Microbench for HC prenorm v2 vs scalar fallback      |
| `docs/SM120_FUSED_KERNELS.md`                              | NEW      | Architecture + feature-flag matrix                   |
| `docs/SM120_SMEM_BUDGET.md`                                | NEW      | Per-kernel SMEM audit (Phase 1.2)                    |
| `docs/CUTLASS_REBASE.md`                                   | NEW      | Phase 4 prep (PR #3121, K=64 tile plan)              |
| `INSTALL_AND_TEST.md`                                      | NEW      | This file                                            |
| `AGENTS.md`                                                | edit     | Recent Delta entry + new flag defaults               |
