# Install + Test on a 4x RTX PRO 6000 Blackwell host (bare-metal venv)

This is the **bare-metal** counterpart to `INSTALL_AND_TEST.md`. It replaces
the Docker container with a Python venv on the host. Same goal: install this
fork, build the SM120 DeepGEMM extension (including the new fused v2
kernels), launch vLLM with DeepSeek V4 Flash, and validate the new `*_V2`
paths against the regression gate.

It assumes a Linux host with **4x NVIDIA RTX PRO 6000 Blackwell** (compute
capability SM120), ~768 GB system RAM, NVMe scratch, and a network/IP
egress that can pull container images for the one-time venv extraction step.

> **Important**: this fork is a research prototype, not an upstream-quality
> patch. The new v2 fused kernels were authored without iterative testing on
> the dev box and need on-host validation before being relied on. Follow the
> recommended order below and only flip the v2 flags after the regression
> gate is green on the v1 baseline.
>
> The Docker path in `INSTALL_AND_TEST.md` is the lower-friction option. Use
> bare-metal **only** when you need direct access to the venv (e.g. to install
> system Nsight Compute, profile with host tooling, or run vLLM under
> systemd).

---

## 0. Prerequisites

1. **NVIDIA driver 580.x or newer** (CUDA 13 capable).
   ```bash
   nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
   # expected: 4 rows, compute_cap=12.0, driver_version >= 580.xx
   ```
2. **CUDA 13.0 toolkit** (the `nvcc` matching the driver). On Ubuntu 22.04+:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get install -y cuda-toolkit-13-0 cuda-libraries-dev-13-0 cuda-nsight-compute-13-0
   ```
   Verify:
   ```bash
   /usr/local/cuda/bin/nvcc --version | head -5
   # expected: Cuda compilation tools, release 13.0
   ```
3. **System libraries** the vLLM venv links against at runtime (Ubuntu 22.04+):
   ```bash
   sudo apt-get install -y \
     build-essential git curl jq tmux \
     python3.12 python3.12-venv python3.12-dev \
     libnuma1 libnuma-dev \
     libnccl2 libnccl-dev \
     libnvidia-ml-dev \
     libibverbs1 librdmacm1
   ```
   The `python3.12*` packages are required because the upstream
   `vllm/vllm-openai:deepseekv4-cu130` image is built against CPython 3.12;
   any other minor version will break the venv extraction step below.
4. **Docker Engine 24.x+** (only needed for the one-time venv extraction in
   step 2; no daemon dependency at runtime).
5. **Hugging Face token** in `~/.cache/huggingface/token` so DeepSeek V4 Flash
   weights (`deepseek-ai/DeepSeek-V4-Flash`) can download.
6. **Power cap**: leave whatever cap the platform has. The fork is tuned for
   the user's 300 W cap; do not raise.

Disk: ~600 GB free for model weights, ~30 GB for the venv + caches, ~30 GB
for the temporary container image used in step 2.

---

## 1. Clone the repo and set environment

```bash
git clone https://your.repo.url/DS4_SM120_Flash_VLLM_Experiment.git
cd DS4_SM120_Flash_VLLM_Experiment
export REPO_ROOT="$(pwd)"

# Permanent shell exports (also written into the launcher; this is for ad-hoc
# work inside the repo, e.g. running scripts/test_*.py).
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
```

---

## 2. Extract the upstream venv from `vllm-openai:deepseekv4-cu130`

The upstream image bakes a vetted vLLM nightly with DeepSeek V4 patches,
PyTorch nightly cu130, FlashMLA, FlashInfer, and the various b12x/CUTLASS
DSL deps. Rebuilding all of those from source on the host is fragile;
instead we copy the image's `site-packages` directly into a host venv.

### 2.1. Pull the image once

```bash
docker pull vllm/vllm-openai:deepseekv4-cu130
```

### 2.2. Create a Python 3.12 venv on the host

Pick a path with plenty of disk; `/opt/vllm-sm120/venv` is a reasonable
default for a workstation, or `${REPO_ROOT}/.venv-sm120` if you want it
sibling to the source tree.

```bash
export VENV_DIR="${REPO_ROOT}/.venv-sm120"          # or /opt/vllm-sm120/venv
python3.12 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip wheel
```

### 2.3. Copy site-packages out of the upstream image

```bash
# Discover the exact site-packages path inside the image.
SP_IN_IMAGE="$(docker run --rm vllm/vllm-openai:deepseekv4-cu130 \
    python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "Image site-packages: ${SP_IN_IMAGE}"
# Typical: /usr/local/lib/python3.12/dist-packages
#       or /opt/conda/lib/python3.12/site-packages

# Determine the matching path inside the host venv.
SP_IN_VENV="$("${VENV_DIR}/bin/python3" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "Host venv site-packages: ${SP_IN_VENV}"

# Copy. We tar/untar to preserve permissions, symlinks, and *.so files.
TMP_CID="$(docker create vllm/vllm-openai:deepseekv4-cu130 /bin/true)"
docker cp "${TMP_CID}:${SP_IN_IMAGE}/." "${SP_IN_VENV}/"
docker rm -f "${TMP_CID}" >/dev/null
```

> The image may have packages installed in **both** `dist-packages` and
> `site-packages` if it stacks pip on top of apt; if your `python3 -c
> 'import sysconfig'` returns a different path, also copy from there.
> A second pass is cheap: it just overwrites with newer files.

### 2.4. Install the CUDA 13 dev / DSL extras the Dockerfile adds on top

The shipped `Dockerfile.vllm-nightly-sm120` adds two pip wheels on top of
the base image. Mirror that:

```bash
python3 -m pip install --no-deps \
    b12x==0.7.0 \
    nvidia-cutlass-dsl-libs-cu13==4.4.2
```

(Use `--no-deps` to avoid downgrading vLLM/Torch dependencies that came
with the image.)

### 2.5. Sanity-check the venv

```bash
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
# expected: 2.x.x.devNNNN+cu130   13.0
python3 -c 'import vllm; print(vllm.__version__, vllm.__file__)'
# expected: a deepseekv4-flavored version, path inside the venv
python3 -c 'import flashinfer, b12x; print("flashinfer", flashinfer.__version__); print("b12x", b12x.__version__)'
```

If any of these fail, redo step 2.3 and confirm `SP_IN_IMAGE` matches what
the image actually ships (`docker run --rm <image> ls -la
/usr/local/lib/python3.12/`).

---

## 3. Editable-install this DeepGEMM fork

```bash
cd "${REPO_ROOT}"
DG_FORCE_BUILD=0 CUDA_HOME=/usr/local/cuda \
    python3 -m pip install -e . --no-build-isolation
```

The first import will JIT-build the C++ extension, including the new fused
v2 sources (`csrc/sm120_tf32_hc_prenorm_gemm.cu`,
`csrc/sm120_sparse_mla_decode_v2.cu`,
`csrc/sm120_sparse_mla_prefill_v2.cu`). To force a rebuild after editing
any `csrc/*` file:

```bash
DG_FORCE_BUILD=1 MAX_JOBS=8 \
    python3 setup.py build_ext --inplace 2>&1 | tail -50
```

Expect ~2-3 minutes for the first compile. Build errors at this step are
almost always a CUDA toolkit / driver mismatch — re-verify step 0.2.

---

## 4. Patch the host-installed vLLM

```bash
python3 docker/patch_vllm_deepseekv4.py
```

This rewrites the venv's vLLM source in place, wiring SM120 fast paths and
the fused decode/prefill v2 fast-paths. It is **idempotent** in the sense
that re-running on already-patched files no-ops, but you must re-run it
after any `pip install -U vllm` (or after re-extracting the venv from a
newer image).

If the patcher prints `Could not patch ...`, the source shape inside the
venv has drifted from what the patcher expects. Re-extract a known-good
venv (step 2.3) before serving.

---

## 5. Choose your GPU layout

The shipped `scripts/serve_baremetal.sh` defaults to
`CUDA_VISIBLE_DEVICES=0,1,2,3` and `TP=4`, which is the layout you almost
certainly want for "running DeepSeek V4 Flash on a 4x RTX PRO 6000 box".

Two valid alternatives:

### Option A — keep `tp=4`, single service (recommended)

No edits required. Just run the launcher (next section).

### Option B — `tp=2`, two services on disjoint GPU pairs

Useful if you want to compare two configurations side-by-side or run the
regression gate on one half while serving from the other:

```bash
# Stack 1 on GPUs 0/2 (matches the dev workstation)
TP=2 CUDA_VISIBLE_DEVICES="0,2" PORT=8080 ./scripts/serve_baremetal.sh

# Stack 2 on GPUs 1/3 (separate tmux pane / systemd unit)
TP=2 CUDA_VISIBLE_DEVICES="1,3" PORT=8081 ./scripts/serve_baremetal.sh
```

> If you change `TP`, also re-check `VLLM_KV_CACHE_MEMORY_BYTES`. The
> default 8 GiB is per-rank, and 4 ranks can hold a larger total KV cache;
> bump `VLLM_KV_CACHE_MEMORY_BYTES` if long-context concurrency is the
> goal.

---

## 6. Start the service

The launcher mirrors the inline `command:` block from `docker-compose.yml`
exactly — same env defaults, same `vllm serve` flags, same speculative
decode config (MTP1 + local argmax) — but runs the host venv directly with
no container layer.

Recommended pattern: **tmux** so the foreground log is easy to attach
to / detach from.

```bash
tmux new -s vllm

# inside the tmux session:
cd "${REPO_ROOT}"
export VENV_DIR="${REPO_ROOT}/.venv-sm120"
./scripts/serve_baremetal.sh
```

Watch for these signs of a healthy startup:

- `Starting vLLM API server` followed by no traceback.
- `SpecDecoding metrics` appears in the logs (confirms MTP1 was wired).
- `INFO:     Application startup complete.`

Detach: `Ctrl-b d`. Reattach: `tmux attach -t vllm`.

### 6.1. (Optional) systemd unit

For unattended hosts, drop the following at `/etc/systemd/system/vllm-sm120.service`:

```ini
[Unit]
Description=vLLM DeepSeek V4 Flash (SM120 / DeepGEMM fork)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/DS4_SM120_Flash_VLLM_Experiment
Environment="VENV_DIR=/home/YOUR_USER/DS4_SM120_Flash_VLLM_Experiment/.venv-sm120"
Environment="HUGGING_FACE_HUB_TOKEN_FILE=/home/YOUR_USER/.cache/huggingface/token"
ExecStart=/home/YOUR_USER/DS4_SM120_Flash_VLLM_Experiment/scripts/serve_baremetal.sh
Restart=on-failure
RestartSec=15
TimeoutStartSec=3600
LimitMEMLOCK=infinity
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-sm120.service
journalctl -u vllm-sm120.service -f
```

---

## 7. Health + tiny correctness check

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

If `/health` 200s but the chat 4xx/5xx, attach to the tmux session (or
`journalctl -u vllm-sm120 --since=-5m`) for tracebacks. The two most
common bare-metal-only failure modes:

1. **Patcher not run after a venv refresh.** Re-run step 4.
2. **Wrong CUDA in `LD_LIBRARY_PATH`.** Verify `ldconfig -p | grep cudart`
   resolves to `/usr/local/cuda/lib64` (or wherever you installed CUDA 13)
   and not a stale CUDA 12 install.

---

## 8. Run the regression gate (Phase 0 baseline)

This runs the deterministic c1/c4/4k/8k/16k matrix described in
`scripts/_regression_gate_runner.py`:

```bash
cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
scripts/regression_gate.sh --baseline --label phase0_v1baseline
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
> v2 flags must only be evaluated against a healthy v1 baseline. Most
> common cause: a stale or partially-extracted venv. Re-do step 2.3 (it's
> safe to overwrite the existing venv `site-packages`), then re-run step 4.

---

## 9. Validate the new HC prenorm v2 (default-on)

`DG_SM120_HC_PRENORM_V2=1` is on by default in `scripts/serve_baremetal.sh`.
Validate it didn't break anything:

```bash
# A) Synthetic vs scalar fallback bench
python3 scripts/bench_sm120_hc_prenorm.py --m 16 --n 32 --k 7168 --num-splits 1
# Expected: bit-exact (max_abs_diff < 1e-5); v2 should be at least as fast as v1.

# B) Run the regression gate again with v2 on (default).
scripts/regression_gate.sh --label phase1_hc_prenorm_v2
```

If decoder output now mismatches the v1 baseline by more than the floors,
disable v2 by stopping the server and re-launching with the flag flipped:

```bash
# tmux pane: Ctrl-c the running launcher
DG_SM120_HC_PRENORM_V2=0 ./scripts/serve_baremetal.sh
```

Or for systemd: add `Environment="DG_SM120_HC_PRENORM_V2=0"` to the unit
file and `systemctl daemon-reload && systemctl restart vllm-sm120`.

---

## 10. Validate the fused decode v2 (opt-in, single-cache only)

Synthetic correctness first:

```bash
python3 scripts/test_sm120_fused_decode.py \
    --batch 4 --heads 32 --topk 128 --block-size 256
# Expected: "OK" with out_diff well within BF16 tolerance.
# The harness compares against the existing
# sm120_dequantize_and_gather_indexed_k_cache + workspace-split decode.
```

Microbench:

```bash
python3 scripts/bench_sm120_fused_decode.py \
    --batch 1 --heads 32 --topk 128 --block-size 256 --iters 200
# Expected: prints v2 vs reference us; v2 should be at least neutral. The
# headline win comes after the MMA upgrade marked in
# csrc/sm120_sparse_mla_decode_v2.cu.
```

Now flip the live flag and re-run the regression gate. Restart the server
with the flag enabled:

```bash
# tmux pane: Ctrl-c the running launcher
DG_SM120_FUSED_DECODE_V2=1 \
DG_SM120_FUSED_DECODE_V2_STRICT=0 \
./scripts/serve_baremetal.sh
```

After `/health` 200s and the tiny correctness still returns `OK.`, run:

```bash
scripts/regression_gate.sh --label phase2_fused_decode_v2
```

Acceptance:
- The gate exit code is **0**.
- c1 256-token steady decode improves vs the v1 baseline (the synthetic
  harness numbers will look more dramatic than live numbers).
- No `illegal memory access` / `RuntimeError` / `OOM` lines in the tmux
  log or `journalctl -u vllm-sm120`.

If anything is wrong, set the strict flag to surface the underlying error:

```bash
# tmux pane: Ctrl-c the running launcher
DG_SM120_FUSED_DECODE_V2=1 \
DG_SM120_FUSED_DECODE_V2_STRICT=1 \
./scripts/serve_baremetal.sh
# repeat the failing request; the traceback will now propagate.
```

To disable v2 decode entirely, restart without the flag (or set it to `0`).

---

## 11. Validate the fused prefill v2 (opt-in, compressed-only)

Synthetic correctness first:

```bash
python3 scripts/test_sm120_fused_prefill.py \
    --tokens 16 --heads 32 --topk 512 --block-size 256
# Expected: max output diff within BF16 tolerance vs the BF16
# workspace-split reference.
```

Now wire it up live (keep decode v2 from step 10):

```bash
# tmux pane: Ctrl-c the running launcher
DG_SM120_FUSED_DECODE_V2=1 \
DG_SM120_FUSED_PREFILL_V2=1 \
DG_SM120_FUSED_PREFILL_V2_STRICT=0 \
./scripts/serve_baremetal.sh
```

Stress test with **20 unique long prompts** to flush out the kind of
illegal memory access that bit prior compiled-bridge attempts (per
AGENTS.md):

```bash
for i in $(seq 1 20); do
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
done
```

If any iteration fails, restart with `DG_SM120_FUSED_PREFILL_V2_STRICT=1`
and reproduce to capture the underlying error.

After clean stress, run the regression gate:

```bash
scripts/regression_gate.sh --label phase3_fused_prefill_v2
```

Acceptance: gate green, 8k TTFT cap held, no `illegal memory access`.

---

## 12. (Optional) Push toward 100 tok/s — Phase 4+

Once Phase 2 + Phase 3 are validated, the next bottleneck is the MoE
FP8xFP4 GEMM body. Two paths:

- **Phase 4.1**: rebase CUTLASS to a commit that includes PR #3121 and add
  the K=64 dense FP8 tile variant. See `docs/CUTLASS_REBASE.md` for the
  full procedure and risk register.
- **MMA upgrade for v2 kernels**: the `// TODO(SM120-MMA):` markers in
  `csrc/sm120_sparse_mla_decode_v2.cu` and
  `csrc/sm120_sparse_mla_prefill_v2.cu` indicate where to slot in
  `mma.sync.aligned.kind::f8f6f4.block_scale` warp-level instructions.
  This is the change that turns "structurally fused" into "tensor-core
  fused" and is expected to deliver the headline 2-4x kernel-level win
  called for in the plan (`docs/SM120_FUSED_KERNELS.md`).

Both are deliberately outside the scope of this initial integration.

---

## 13. Tearing down

Stop the server:

```bash
# tmux pane: Ctrl-c the running launcher
# or for systemd:
sudo systemctl stop vllm-sm120.service
sudo systemctl disable vllm-sm120.service   # optional
```

Remove the venv and editable install:

```bash
deactivate 2>/dev/null || true
rm -rf "${VENV_DIR}"
# Editable install metadata only — does not delete the source tree.
rm -rf "${REPO_ROOT}/build" "${REPO_ROOT}"/*.egg-info
```

Model weights / pip cache:

```bash
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash
# pip cache:
rm -rf ~/.cache/pip
```

The temporary container image used to seed the venv can be removed once
the venv is healthy:

```bash
docker image rm vllm/vllm-openai:deepseekv4-cu130
```

---

## Troubleshooting

| Symptom                                              | Likely cause + fix                                                                                                                          |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia-smi` shows compute_cap 8.x or 9.x, not 12.0  | These aren't RTX PRO 6000s. The build will fail; abort.                                                                                     |
| First compile fails with `unrecognized arch sm_120f` | CUDA toolkit too old (need 13.0+). Re-run step 0.2; verify `nvcc --version`.                                                                |
| `python3 -c 'import vllm'` fails with `ImportError`  | Venv extract was incomplete. Re-run step 2.3 with the correct `SP_IN_IMAGE` discovered via the image's `sysconfig`.                         |
| `.so` files missing / wrong glibc                    | Host glibc is too old vs the upstream image's. Upgrade Ubuntu (or use a Debian/Ubuntu host whose glibc matches the image's).                |
| `OSError: libcudart.so.13: cannot open`              | `LD_LIBRARY_PATH` not pointing at CUDA 13, or `ldconfig` cache stale. Run `sudo ldconfig` and verify `/etc/ld.so.conf.d/cuda*.conf`.        |
| `/health` 200s but chat returns garbage              | Make sure `DG_SM120_BYPASS_TP_ALLREDUCE=0`. The bypass is invalid for serving and only used for diagnostics.                                |
| Patcher fails on startup with "Could not patch"      | vLLM source shape changed in the venv. Re-extract a known-good venv (step 2.3), then re-run step 4. Open an issue with the failing line.    |
| `DG_SM120_FUSED_*_V2` regresses TTFT/decode          | Restart without the flag and run the synthetic harness to confirm the v2 kernel is still bit-exact.                                          |
| `illegal memory access` after enabling fused prefill | Restart with `DG_SM120_FUSED_PREFILL_V2_STRICT=1` and reproduce; capture the traceback. Likely a workspace_map shape mismatch in a corner case. |
| 4-GPU run produces NaN / wrong tokens                | Check NCCL: `python3 -c 'import torch.distributed as d; print(d.is_nccl_available())'`. Then verify `libnccl2` is installed and matches the venv's bundled NCCL. |
| Regression gate fails on c4 only                     | c4 is sensitive to the first-after-cold path; warm with two short c1 runs first, or rerun after a few minutes of idle.                     |
| Docker pull of `vllm-openai:deepseekv4-cu130` fails  | The image is hosted on Docker Hub by the upstream vLLM project; verify network egress and `docker login` if your registry requires it.      |

---

## Container vs. bare-metal at a glance

| Aspect                  | Container (`INSTALL_AND_TEST.md`)             | Bare-metal venv (this doc)                                |
| ----------------------- | --------------------------------------------- | --------------------------------------------------------- |
| Setup time              | ~10 min (image pull + first build)            | ~25 min (toolkit install + venv extract + first build)    |
| Reproducibility         | High (image is the entire env)                | Medium (host CUDA / glibc / NCCL must match)              |
| `vllm serve` invocation | `docker compose up -d vllm`                   | `./scripts/serve_baremetal.sh` (in tmux or systemd)       |
| Iterating on `csrc/*`   | `docker compose exec ... build_ext --inplace` | `DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace`   |
| Iterating on patcher    | `docker compose up --force-recreate`          | `python3 docker/patch_vllm_deepseekv4.py` then restart    |
| Host tooling (Nsight)   | Requires Nsight in the image                  | Native (apt-get install)                                  |
| Multi-tenant on host    | Trivial via compose stacks                    | Manual (separate venvs / users / systemd units)           |
| CI/CD friendliness      | High                                          | Medium                                                    |

The container path is the **safer** default for this fork. Use bare-metal
when host tooling or systemd integration matters more than image-level
reproducibility.

---

## Reference of files added / changed for this milestone

The bare-metal path uses the same source tree as the container path; the
only files specific to this guide are:

| File                                | Status   | Purpose                                                     |
| ----------------------------------- | -------- | ----------------------------------------------------------- |
| `scripts/serve_baremetal.sh`        | NEW      | Launcher mirroring `docker-compose.yml`'s command for venv  |
| `Install_and_Test_BareMetal.md`     | NEW      | This file                                                   |

For the full list of v2-kernel-related changes (kernels, bench scripts,
docs, patcher updates), see the table at the bottom of
`INSTALL_AND_TEST.md`.
