#!/usr/bin/env bash
# Bare-metal launcher for DeepSeek V4 Flash on SM120 with the DeepGEMM fork.
#
# Mirrors the inline `command:` block in docker-compose.yml so that a host
# venv install can serve the same configuration as the container path.
#
# Assumes:
#   - A Python venv is already activated (or VENV_DIR is set).
#   - `pip install -e .` for this DeepGEMM tree has already been run inside the
#     venv.
#   - `python3 docker/patch_vllm_deepseekv4.py` has already been run against the
#     vLLM installed in the venv (re-run after any vLLM upgrade).
#
# Override any of HOST / PORT / TP / CUDA_VISIBLE_DEVICES / VLLM_* / DG_SM120_*
# via environment before invocation, e.g.:
#   TP=4 CUDA_VISIBLE_DEVICES="0,1,2,3" PORT=8080 ./scripts/serve_baremetal.sh
#
# Run under tmux/screen or as a systemd unit; the foreground process logs to
# stdout (use `tee` if you want a file copy).

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate repo root so VLLM_SPECULATIVE_CONFIG_FILE resolves correctly even
# when the script is invoked from a different working directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Optional: auto-activate a venv if VENV_DIR is set and we're not already in one
# ---------------------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" && -n "${VENV_DIR:-}" ]]; then
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  else
    echo "VENV_DIR='${VENV_DIR}' has no bin/activate; aborting." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Host / device defaults (override via env). 4 GPUs + tp=4 is the recommended
# layout for "DeepSeek V4 Flash on a 4x RTX PRO 6000 box".
# ---------------------------------------------------------------------------
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"
export TP="${TP:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Where to load the model from. MODEL_PATH may be a local directory (full HF
# snapshot incl. config.json, safetensors, tokenizer, and the DeepSeek V4 *.py
# files) or an HF Hub repo id. Default falls back to HF Hub download; override
# to your local snapshot path, e.g.:
#   MODEL_PATH=/media/fmodels/deepseek-ai/DeepSeek-V4-Flash ./scripts/serve_baremetal.sh
export MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}"
# Stable served name (clients POST to `model: <MODEL_NAME>`). Leave alone unless
# you know your clients will retarget; the regression gate hard-codes the HF id.
export MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-V4-Flash}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export VLLM_TARGET_DEVICE=cuda

# ---------------------------------------------------------------------------
# DeepGEMM SM120 knobs (mirror docker-compose.yml's `command:` exports).
# Anything you don't override falls back to the AGENTS.md default.
# ---------------------------------------------------------------------------
export VLLM_USE_DEEP_GEMM=1
export VLLM_USE_DEEP_GEMM_E8M0=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=600000
export TILELANG_CLEANUP_TEMP_FILES=1

export DG_SM120_SKIP_PADDED_ZERO=1
export DG_SM120_FP4_COLS_PER_BLOCK=8
export DG_SM120_FULL_CONTEXT_BF16_BMM=1
export DG_SM120_INDEXED_BF16_BMM=1
export DG_SM120_WORKSPACE_FUSED_ATTENTION=1
export DG_SM120_WORKSPACE_SPLIT_ATTENTION="${DG_SM120_WORKSPACE_SPLIT_ATTENTION:-1}"

export DG_SM120_ENABLE_FP8_M1_KBLOCK="${DG_SM120_ENABLE_FP8_M1_KBLOCK:-1}"
export DG_SM120_ENABLE_FP8_M1_WARPCOL_HEURISTIC="${DG_SM120_ENABLE_FP8_M1_WARPCOL_HEURISTIC:-1}"
export DG_SM120_ENABLE_FP8_M1_FUSED="${DG_SM120_ENABLE_FP8_M1_FUSED:-1}"
export DG_SM120_ENABLE_BHR_M1_MMA="${DG_SM120_ENABLE_BHR_M1_MMA:-1}"
export DG_SM120_MOE_SKIP_SFA_FILL="${DG_SM120_MOE_SKIP_SFA_FILL:-1}"
export DG_SM120_MOE_SKIP_SFA_FILL_MAX_M="${DG_SM120_MOE_SKIP_SFA_FILL_MAX_M:-16}"
export DG_SM120_MOE_ROW_GROUPED="${DG_SM120_MOE_ROW_GROUPED:-1}"
export DG_SM120_MOE_ROW_GROUPED_MAX_M="${DG_SM120_MOE_ROW_GROUPED_MAX_M:-6}"
export DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL="${DG_SM120_MOE_ROW_GROUPED_SKIP_SFA_FILL:-1}"
export DG_SM120_MOE_DIRECT_GROUPS_WHEN_NO_SHRINK="${DG_SM120_MOE_DIRECT_GROUPS_WHEN_NO_SHRINK:-1}"
export DG_SM120_CACHE_FP8_SFB="${DG_SM120_CACHE_FP8_SFB:-1}"
export DG_SM120_CACHE_FP8_SFA_FILL="${DG_SM120_CACHE_FP8_SFA_FILL:-1}"
export DG_SM120_SPEC_ARGMAX_FASTPATH="${DG_SM120_SPEC_ARGMAX_FASTPATH:-1}"
export DG_SM120_BHR_D_PER_BLOCK="${DG_SM120_BHR_D_PER_BLOCK:-1}"
export DG_SM120_BHR_KBLOCK_SCALE="${DG_SM120_BHR_KBLOCK_SCALE:-1}"
export DG_SM120_ENABLE_BHR_WARPCOL_HEURISTIC="${DG_SM120_ENABLE_BHR_WARPCOL_HEURISTIC:-1}"
export DG_SM120_BHR_WARPCOL_WARPS="${DG_SM120_BHR_WARPCOL_WARPS:-4}"
export DG_SM120_BHR_WARPCOL_COLS="${DG_SM120_BHR_WARPCOL_COLS:-0}"
export DG_SM120_ENABLE_SMALL_M_MMA=0
export DG_SM120_MHC_REUSE_BUFFERS="${DG_SM120_MHC_REUSE_BUFFERS:-1}"
export DG_SM120_MAIN_TOPK_CAP="${DG_SM120_MAIN_TOPK_CAP:-128}"
export DG_SM120_ACTIVE_HEADS=32
export DG_SM120_SEQUENTIAL_COMPRESSOR="${DG_SM120_SEQUENTIAL_COMPRESSOR:-1}"
export DG_SM120_FLASHMLA_PREFILL_WORKSPACE_FACTOR="${DG_SM120_FLASHMLA_PREFILL_WORKSPACE_FACTOR:-1}"

export DG_SM120_PREFILL_WORKSPACE_CHUNK="${DG_SM120_PREFILL_WORKSPACE_CHUNK:-256}"
export DG_SM120_PREFILL_TORCH_BMM="${DG_SM120_PREFILL_TORCH_BMM:-1}"
export DG_SM120_PREFILL_TORCH_COMPILE="${DG_SM120_PREFILL_TORCH_COMPILE:-0}"
export DG_SM120_PREFILL_TORCH_COMPILE_MIN_ROWS="${DG_SM120_PREFILL_TORCH_COMPILE_MIN_ROWS:-64}"
export DG_SM120_PREFILL_CUDNN="${DG_SM120_PREFILL_CUDNN:-0}"
export DG_SM120_PREFILL_CUDNN_UNMASKED="${DG_SM120_PREFILL_CUDNN_UNMASKED:-0}"
export DG_SM120_PREFILL_TRUST_INDICES="${DG_SM120_PREFILL_TRUST_INDICES:-1}"
export DG_SM120_PREFILL_TRIM_TOPK_MIN_WIDTH="${DG_SM120_PREFILL_TRIM_TOPK_MIN_WIDTH:-2048}"
export DG_SM120_PREFILL_TRIM_TOPK="${DG_SM120_PREFILL_TRIM_TOPK:-1}"
export DG_SM120_PREFILL_DYNAMIC_COMPRESSED_N="${DG_SM120_PREFILL_DYNAMIC_COMPRESSED_N:-1}"
export DG_SM120_PREFILL_EMPTY_COMBINED_INDICES="${DG_SM120_PREFILL_EMPTY_COMBINED_INDICES:-1}"
export DG_SM120_PREFILL_INDEXED_SPLIT="${DG_SM120_PREFILL_INDEXED_SPLIT:-0}"
export DG_SM120_PREFILL_INDEX_SELECT="${DG_SM120_PREFILL_INDEX_SELECT:-0}"
export DG_SM120_PREFILL_GATHER_WORKSPACE="${DG_SM120_PREFILL_GATHER_WORKSPACE:-0}"
export DG_SM120_PREFILL_DIRECT_FP8_MAP="${DG_SM120_PREFILL_DIRECT_FP8_MAP:-0}"
export DG_SM120_PREFILL_DIRECT_FP8_MAP_STRICT="${DG_SM120_PREFILL_DIRECT_FP8_MAP_STRICT:-0}"

# v2 fused kernels (Phase 2 / 3 / 1.1 from Install_and_Test_BareMetal.md).
# Decode/prefill v2 stay default-off until live validation; HC prenorm v2 is
# safe default-on.
export DG_SM120_FUSED_DECODE_V2="${DG_SM120_FUSED_DECODE_V2:-0}"
export DG_SM120_FUSED_DECODE_V2_STRICT="${DG_SM120_FUSED_DECODE_V2_STRICT:-0}"
export DG_SM120_FUSED_PREFILL_V2="${DG_SM120_FUSED_PREFILL_V2:-0}"
export DG_SM120_FUSED_PREFILL_V2_STRICT="${DG_SM120_FUSED_PREFILL_V2_STRICT:-0}"
export DG_SM120_HC_PRENORM_V2="${DG_SM120_HC_PRENORM_V2:-1}"

export DG_SM120_ENABLE_B12X_MOE="${DG_SM120_ENABLE_B12X_MOE:-0}"
export DG_SM120_B12X_A1_GS="${DG_SM120_B12X_A1_GS:-1.0}"
export DG_SM120_B12X_A2_GS="${DG_SM120_B12X_A2_GS:-1.0}"
export DG_SM120_BYPASS_TP_ALLREDUCE="${DG_SM120_BYPASS_TP_ALLREDUCE:-0}"

export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export B12X_CUTE_COMPILE_CACHE_DIR="${B12X_CUTE_COMPILE_CACHE_DIR:-/tmp/b12x_cute_cache}"

export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
export VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE="${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE:-8}"
export VLLM_PERFORMANCE_MODE="${VLLM_PERFORMANCE_MODE:-balanced}"
export VLLM_OPTIMIZATION_LEVEL="${VLLM_OPTIMIZATION_LEVEL:-2}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.980}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-8589934592}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# Default speculative-config to MTP1 + local-argmax (matches compose default).
# Set VLLM_SPECULATIVE_CONFIG_FILE="" before invoking to disable.
export VLLM_SPECULATIVE_CONFIG_FILE="${VLLM_SPECULATIVE_CONFIG_FILE:-${REPO_ROOT}/docker/vllm_speculative_mtp1_local_argmax.json}"
export VLLM_SPECULATIVE_CONFIG_JSON="${VLLM_SPECULATIVE_CONFIG_JSON:-}"
export VLLM_PROFILER_CONFIG_FILE="${VLLM_PROFILER_CONFIG_FILE:-}"
export VLLM_PROFILER_CONFIG_JSON="${VLLM_PROFILER_CONFIG_JSON:-}"

# Diagnostic profiling defaults (off for real benchmarks).
export DG_SM120_INSTALL_PROFILE_PATCH="${DG_SM120_INSTALL_PROFILE_PATCH:-0}"
export DG_SM120_INSTALL_MOE_PROFILE_PATCH="${DG_SM120_INSTALL_MOE_PROFILE_PATCH:-0}"
export DG_SM120_PROFILE_LAYER="${DG_SM120_PROFILE_LAYER:-0}"
export DG_SM120_PROFILE_EVERY="${DG_SM120_PROFILE_EVERY:-2048}"
export DG_SM120_KERNEL_PROFILE="${DG_SM120_KERNEL_PROFILE:-0}"
export DG_SM120_KERNEL_PROFILE_EVERY="${DG_SM120_KERNEL_PROFILE_EVERY:-256}"
export DG_SM120_KERNEL_PROFILE_PATH="${DG_SM120_KERNEL_PROFILE_PATH:-/tmp/dg_sm120_kernel_profile.tsv}"
export DG_SM120_PROFILE_MOE_SHAPES="${DG_SM120_PROFILE_MOE_SHAPES:-0}"
export DG_SM120_PROFILE_MOE_EVERY="${DG_SM120_PROFILE_MOE_EVERY:-1}"
export DG_SM120_MOE_PROFILE_PATH="${DG_SM120_MOE_PROFILE_PATH:-/tmp/dg_sm120_moe_shapes.jsonl}"

# ---------------------------------------------------------------------------
# Build the dynamic argument list (mirrors the compose set -- /case/if blocks).
# ---------------------------------------------------------------------------
extra_args=()

case "$(printenv VLLM_ENFORCE_EAGER)" in
  1|true|TRUE|yes|YES|on|ON)
    extra_args+=(--enforce-eager)
    ;;
esac

if [[ -n "$(printenv VLLM_KV_CACHE_MEMORY_BYTES)" ]]; then
  extra_args+=(--kv-cache-memory-bytes "$(printenv VLLM_KV_CACHE_MEMORY_BYTES)")
fi

if [[ -n "$(printenv VLLM_PROFILER_CONFIG_FILE)" ]]; then
  extra_args+=(--profiler-config "$(cat "$(printenv VLLM_PROFILER_CONFIG_FILE)")")
elif [[ -n "$(printenv VLLM_PROFILER_CONFIG_JSON)" ]]; then
  extra_args+=(--profiler-config "$(printenv VLLM_PROFILER_CONFIG_JSON)")
fi

if [[ -n "$(printenv VLLM_SPECULATIVE_CONFIG_FILE)" ]]; then
  if [[ -f "$(printenv VLLM_SPECULATIVE_CONFIG_FILE)" ]]; then
    extra_args+=(--speculative-config "$(cat "$(printenv VLLM_SPECULATIVE_CONFIG_FILE)")")
  else
    echo "WARN: VLLM_SPECULATIVE_CONFIG_FILE='${VLLM_SPECULATIVE_CONFIG_FILE}' not found; serving without --speculative-config." >&2
  fi
elif [[ -n "$(printenv VLLM_SPECULATIVE_CONFIG_JSON)" ]]; then
  extra_args+=(--speculative-config "$(printenv VLLM_SPECULATIVE_CONFIG_JSON)")
fi

# Print a small banner so logs are easy to grep.
echo "[serve_baremetal] cwd=${REPO_ROOT}"
echo "[serve_baremetal] MODEL_PATH=${MODEL_PATH}  MODEL_NAME=${MODEL_NAME}"
echo "[serve_baremetal] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} TP=${TP} PORT=${PORT}"
echo "[serve_baremetal] VLLM_SPECULATIVE_CONFIG_FILE=${VLLM_SPECULATIVE_CONFIG_FILE:-<unset>}"
echo "[serve_baremetal] DG_SM120_HC_PRENORM_V2=${DG_SM120_HC_PRENORM_V2}  DG_SM120_FUSED_DECODE_V2=${DG_SM120_FUSED_DECODE_V2}  DG_SM120_FUSED_PREFILL_V2=${DG_SM120_FUSED_PREFILL_V2}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --trust-remote-code \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend deep_gemm \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --enable-expert-parallel \
  -tp "${TP}" \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --max-num-partial-prefills 1 \
  --max-long-partial-prefills 1 \
  --enable-chunked-prefill \
  --optimization-level "${VLLM_OPTIMIZATION_LEVEL}" \
  --performance-mode "${VLLM_PERFORMANCE_MODE}" \
  --max-cudagraph-capture-size "${VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --disable-uvicorn-access-log \
  ${VLLM_EXTRA_ARGS} \
  "${extra_args[@]}" \
  "$@"
