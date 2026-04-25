#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-all}"
RUNNER="${RUNNER:-exec}"
SERVICE="${SERVICE:-vllm}"
COMPOSE="${COMPOSE:-docker compose}"
WORKDIR_IN_CONTAINER="${WORKDIR_IN_CONTAINER:-/workspace/DeepGEMM}"
MAX_JOBS="${MAX_JOBS:-16}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
BENCH_DEVICE="${BENCH_DEVICE:-cuda}"

ATTN_BATCHES="${ATTN_BATCHES:-1 2 4}"
ATTN_HEADS="${ATTN_HEADS:-32}"
ATTN_TOPKS="${ATTN_TOPKS:-128 512 768}"
ATTN_ITERS="${ATTN_ITERS:-200}"
ATTN_WARMUP="${ATTN_WARMUP:-20}"

MOE_MS="${MOE_MS:-1 2 4 8 16}"
MOE_N="${MOE_N:-7168}"
MOE_K="${MOE_K:-2048}"
MOE_GROUPS="${MOE_GROUPS:-128}"
MOE_ITERS="${MOE_ITERS:-200}"
MOE_WARMUP="${MOE_WARMUP:-20}"

DIRECT_BATCHES="${DIRECT_BATCHES:-1 2 4}"
DIRECT_HEADS="${DIRECT_HEADS:-32}"
DIRECT_TOPKS="${DIRECT_TOPKS:-128 512 768}"
DIRECT_NUM_BLOCKS="${DIRECT_NUM_BLOCKS:-64}"
DIRECT_BLOCK_SIZE="${DIRECT_BLOCK_SIZE:-256}"
DIRECT_ITERS="${DIRECT_ITERS:-200}"
DIRECT_WARMUP="${DIRECT_WARMUP:-20}"

FULLCTX_BATCHES="${FULLCTX_BATCHES:-1 2 4}"
FULLCTX_HEADS="${FULLCTX_HEADS:-32}"
FULLCTX_SEQS="${FULLCTX_SEQS:-128 256 512 768}"
FULLCTX_BLOCK_SIZE="${FULLCTX_BLOCK_SIZE:-256}"
FULLCTX_ITERS="${FULLCTX_ITERS:-200}"
FULLCTX_WARMUP="${FULLCTX_WARMUP:-20}"

usage() {
  cat <<'EOF'
Usage:
  scripts/dev_sm120_kernel_loop.sh [build|attn|direct|fullctx|moe|all|shell]

Purpose:
  Fast SM120 kernel development loop without reloading DeepSeek V4 in vLLM.

How it works:
  - Rebuilds only DeepGEMM's torch extension with build_ext --inplace.
  - Runs each benchmark in a fresh Python process so it imports the new .so.
  - Does not restart the vLLM server. The loaded model keeps using the old
    extension until vLLM workers are restarted, but standalone benches use the
    newly built extension immediately.

Environment:
  RUNNER=exec|host      Default: exec. exec runs inside docker compose service.
  SERVICE=vllm         Compose service name for RUNNER=exec.
  MAX_JOBS=16          Parallel build jobs.

Attention workspace bench knobs:
  ATTN_BATCHES="1 2 4"
  ATTN_HEADS=32
  ATTN_TOPKS="128 512 768"
  ATTN_ITERS=200
  ATTN_WARMUP=20

Direct indexed-cache decode knobs:
  DIRECT_BATCHES="1 2 4"
  DIRECT_HEADS=32
  DIRECT_TOPKS="128 512 768"
  DIRECT_NUM_BLOCKS=64
  DIRECT_BLOCK_SIZE=256
  DIRECT_ITERS=200
  DIRECT_WARMUP=20

Direct full-context decode knobs:
  FULLCTX_BATCHES="1 2 4"
  FULLCTX_HEADS=32
  FULLCTX_SEQS="128 256 512 768"
  FULLCTX_BLOCK_SIZE=256
  FULLCTX_ITERS=200
  FULLCTX_WARMUP=20

MoE bench knobs:
  MOE_MS="1 2 4 8 16"
  MOE_N=7168
  MOE_K=2048
  MOE_GROUPS=128
  MOE_ITERS=200
  MOE_WARMUP=20

Examples:
  scripts/dev_sm120_kernel_loop.sh all
  MODE=attn ATTN_TOPKS="128 768" scripts/dev_sm120_kernel_loop.sh
  MODE=direct DIRECT_TOPKS="128 512" scripts/dev_sm120_kernel_loop.sh
  MODE=fullctx FULLCTX_SEQS="256 512" scripts/dev_sm120_kernel_loop.sh
  MODE=moe MOE_MS="1 2 4" scripts/dev_sm120_kernel_loop.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  MODE="$1"
fi

run_host() {
  "$@"
}

run_exec() {
  ${COMPOSE} exec -T "${SERVICE}" bash -lc "$*"
}

run_cmd() {
  case "${RUNNER}" in
    host)
      run_host bash -lc "$*"
      ;;
    exec)
      run_exec "$*"
      ;;
    *)
      echo "ERROR: RUNNER must be 'exec' or 'host', got '${RUNNER}'" >&2
      exit 2
      ;;
  esac
}

prefix() {
  if [[ "${RUNNER}" == "exec" ]]; then
    printf 'cd %q && ' "${WORKDIR_IN_CONTAINER}"
  else
    printf ''
  fi
}

build_ext() {
  local p
  p="$(prefix)"
  run_cmd "${p}CUDA_HOME=${CUDA_HOME} MAX_JOBS=${MAX_JOBS} DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace"
}

bench_attn() {
  local p batch topk
  p="$(prefix)"
  for batch in ${ATTN_BATCHES}; do
    for topk in ${ATTN_TOPKS}; do
      echo "=== attn batch=${batch} heads=${ATTN_HEADS} topk=${topk} ==="
      run_cmd "${p}python3 scripts/bench_sm120_workspace_attention.py --batch ${batch} --heads ${ATTN_HEADS} --topk ${topk} --warmup ${ATTN_WARMUP} --iters ${ATTN_ITERS} --device ${BENCH_DEVICE}"
    done
  done
}

bench_moe() {
  local p m
  p="$(prefix)"
  for m in ${MOE_MS}; do
    echo "=== moe m=${m} n=${MOE_N} k=${MOE_K} groups=${MOE_GROUPS} ==="
    run_cmd "${p}python3 scripts/bench_sm120_moe_small_m.py --m ${m} --n ${MOE_N} --k ${MOE_K} --groups ${MOE_GROUPS} --warmup ${MOE_WARMUP} --iters ${MOE_ITERS} --device ${BENCH_DEVICE}"
  done
}

bench_direct() {
  local p batch topk
  p="$(prefix)"
  for batch in ${DIRECT_BATCHES}; do
    for topk in ${DIRECT_TOPKS}; do
      echo "=== direct batch=${batch} heads=${DIRECT_HEADS} topk=${topk} ==="
      run_cmd "${p}python3 scripts/bench_sm120_fp8_cache_decode.py --batch ${batch} --heads ${DIRECT_HEADS} --topk ${topk} --num-blocks ${DIRECT_NUM_BLOCKS} --block-size ${DIRECT_BLOCK_SIZE} --warmup ${DIRECT_WARMUP} --iters ${DIRECT_ITERS} --device ${BENCH_DEVICE}"
    done
  done
}

bench_fullctx() {
  local p batch seq
  p="$(prefix)"
  for batch in ${FULLCTX_BATCHES}; do
    for seq in ${FULLCTX_SEQS}; do
      echo "=== fullctx batch=${batch} heads=${FULLCTX_HEADS} seq=${seq} ==="
      run_cmd "${p}python3 scripts/bench_sm120_fp8_full_context_decode.py --batch ${batch} --heads ${FULLCTX_HEADS} --seq-len ${seq} --block-size ${FULLCTX_BLOCK_SIZE} --warmup ${FULLCTX_WARMUP} --iters ${FULLCTX_ITERS} --device ${BENCH_DEVICE}"
    done
  done
}

case "${MODE}" in
  build)
    build_ext
    ;;
  attn)
    build_ext
    bench_attn
    ;;
  direct)
    build_ext
    bench_direct
    ;;
  fullctx)
    build_ext
    bench_fullctx
    ;;
  moe)
    build_ext
    bench_moe
    ;;
  all)
    build_ext
    bench_attn
    bench_direct
    bench_fullctx
    bench_moe
    ;;
  shell)
    if [[ "${RUNNER}" == "exec" ]]; then
      ${COMPOSE} exec "${SERVICE}" bash
    else
      bash
    fi
    ;;
  *)
    echo "ERROR: unknown mode '${MODE}'" >&2
    usage >&2
    exit 2
    ;;
esac
