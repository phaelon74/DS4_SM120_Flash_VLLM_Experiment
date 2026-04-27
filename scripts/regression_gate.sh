#!/usr/bin/env bash
# scripts/regression_gate.sh
#
# Phase 0 acceptance gate for the SM120 100-tok/s push.
#
# Runs a deterministic short matrix against a *running* vLLM service (the one
# launched by `docker compose up -d vllm`) and asserts that:
#
#   * c1 256-token streaming steady decode    >= --c1-floor   (default 85.0)
#   * c4 256-token streaming aggregate steady >= --c4-floor   (default 175.0)
#   * unique 4k-token TTFT                    <= --ttft4k-cap (default 2.5 s)
#   * unique 8k-token TTFT                    <= --ttft8k-cap (default 6.0 s)
#   * unique 16k-token TTFT                   <= --ttft16k-cap(default 18.0 s)
#
# Exit code is 0 when every floor/cap holds; non-zero otherwise. Output goes to
# stdout in plain text plus a single JSONL summary line so it is easy to grep
# from CI logs and to archive into profiles/.
#
# This script intentionally does NOT modify the service. It only measures.
#
# Usage:
#   scripts/regression_gate.sh                  # default thresholds
#   scripts/regression_gate.sh --c1-floor 95    # tighten c1 after Phase 2
#   scripts/regression_gate.sh --baseline       # write profiles/baseline_phaseN.txt

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"

C1_FLOOR="${C1_FLOOR:-85.0}"
C4_FLOOR="${C4_FLOOR:-175.0}"
TTFT_4K_CAP="${TTFT_4K_CAP:-2.5}"
TTFT_8K_CAP="${TTFT_8K_CAP:-6.0}"
TTFT_16K_CAP="${TTFT_16K_CAP:-18.0}"

BASELINE=0
LABEL="phase0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --c1-floor)        C1_FLOOR="$2"; shift 2 ;;
        --c4-floor)        C4_FLOOR="$2"; shift 2 ;;
        --ttft4k-cap)      TTFT_4K_CAP="$2"; shift 2 ;;
        --ttft8k-cap)      TTFT_8K_CAP="$2"; shift 2 ;;
        --ttft16k-cap)     TTFT_16K_CAP="$2"; shift 2 ;;
        --baseline)        BASELINE=1; shift ;;
        --label)           LABEL="$2"; shift 2 ;;
        --base-url)        BASE_URL="$2"; shift 2 ;;
        --model)           MODEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "==> regression gate (label=${LABEL})"
echo "    base_url=${BASE_URL} model=${MODEL}"

if ! curl -fsS "${BASE_URL}/health" >/dev/null; then
    echo "ERROR: vLLM /health is not reachable at ${BASE_URL}" >&2
    exit 3
fi

python3 scripts/_regression_gate_runner.py \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --c1-floor "${C1_FLOOR}" \
    --c4-floor "${C4_FLOOR}" \
    --ttft4k-cap "${TTFT_4K_CAP}" \
    --ttft8k-cap "${TTFT_8K_CAP}" \
    --ttft16k-cap "${TTFT_16K_CAP}" \
    --label "${LABEL}" \
    $([[ "${BASELINE}" == "1" ]] && echo --baseline)
