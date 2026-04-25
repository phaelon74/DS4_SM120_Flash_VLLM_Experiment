#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
PROMPT="${PROMPT:-Write a concise explanation of why tensor cores matter for LLM inference.}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MIN_TOKENS="${MIN_TOKENS:-${MAX_TOKENS}}"
IGNORE_EOS="${IGNORE_EOS:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
TIMEOUT_S="${TIMEOUT_S:-600}"
UNIQUE_PROMPTS="${UNIQUE_PROMPTS:-1}"
START_BARRIER="${START_BARRIER:-1}"

if [[ -z "${BASE_URL}" ]]; then
  echo "ERROR: BASE_URL is required. Example: BASE_URL=http://127.0.0.1:8080 $0" >&2
  exit 2
fi

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

wait_for_health() {
  local deadline=$((SECONDS + TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "ERROR: ${BASE_URL}/health did not become ready within ${TIMEOUT_S}s" >&2
  exit 1
}

wait_for_health

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

model_json="$(printf '%s' "${MODEL}" | json_escape)"
if [[ "${IGNORE_EOS}" == "1" || "${IGNORE_EOS}" == "true" || "${IGNORE_EOS}" == "TRUE" ]]; then
  ignore_eos_json="true"
else
  ignore_eos_json="false"
fi

make_payload() {
  local idx="$1"
  local prompt_text="${PROMPT}"
  if [[ "${UNIQUE_PROMPTS}" == "1" || "${UNIQUE_PROMPTS}" == "true" || "${UNIQUE_PROMPTS}" == "TRUE" ]]; then
    prompt_text="${PROMPT} Request id: ${idx}. Use a distinct first sentence."
  fi
  local prompt_json
  prompt_json="$(printf '%s' "${prompt_text}" | json_escape)"
  cat <<JSON
{
  "model": ${model_json},
  "messages": [{"role": "user", "content": ${prompt_json}}],
  "max_tokens": ${MAX_TOKENS},
  "min_tokens": ${MIN_TOKENS},
  "ignore_eos": ${ignore_eos_json},
  "temperature": 0
}
JSON
}

barrier="${tmpdir}/go"
for i in $(seq 1 "${CONCURRENCY}"); do
  (
    if [[ "${START_BARRIER}" == "1" || "${START_BARRIER}" == "true" || "${START_BARRIER}" == "TRUE" ]]; then
      while [[ ! -e "${barrier}" ]]; do
        sleep 0.01
      done
    fi
    start="$(date +%s%N)"
    payload="$(make_payload "${i}")"
    curl -fsS "${BASE_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "${payload}" >"${tmpdir}/${i}.json"
    end="$(date +%s%N)"
    printf '%s %s\n' "${start}" "${end}" >"${tmpdir}/${i}.time"
  ) &
done
sleep 0.2
start_all="$(date +%s%N)"
touch "${barrier}"
wait
end_all="$(date +%s%N)"

python3 - "${tmpdir}" "${start_all}" "${end_all}" "${CONCURRENCY}" <<'PY'
import json
import pathlib
import statistics
import sys

tmpdir = pathlib.Path(sys.argv[1])
start_all = int(sys.argv[2])
end_all = int(sys.argv[3])
concurrency = int(sys.argv[4])

tokens = []
request_rates = []
for path in sorted(tmpdir.glob("*.json")):
    data = json.loads(path.read_text())
    usage = data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    tokens.append(completion_tokens)
    t0, t1 = map(int, path.with_suffix(".time").read_text().split())
    elapsed = max((t1 - t0) / 1e9, 1e-9)
    request_rates.append(completion_tokens / elapsed)

elapsed_all = max((end_all - start_all) / 1e9, 1e-9)
total_tokens = sum(tokens)
print(f"requests: {concurrency}")
print(f"completion_tokens_total: {total_tokens}")
print(f"wall_seconds: {elapsed_all:.3f}")
print(f"aggregate_tok_s: {total_tokens / elapsed_all:.2f}")
if request_rates:
    print(f"mean_request_tok_s: {statistics.mean(request_rates):.2f}")
    print(f"min_request_tok_s: {min(request_rates):.2f}")
    print(f"max_request_tok_s: {max(request_rates):.2f}")
PY
