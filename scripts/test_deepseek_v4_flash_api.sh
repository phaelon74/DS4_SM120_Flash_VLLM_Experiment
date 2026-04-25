#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
DEFAULT_PROMPT="Reply with OK only."
PROMPT="${PROMPT:-${PROMT:-${DEFAULT_PROMPT}}}"
MAX_TOKENS="${MAX_TOKENS:-1}"
TIMEOUT_S="${TIMEOUT_S:-120}"
if [[ -z "${EXPECTED+x}" ]]; then
  if [[ "${PROMPT}" == "${DEFAULT_PROMPT}" && "${MAX_TOKENS}" == "1" ]]; then
    EXPECTED="OK"
  else
    EXPECTED=""
  fi
fi

usage() {
  cat >&2 <<'EOF'
Usage:
  BASE_URL=http://host:port ./scripts/test_deepseek_v4_flash_api.sh

Environment:
  BASE_URL     Required API base URL. Example: http://127.0.0.1:8001
  MODEL        Default: deepseek-ai/DeepSeek-V4-Flash
  PROMPT       Default: Reply with OK only.
  PROMT        Accepted as a typo alias for PROMPT.
  EXPECTED     Exact expected output. Defaults to OK only for the default 1-token smoke test.
               Set EXPECTED= to skip exact match.
  MAX_TOKENS   Default: 1
  TIMEOUT_S    Default: 120
EOF
}

if [[ -z "${BASE_URL}" ]]; then
  usage
  echo "ERROR: BASE_URL is required to avoid accidentally testing the wrong service." >&2
  exit 2
fi

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

wait_for_health() {
  local deadline=$((SECONDS + TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      echo "health: ok"
      return 0
    fi
    sleep 2
  done

  echo "ERROR: ${BASE_URL}/health did not become ready within ${TIMEOUT_S}s" >&2
  return 1
}

echo "Testing ${MODEL} at ${BASE_URL}"
wait_for_health

echo "models:"
curl -fsS "${BASE_URL}/v1/models" \
  | python3 -c 'import json,sys; data=json.load(sys.stdin); print("\n".join(m.get("id","<missing>") for m in data.get("data", [])))'

prompt_json="$(printf '%s' "${PROMPT}" | json_escape)"
model_json="$(printf '%s' "${MODEL}" | json_escape)"

payload="$(
  cat <<JSON
{
  "model": ${model_json},
  "messages": [{"role": "user", "content": ${prompt_json}}],
  "max_tokens": ${MAX_TOKENS},
  "temperature": 0
}
JSON
)"

echo "completion:"
response="$(
  curl -fsS "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "${payload}"
)"

content="$(
  printf '%s' "${response}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
)"

printf 'content: %q\n' "${content}"

if [[ -n "${EXPECTED}" && "${content}" != "${EXPECTED}" ]]; then
  echo "ERROR: expected ${EXPECTED@Q}, got ${content@Q}" >&2
  echo "${response}" >&2
  exit 1
fi

echo "api smoke test passed"
