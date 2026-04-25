#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 8 16}"
MIN_SCALE_FACTOR="${MIN_SCALE_FACTOR:-0}"

if [[ -z "${BASE_URL}" ]]; then
  echo "ERROR: BASE_URL is required. Example: BASE_URL=http://127.0.0.1:8080 $0" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench="${script_dir}/bench_deepseek_v4_flash_api.sh"
tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

printf 'concurrency\taggregate_tok_s\tmean_request_tok_s\tmin_request_tok_s\tmax_request_tok_s\n' >"${tmp}"

for concurrency in ${CONCURRENCY_LIST}; do
  echo "=== concurrency=${concurrency} ==="
  output="$(
    BASE_URL="${BASE_URL}" \
    CONCURRENCY="${concurrency}" \
    UNIQUE_PROMPTS="${UNIQUE_PROMPTS:-1}" \
    START_BARRIER="${START_BARRIER:-1}" \
    "${bench}"
  )"
  echo "${output}"

  aggregate="$(awk -F': ' '/^aggregate_tok_s:/ {print $2}' <<<"${output}")"
  mean="$(awk -F': ' '/^mean_request_tok_s:/ {print $2}' <<<"${output}")"
  min_rate="$(awk -F': ' '/^min_request_tok_s:/ {print $2}' <<<"${output}")"
  max_rate="$(awk -F': ' '/^max_request_tok_s:/ {print $2}' <<<"${output}")"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${concurrency}" "${aggregate}" "${mean}" "${min_rate}" "${max_rate}" >>"${tmp}"
done

echo "=== summary ==="
awk -F '\t' '{
  printf "%-12s %-18s %-20s %-18s %-18s\n", $1, $2, $3, $4, $5
}' "${tmp}"

python3 - "${tmp}" "${MIN_SCALE_FACTOR}" <<'PY'
import csv
import sys

path = sys.argv[1]
min_scale = float(sys.argv[2])
rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
if not rows or min_scale <= 0:
    raise SystemExit(0)

base = float(rows[0]["aggregate_tok_s"])
best = max(float(row["aggregate_tok_s"]) for row in rows)
scale = best / max(base, 1e-9)
if scale < min_scale:
    raise SystemExit(
        f"ERROR: best aggregate scaling {scale:.2f}x is below required {min_scale:.2f}x"
    )
print(f"parallel_scale_factor: {scale:.2f}x")
PY
