"""Regression-gate measurement runner.

Called by ``scripts/regression_gate.sh``. Streams a small, deterministic matrix
of OpenAI-style chat-completion requests against a running vLLM service and
checks the measured numbers against floor/cap thresholds.

Design choices:

  * We hit the running service exactly the way ``scripts/bench_*`` already do,
    so the numbers are directly comparable to the existing AGENTS.md notes.
  * We use unique random text from token 0 for the long-prompt cases so prefix
    caching cannot hide TTFT.
  * c4 (4 concurrent) is launched with a thread pool. We measure aggregate
    request-level throughput and aggregate steady decode (excluding the first
    output token to back out queue/launch jitter).
  * Output is human-readable + a single JSONL summary line so an outer CI
    script can grep ``"summary":`` and archive a flat record.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests


def _gen_unique_prompt(num_words: int, seed: int) -> str:
    rng = random.Random(seed)
    words = []
    for _ in range(num_words):
        n = rng.randint(3, 10)
        words.append("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=n)))
    return " ".join(words)


@dataclass
class StreamResult:
    request_tok_s: float
    steady_tok_s: float
    ttft_s: float
    completion_tokens: int
    total_wall_s: float
    error: Optional[str] = None


def _stream_one(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float = 600.0,
) -> StreamResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first_token_at = None
    last_token_at = None
    completion_tokens = 0

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            stream=True,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]
            if line == "[DONE]":
                break
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = evt.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    completion_tokens += 1
            usage = evt.get("usage")
            if usage and isinstance(usage, dict):
                if usage.get("completion_tokens"):
                    completion_tokens = max(
                        completion_tokens, int(usage["completion_tokens"])
                    )
    except Exception as exc:
        return StreamResult(0.0, 0.0, 0.0, 0, time.perf_counter() - t0, str(exc))

    total_wall = time.perf_counter() - t0
    if completion_tokens <= 0 or first_token_at is None:
        return StreamResult(0.0, 0.0, 0.0, completion_tokens, total_wall, "no tokens")

    ttft = first_token_at - t0
    request_tok_s = completion_tokens / max(total_wall, 1e-6)
    steady_window = (last_token_at - first_token_at) if last_token_at else 0.0
    steady_tok_s = (
        (completion_tokens - 1) / steady_window if steady_window > 0 else 0.0
    )
    return StreamResult(
        request_tok_s=request_tok_s,
        steady_tok_s=steady_tok_s,
        ttft_s=ttft,
        completion_tokens=completion_tokens,
        total_wall_s=total_wall,
    )


def _stream_many(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
) -> list[StreamResult]:
    results: list[StreamResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(_stream_one, base_url, model, p, max_tokens) for p in prompts
        ]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--c1-floor", type=float, required=True)
    p.add_argument("--c4-floor", type=float, required=True)
    p.add_argument("--ttft4k-cap", type=float, required=True)
    p.add_argument("--ttft8k-cap", type=float, required=True)
    p.add_argument("--ttft16k-cap", type=float, required=True)
    p.add_argument("--label", default="phase0")
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()

    short_prompt = "Reply with a brief poem about the moon."

    print("--- c1 short prompt 256 tokens (warmup + measure) ---")
    _ = _stream_one(args.base_url, args.model, short_prompt, max_tokens=64)
    c1 = _stream_one(args.base_url, args.model, short_prompt, max_tokens=256)
    print(
        f"c1: request {c1.request_tok_s:.2f} tok/s, steady {c1.steady_tok_s:.2f} tok/s, "
        f"TTFT {c1.ttft_s:.3f}s"
    )

    print("--- c4 short prompt 256 tokens (warm) ---")
    c4_results = _stream_many(
        args.base_url, args.model, [short_prompt] * 4, max_tokens=256, concurrency=4
    )
    c4_request_sum = sum(r.request_tok_s for r in c4_results)
    c4_steady_sum = sum(r.steady_tok_s for r in c4_results)
    print(
        f"c4 aggregate: request {c4_request_sum:.2f} tok/s, steady {c4_steady_sum:.2f} tok/s"
    )

    print("--- unique 4k prompt / 16 output ---")
    p4 = _gen_unique_prompt(num_words=4100, seed=4001)
    r4 = _stream_one(args.base_url, args.model, p4, max_tokens=16)
    print(f"4k: TTFT {r4.ttft_s:.3f}s, steady {r4.steady_tok_s:.2f} tok/s")

    print("--- unique 8k prompt / 16 output ---")
    p8 = _gen_unique_prompt(num_words=8200, seed=8001)
    r8 = _stream_one(args.base_url, args.model, p8, max_tokens=16)
    print(f"8k: TTFT {r8.ttft_s:.3f}s, steady {r8.steady_tok_s:.2f} tok/s")

    print("--- unique 16k prompt / 16 output ---")
    p16 = _gen_unique_prompt(num_words=16400, seed=16001)
    r16 = _stream_one(args.base_url, args.model, p16, max_tokens=16)
    print(f"16k: TTFT {r16.ttft_s:.3f}s, steady {r16.steady_tok_s:.2f} tok/s")

    summary = {
        "label": args.label,
        "c1_request_tok_s": round(c1.request_tok_s, 3),
        "c1_steady_tok_s": round(c1.steady_tok_s, 3),
        "c1_ttft_s": round(c1.ttft_s, 4),
        "c4_aggregate_request_tok_s": round(c4_request_sum, 3),
        "c4_aggregate_steady_tok_s": round(c4_steady_sum, 3),
        "ttft_4k_s": round(r4.ttft_s, 4),
        "ttft_8k_s": round(r8.ttft_s, 4),
        "ttft_16k_s": round(r16.ttft_s, 4),
        "decode_4k_steady_tok_s": round(r4.steady_tok_s, 3),
        "decode_8k_steady_tok_s": round(r8.steady_tok_s, 3),
        "decode_16k_steady_tok_s": round(r16.steady_tok_s, 3),
        "thresholds": {
            "c1_floor": args.c1_floor,
            "c4_floor": args.c4_floor,
            "ttft4k_cap": args.ttft4k_cap,
            "ttft8k_cap": args.ttft8k_cap,
            "ttft16k_cap": args.ttft16k_cap,
        },
    }
    print("summary:", json.dumps(summary, separators=(",", ":")))

    if args.baseline:
        out = os.path.join("profiles", f"baseline_{args.label}.txt")
        os.makedirs("profiles", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2) + "\n")
        print(f"baseline written to {out}")

    failures: list[str] = []
    if c1.steady_tok_s < args.c1_floor:
        failures.append(
            f"c1_steady={c1.steady_tok_s:.2f} < floor {args.c1_floor:.2f}"
        )
    if c4_steady_sum < args.c4_floor:
        failures.append(
            f"c4_steady_sum={c4_steady_sum:.2f} < floor {args.c4_floor:.2f}"
        )
    if r4.ttft_s > args.ttft4k_cap:
        failures.append(f"ttft_4k={r4.ttft_s:.3f}s > cap {args.ttft4k_cap:.3f}s")
    if r8.ttft_s > args.ttft8k_cap:
        failures.append(f"ttft_8k={r8.ttft_s:.3f}s > cap {args.ttft8k_cap:.3f}s")
    if r16.ttft_s > args.ttft16k_cap:
        failures.append(
            f"ttft_16k={r16.ttft_s:.3f}s > cap {args.ttft16k_cap:.3f}s"
        )

    if failures:
        print("REGRESSION GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("REGRESSION GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
