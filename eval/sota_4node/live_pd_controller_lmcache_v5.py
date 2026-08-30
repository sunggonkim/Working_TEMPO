"""Token-accurate SSE adapter for the live LMCache P/D experiment.

vLLM may place more than one generated token in one completion stream event.
The earlier client counted SSE events, which rejected a valid 32-token warmup.
This revision counts ``choice.logprobs.tokens`` and records one arrival for
each generated token while retaining the exact usage and output checks.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.request
from typing import Any

from eval.sota_4node import live_pd_controller_lmcache_v2 as wire
from eval.sota_4node import live_pd_controller_lmcache_v4 as previous
from eval.sota_4node import live_pd_controller_v1 as base


def _choice_token_count(choice: dict[str, Any]) -> int:
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        tokens = logprobs.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
    return 1 if str(choice.get("text", "")) else 0


def _consume_stream(
    url: str,
    body: dict[str, Any],
    request_id: str,
    origin_ns: int,
    *,
    proxy: bool,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Connection": "close",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    arrivals_ns: list[int] = []
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    stream_event_count = 0
    with urllib.request.urlopen(request, timeout=base.REQUEST_TIMEOUT_S) as response:
        base._require(response.status == 200, f"HTTP status {response.status} from {url}")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            count = _choice_token_count(choice)
            if count:
                arrived_ns = time.perf_counter_ns()
                arrivals_ns.extend([arrived_ns] * count)
                stream_event_count += 1
                pieces.append(str(choice.get("text", "")))
    finished_ns = time.perf_counter_ns()
    base._require(len(arrivals_ns) == base.OUTPUT_TOKENS, (
        f"expected {base.OUTPUT_TOKENS} generated tokens, got {len(arrivals_ns)} "
        f"across {stream_event_count} SSE events"
    ))
    base._require(usage is not None, "stream response did not include usage")
    expected_usage = {base.OUTPUT_TOKENS}
    if proxy:
        expected_usage.add(base.OUTPUT_TOKENS - 1)
    base._require(
        int(usage.get("completion_tokens", -1)) in expected_usage,
        "completion token count mismatch",
    )
    gaps_ms = [
        (right - left) / 1_000_000.0
        for left, right in zip(arrivals_ns, arrivals_ns[1:])
    ]
    output = "".join(pieces)
    return {
        "http_status": 200,
        "prompt_tokens": int(usage.get("prompt_tokens", -1)),
        "completion_tokens": len(arrivals_ns),
        "output_sha256": base._sha256_text(output),
        "output_text": output,
        "ttft_ms": (arrivals_ns[0] - origin_ns) / 1_000_000.0,
        "e2e_ms": (finished_ns - origin_ns) / 1_000_000.0,
        "tpot_p50_ms": statistics.median(gaps_ms),
        "tpot_p99_ms": base._percentile(gaps_ms, 0.99),
        "tpot_max_ms": max(gaps_ms),
        "token_arrival_count": len(arrivals_ns),
        "sse_token_event_count": stream_event_count,
        "sse_events_may_batch_tokens": stream_event_count != len(arrivals_ns),
    }


def _stream_decode(
    url: str, body: dict[str, Any], request_id: str, origin_ns: int
) -> dict[str, Any]:
    return _consume_stream(url, body, request_id, origin_ns, proxy=False)


def _stream_proxy(
    url: str, prompt: str, request_id: str, origin_ns: int
) -> dict[str, Any]:
    return _consume_stream(
        url.rstrip("/") + "/v1/completions",
        base._base_decode_body(prompt),
        request_id,
        origin_ns,
        proxy=True,
    )


def main() -> int:
    old_decode = base._stream_decode
    old_proxy = wire._stream_proxy
    base._stream_decode = _stream_decode
    wire._stream_proxy = _stream_proxy
    try:
        return previous.main()
    finally:
        base._stream_decode = old_decode
        wire._stream_proxy = old_proxy


if __name__ == "__main__":
    raise SystemExit(main())
