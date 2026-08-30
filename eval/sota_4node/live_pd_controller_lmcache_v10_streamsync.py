"""Loaded live-P/D experiment synchronized to the background's first token."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any, Callable

from eval.sota_4node import live_pd_controller_lmcache_v5 as token_stream
from eval.sota_4node import live_pd_controller_lmcache_v8_loaded as loaded
from eval.sota_4node import live_pd_controller_v1 as base


_ORIGINAL_WITH_BACKGROUND = loaded._with_background


def _with_background(
    decoder_urls_csv: str,
    pair: int,
    bucket: int,
    tag: str,
    foreground: Callable[[], Any],
) -> dict[str, Any]:
    decoder_urls = decoder_urls_csv.split(",")
    base._require(len(decoder_urls) == 2, "two decoder URLs are required")
    first_token = threading.Event()
    holder: dict[str, Any] = {}

    def run_background() -> None:
        started_ns = time.perf_counter_ns()
        body = {
            "model": base.SERVED_MODEL,
            "prompt": base._prompt("decoder-background", bucket, 16),
            "max_tokens": loaded.BACKGROUND_TOKENS,
            "min_tokens": loaded.BACKGROUND_TOKENS,
            "ignore_eos": True,
            "temperature": 0.0,
            "seed": 20260814,
            "stream": True,
            "stream_options": {"include_usage": True},
            "logprobs": 1,
        }
        request = urllib.request.Request(
            decoder_urls[pair].rstrip("/") + "/v1/completions",
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Connection": "close",
                "X-Request-Id": f"background-{tag}-{pair}",
            },
            method="POST",
        )
        usage: dict[str, Any] | None = None
        token_count = 0
        first_token_ns: int | None = None
        try:
            with urllib.request.urlopen(request, timeout=base.REQUEST_TIMEOUT_S) as response:
                base._require(response.status == 200, "background HTTP failure")
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
                    count = token_stream._choice_token_count(choices[0])
                    if count and first_token_ns is None:
                        first_token_ns = time.perf_counter_ns()
                        holder["first_token_ns"] = first_token_ns
                        first_token.set()
                    token_count += count
            holder["usage"] = usage
            holder["token_count"] = token_count
        except BaseException as exc:
            holder["error"] = exc
            first_token.set()
        finally:
            holder["e2e_ms"] = (time.perf_counter_ns() - started_ns) / 1_000_000.0

    worker = threading.Thread(target=run_background, name=f"live-pd-stream-bg-{tag}")
    worker.start()
    base._require(first_token.wait(timeout=30.0), "background first token timeout")
    if "error" in holder:
        worker.join()
        raise holder["error"]
    base._require(worker.is_alive(), "background completed before foreground start")
    foreground_start_ns = time.perf_counter_ns()
    result = foreground()
    if callable(result):
        result = result()
    base._require(isinstance(result, dict), "foreground callback must return an object")
    worker.join(timeout=base.REQUEST_TIMEOUT_S)
    base._require(not worker.is_alive(), "background request did not finish")
    if "error" in holder:
        raise holder["error"]
    usage = holder.get("usage")
    base._require(isinstance(usage, dict), "background stream omitted usage")
    base._require(
        holder.get("token_count") == loaded.BACKGROUND_TOKENS
        and int(usage.get("completion_tokens", -1)) == loaded.BACKGROUND_TOKENS,
        "background token count mismatch",
    )
    result["background_decode"] = {
        "pair_index": pair,
        "completion_tokens": loaded.BACKGROUND_TOKENS,
        "e2e_ms": holder["e2e_ms"],
        "overlap_at_foreground_start": True,
        "synchronized_on_first_token": True,
        "first_token_to_foreground_start_ms": (
            foreground_start_ns - int(holder["first_token_ns"])
        ) / 1_000_000.0,
    }
    return result


def main() -> int:
    old = loaded._with_background
    loaded._with_background = _with_background
    try:
        return loaded.main()
    finally:
        loaded._with_background = old


if __name__ == "__main__":
    raise SystemExit(main())
