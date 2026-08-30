"""Qwen7B loaded crossover with three synchronized decoder streams."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any, Callable

from eval.sota_4node import live_pd_controller_lmcache_v5 as token_stream
from eval.sota_4node import live_pd_controller_lmcache_v8_loaded as loaded
from eval.sota_4node import live_pd_controller_lmcache_v10_streamsync as streamsync
from eval.sota_4node import live_pd_controller_lmcache_v17_qwen7b_loaded_short as short
from eval.sota_4node import live_pd_controller_v1 as base


BACKGROUND_STREAMS = 3


def _with_background(
    decoder_urls_csv: str,
    pair: int,
    bucket: int,
    tag: str,
    foreground: Callable[[], Any],
) -> dict[str, Any]:
    decoder_urls = decoder_urls_csv.split(",")
    base._require(len(decoder_urls) == 2, "two decoder URLs are required")
    ready = [threading.Event() for _ in range(BACKGROUND_STREAMS)]
    holders: list[dict[str, Any]] = [{} for _ in range(BACKGROUND_STREAMS)]

    def run_background(index: int) -> None:
        holder = holders[index]
        started_ns = time.perf_counter_ns()
        body = {
            "model": base.SERVED_MODEL,
            "prompt": base._prompt(f"decoder-background-{index}", bucket, 16),
            "max_tokens": loaded.BACKGROUND_TOKENS,
            "min_tokens": loaded.BACKGROUND_TOKENS,
            "ignore_eos": True,
            "temperature": 0.0,
            "seed": 20260814 + index,
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
                "X-Request-Id": f"background-{tag}-{pair}-{index}",
            },
            method="POST",
        )
        token_count = 0
        usage: dict[str, Any] | None = None
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
                    if count and "first_token_ns" not in holder:
                        holder["first_token_ns"] = time.perf_counter_ns()
                        ready[index].set()
                    token_count += count
            holder["usage"] = usage
            holder["token_count"] = token_count
        except BaseException as exc:
            holder["error"] = exc
            ready[index].set()
        finally:
            holder["e2e_ms"] = (time.perf_counter_ns() - started_ns) / 1_000_000.0

    workers = [
        threading.Thread(target=run_background, args=(i,), name=f"live-pd-heavy-{tag}-{i}")
        for i in range(BACKGROUND_STREAMS)
    ]
    for worker in workers:
        worker.start()
    for event in ready:
        base._require(event.wait(timeout=30.0), "background first token timeout")
    for holder, worker in zip(holders, workers):
        if "error" in holder:
            for pending in workers:
                pending.join()
            raise holder["error"]
        base._require(worker.is_alive(), "background completed before foreground start")

    foreground_start_ns = time.perf_counter_ns()
    result = foreground()
    if callable(result):
        result = result()
    base._require(isinstance(result, dict), "foreground callback must return an object")
    for worker in workers:
        worker.join(timeout=base.REQUEST_TIMEOUT_S)
        base._require(not worker.is_alive(), "background request did not finish")
    streams: list[dict[str, Any]] = []
    for index, holder in enumerate(holders):
        if "error" in holder:
            raise holder["error"]
        usage = holder.get("usage")
        base._require(isinstance(usage, dict), "background stream omitted usage")
        base._require(
            holder.get("token_count") == loaded.BACKGROUND_TOKENS
            and int(usage.get("completion_tokens", -1)) == loaded.BACKGROUND_TOKENS,
            "background token count mismatch",
        )
        streams.append({
            "stream_index": index,
            "completion_tokens": loaded.BACKGROUND_TOKENS,
            "e2e_ms": holder["e2e_ms"],
            "first_token_to_foreground_start_ms": (
                foreground_start_ns - int(holder["first_token_ns"])
            ) / 1_000_000.0,
        })
    result["background_decode"] = {
        "pair_index": pair,
        "concurrent_streams": BACKGROUND_STREAMS,
        "completion_tokens": BACKGROUND_STREAMS * loaded.BACKGROUND_TOKENS,
        "completion_tokens_per_stream": loaded.BACKGROUND_TOKENS,
        "overlap_at_foreground_start": True,
        "synchronized_on_every_first_token": True,
        "streams": streams,
        "max_first_token_to_foreground_start_ms": max(
            item["first_token_to_foreground_start_ms"] for item in streams
        ),
    }
    return result


def main() -> int:
    old = streamsync._with_background
    streamsync._with_background = _with_background
    try:
        return short.main()
    finally:
        streamsync._with_background = old


if __name__ == "__main__":
    raise SystemExit(main())
