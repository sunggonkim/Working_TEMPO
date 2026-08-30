#!/usr/bin/env python3
"""Forced-token metrics that close at SSE DONE and poll decision completion.

Some proxy responses keep the HTTP body open after a valid ``[DONE]`` event.
Reading to EOF can therefore hang even though vLLM is idle.  Leaving the
response context at DONE closes the upstream stream and lets the router's
``finally`` path publish completion.  Decision polling is short and bounded;
request execution is never retried.
"""

from __future__ import annotations

import time

from eval.sota_4node import run_tempo_pd_stream_metrics_forced_v32 as forced
from eval.sota_4node import run_tempo_pd_stream_metrics_v1 as v1
from eval.sota_4node import run_tempo_pd_stream_metrics_v3 as v3


_ORIGINAL_FETCH = v1._fetch_decisions


def _fetch_decisions(base_url: str, timeout_s: float):
    deadline = time.monotonic() + min(10.0, timeout_s)
    latest = None
    while True:
        latest = _ORIGINAL_FETCH(base_url, min(timeout_s, 10.0))
        rows = latest["decisions"]
        if rows and all(row.get("phase") == "complete" for row in rows):
            return latest
        if time.monotonic() >= deadline:
            return latest
        time.sleep(0.05)


def main() -> int:
    original_execute = v1.execute_request
    original_stream = v1._stream_record
    original_fetch = v1._fetch_decisions
    v1.execute_request = forced.execute_request
    v1._stream_record = v3._stream_record
    v1._fetch_decisions = _fetch_decisions
    try:
        return v1.main()
    finally:
        v1.execute_request = original_execute
        v1._stream_record = original_stream
        v1._fetch_decisions = original_fetch


if __name__ == "__main__":
    raise SystemExit(main())
