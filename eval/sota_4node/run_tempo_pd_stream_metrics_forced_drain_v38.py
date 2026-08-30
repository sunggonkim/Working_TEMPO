#!/usr/bin/env python3
"""Forced-token metrics that drain HTTP EOF after the SSE DONE event."""

from __future__ import annotations

from eval.sota_4node import run_tempo_pd_stream_metrics_forced_v32 as forced
from eval.sota_4node import run_tempo_pd_stream_metrics_v1 as v1
from eval.sota_4node import run_tempo_pd_stream_metrics_v3 as v3


def _stream_record(stream, **kwargs):
    record = v3._stream_record(stream, **kwargs)
    # v3 stops at [DONE].  Consume the HTTP body to EOF so the router's
    # StreamingResponse generator can execute core.complete before the client
    # fetches decision provenance.
    stream.read()
    record["http_eof_drained_after_done"] = True
    return record


def main() -> int:
    v1.execute_request = forced.execute_request
    v1._stream_record = _stream_record
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
