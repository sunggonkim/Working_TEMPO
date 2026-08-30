#!/usr/bin/env python3
"""Forced streaming metrics with a deterministic per-request KV cache salt."""

from __future__ import annotations

import hashlib
import json
from urllib import request

from eval.sota_4node import run_tempo_pd_stream_metrics_forced_drain_v38 as drain
from eval.sota_4node import run_tempo_pd_stream_metrics_forced_v32 as forced
from eval.sota_4node import run_tempo_pd_stream_metrics_v1 as v1


def execute_request(item, *args, opener=request.urlopen, **kwargs):
    salt = "tempo-cold-" + hashlib.sha256(item.request_id.encode("utf-8")).hexdigest()

    def salted_opener(http_request, **call_kwargs):
        body = json.loads(http_request.data)
        body["cache_salt"] = salt
        rewritten = request.Request(
            http_request.full_url,
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers=dict(http_request.headers),
            method=http_request.get_method(),
        )
        return opener(rewritten, **call_kwargs)

    return forced.execute_request(item, *args, opener=salted_opener, **kwargs)


def main() -> int:
    v1.execute_request = execute_request
    v1._stream_record = drain._stream_record
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
