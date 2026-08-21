#!/usr/bin/env python3
"""Streaming metrics with a shared deterministic one-token logit bias."""

from __future__ import annotations

import json
from urllib import request

from eval.sota_4node import run_tempo_pd_stream_metrics_v1 as v1
from eval.sota_4node import run_tempo_pd_stream_metrics_v3 as v3


FORCED_TOKEN_ID = 362  # local Qwen2.5 tokenizer: encode(" A")
_ORIGINAL_EXECUTE = v1.execute_request


def execute_request(*args, opener=request.urlopen, **kwargs):
    def forced_opener(http_request, **call_kwargs):
        body = json.loads(http_request.data)
        body["logit_bias"] = {str(FORCED_TOKEN_ID): 100.0}
        rewritten = request.Request(
            http_request.full_url,
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers=dict(http_request.headers),
            method=http_request.get_method(),
        )
        return opener(rewritten, **call_kwargs)

    return _ORIGINAL_EXECUTE(*args, opener=forced_opener, **kwargs)


def main() -> int:
    v1.execute_request = execute_request
    return v3.main()


if __name__ == "__main__":
    raise SystemExit(main())
