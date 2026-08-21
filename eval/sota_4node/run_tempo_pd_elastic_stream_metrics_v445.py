#!/usr/bin/env python3
"""Proven forced-token/EOF metrics adapted to Elastic-PD headers."""

from __future__ import annotations

from eval.sota_4node import run_tempo_pd_stream_metrics_forced_v32 as forced
from eval.sota_4node import run_tempo_pd_stream_metrics_v1 as v1
from eval.sota_4node import run_tempo_pd_stream_metrics_v3 as v3


ROUTER_SCHEMA = "tempo-elastic-pd-router-444"
LOCAL_ROUTE = "decoder_local_chunked_prefill"
REMOTE_ROUTE = "official_lmcache_remote_prefill"
_ORIGINAL_STREAM = v3._stream_record


def _router_headers(response, request_id):
    headers = response.headers
    result = {
        "schema": headers.get("X-Tempo-PD-Schema"),
        "request_id": headers.get("X-Tempo-PD-Request-Id"),
        "arm": headers.get("X-Tempo-PD-Arm"),
        "route": headers.get("X-Tempo-PD-Route"),
        "reason": headers.get("X-Tempo-PD-Reason"),
        "profile_id": headers.get("X-Tempo-PD-Profile"),
        "profile_fingerprint_sha256": headers.get("X-Tempo-PD-Profile-SHA256"),
    }
    v1._require(result["schema"] == ROUTER_SCHEMA, "elastic router schema mismatch")
    v1._require(result["request_id"] == request_id, "elastic request ID mismatch")
    v1._require(result["arm"] in {
        "always_local", "official_lmcache_remote", "predictor", "tempo"
    }, "elastic arm header mismatch")
    v1._require(result["route"] in {LOCAL_ROUTE, REMOTE_ROUTE},
                "elastic route header mismatch")
    for name in ("reason", "profile_id", "profile_fingerprint_sha256"):
        v1._require(isinstance(result[name], str) and result[name],
                    f"elastic {name} header missing")
    return result


def _stream_record(stream, **kwargs):
    route = kwargs.pop("route")
    legacy_route = "remote_prefill_live_kv" if route == REMOTE_ROUTE else (
        "decoder_local_recompute_or_cache")
    record = _ORIGINAL_STREAM(stream, route=legacy_route, **kwargs)
    # Drain HTTP EOF so StreamingResponse runs core.complete before provenance fetch.
    stream.read()
    record["http_eof_drained_after_done"] = True
    return record


def main() -> int:
    old_execute = v1.execute_request
    old_headers = v1._router_headers
    old_stream = v1._stream_record
    old_schema = v1.ROUTER_SCHEMA
    v1.execute_request = forced.execute_request
    v1._router_headers = _router_headers
    v1._stream_record = _stream_record
    v1.ROUTER_SCHEMA = ROUTER_SCHEMA
    try:
        return v1.main()
    finally:
        v1.execute_request = old_execute
        v1._router_headers = old_headers
        v1._stream_record = old_stream
        v1.ROUTER_SCHEMA = old_schema


if __name__ == "__main__":
    raise SystemExit(main())
