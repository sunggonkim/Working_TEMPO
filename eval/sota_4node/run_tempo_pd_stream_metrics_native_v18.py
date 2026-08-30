#!/usr/bin/env python3
"""Fail-closed streaming metrics for native vLLM Nixl P/D responses.

Unlike the LMCache proxy, native Nixl returns one ordinary decoder stream.  It
therefore has no synthetic first-token SSE event: every requested token must be
proven by vLLM logprobs and the final usage count must match exactly.
"""

from __future__ import annotations

from typing import Any, BinaryIO, Callable

from eval.sota_4node import run_tempo_pd_stream_metrics_v1 as v1
from eval.sota_4node import run_tempo_pd_stream_metrics_v3 as v3


def _stream_record(
    stream: BinaryIO,
    *,
    dispatch_ns: int,
    run_start_ns: int,
    expected_tokens: int,
    route: str,
    clock_ns: Callable[[], int],
) -> dict[str, Any]:
    if route != "remote_prefill_live_kv":
        raise v1.base.ContractError(
            f"native Nixl metrics require remote_prefill_live_kv, got {route!r}"
        )
    # Native Nixl produces a normal, single decoder SSE stream.  Reuse the v3
    # parser's strict local-stream contract: exact usage, one response id, and
    # logprob identity/cardinality proof for every token.
    record = v3._stream_record(
        stream,
        dispatch_ns=dispatch_ns,
        run_start_ns=run_start_ns,
        expected_tokens=expected_tokens,
        route="native_nixl_single_decoder_stream",
        clock_ns=clock_ns,
    )
    record["native_nixl_stream_contract"] = (
        "single_decoder_exact_logprob_token_proof"
    )
    return record


def main() -> int:
    v1._stream_record = _stream_record
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
