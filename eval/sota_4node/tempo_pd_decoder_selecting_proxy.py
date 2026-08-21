#!/usr/bin/env python3
"""Strict decoder-selection wrapper around the official LMCache P/D proxy."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import Mapping


DECODER_INDEX_HEADER = "x-tempo-pd-decoder-index"


def requested_decoder_index(
    headers: Mapping[str, str], decoder_count: int, *, required: bool,
) -> int | None:
    if decoder_count <= 0:
        raise ValueError("decoder_count must be positive")
    raw = headers.get(DECODER_INDEX_HEADER)
    if raw is None:
        if required:
            raise ValueError(
                f"missing required {DECODER_INDEX_HEADER} header")
        return None
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DECODER_INDEX_HEADER} must be an integer") from exc
    if not 0 <= index < decoder_count:
        raise ValueError(
            f"{DECODER_INDEX_HEADER} is outside configured decoders")
    return index


def _load_official_proxy(path: Path):
    spec = importlib.util.spec_from_file_location(
        "_tempo_official_lmcache_disagg_proxy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official LMCache proxy")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    official_path = (
        repo_root
        / "third_party/lmcache/examples/disagg_prefill/disagg_proxy_server.py"
    )
    if not official_path.is_file():
        raise FileNotFoundError("official LMCache proxy is missing")
    proxy = _load_official_proxy(official_path)
    raw_required = os.environ.get(
        "TEMPO_PD_REQUIRE_DECODER_INDEX", "0")
    if raw_required not in ("0", "1"):
        raise ValueError(
            "TEMPO_PD_REQUIRE_DECODER_INDEX must be 0 or 1")
    required = raw_required == "1"
    original_pick = proxy.pick_up_clients

    def pick_up_clients(request):
        tokenization_client, prefill_client, decode_client = (
            original_pick(request))
        decoder_index = requested_decoder_index(
            request.headers, len(proxy.app.state.decode_clients),
            required=required,
        )
        if decoder_index is not None:
            decode_client = proxy.app.state.decode_clients[decoder_index]
            request.state.tempo_pd_decoder_index = decoder_index
        return tokenization_client, prefill_client, decode_client

    proxy.pick_up_clients = pick_up_clients

    @proxy.app.middleware("http")
    async def add_decoder_evidence(request, call_next):
        response = await call_next(request)
        decoder_index = getattr(
            request.state, "tempo_pd_decoder_index", None)
        if decoder_index is not None:
            response.headers[
                "X-Tempo-LMCache-PD-Decoder-Index"
            ] = str(decoder_index)
        return response

    proxy.global_args = proxy.parse_args()
    import uvicorn
    uvicorn.run(
        proxy.app, host=proxy.global_args.host,
        port=proxy.global_args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
