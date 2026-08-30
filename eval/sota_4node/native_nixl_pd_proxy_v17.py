#!/usr/bin/env python3
"""Native Nixl P/D proxy with an internal request identity fallback."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Any
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


SCHEMA = "tempo-native-nixl-pd-proxy-17"


def build_app(prefill_url: str, decode_url: str, served_model: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.prefill = httpx.AsyncClient(base_url=prefill_url, timeout=None)
        app.state.decode = httpx.AsyncClient(base_url=decode_url, timeout=None)
        try:
            yield
        finally:
            await app.state.prefill.aclose()
            await app.state.decode.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"schema": SCHEMA, "ok": True}

    @app.post("/v1/completions")
    async def completions(request: Request):
        external_id = request.headers.get("x-tempo-request-id")
        internal_id = external_id or f"tempo-native-{uuid.uuid4().hex}"
        try:
            payload = await request.json()
            if not isinstance(payload, dict) or payload.get("model") != served_model:
                raise ValueError("served model mismatch")
            if payload.get("stream") is not True:
                raise ValueError("native proxy requires streaming decode")
            output_tokens = payload.get("max_tokens")
            if type(output_tokens) is not int or output_tokens < 2:
                raise ValueError("max_tokens must be at least two")
            prefill_body = dict(payload)
            prefill_body.update({
                "max_tokens": 1, "min_tokens": 1, "ignore_eos": True,
                "temperature": 0.0, "stream": False,
                "kv_transfer_params": {
                    "do_remote_decode": True, "do_remote_prefill": False,
                    "remote_engine_id": None, "remote_block_ids": None,
                    "remote_host": None, "remote_port": None,
                },
            })
            prefill_body.pop("stream_options", None)
            prefill_response = await app.state.prefill.post(
                "/v1/completions", json=prefill_body,
                headers={"X-Request-Id": internal_id + "-prefill"},
            )
            prefill_response.raise_for_status()
            transfer = prefill_response.json().get("kv_transfer_params")
            required = {
                "remote_engine_id", "remote_request_id", "remote_block_ids",
                "remote_host", "remote_port", "remote_num_tokens",
            }
            if not isinstance(transfer, dict) or not required.issubset(transfer):
                raise ValueError("native kv_transfer_params incomplete")
            decode_body = dict(payload)
            decode_body["kv_transfer_params"] = transfer
            upstream_request = app.state.decode.build_request(
                "POST", "/v1/completions", json=decode_body,
                headers={"X-Request-Id": internal_id + "-decode"},
            )
            upstream = await app.state.decode.send(upstream_request, stream=True)
            upstream.raise_for_status()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        async def generate():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            generate(), media_type="text/event-stream",
            headers={"X-Tempo-Native-Nixl-Schema": SCHEMA},
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--served-model", required=True)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(build_app(args.prefill_url, args.decode_url, args.served_model),
                host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
