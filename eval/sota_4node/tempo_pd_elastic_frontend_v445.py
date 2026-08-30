#!/usr/bin/env python3
"""Two-replica frontend preserving Elastic-PD provenance headers."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from eval.sota_4node.tempo_pd_frontend_v1 import pair_index


SCHEMA = "tempo-elastic-pd-two-replica-frontend-445"
PAIR_SCHEMA = "tempo-elastic-pd-router-444"
_FORWARDED = (
    "x-tempo-pd-schema", "x-tempo-pd-request-id", "x-tempo-pd-arm",
    "x-tempo-pd-route", "x-tempo-pd-reason", "x-tempo-pd-profile",
    "x-tempo-pd-profile-sha256",
)


def build_app(pair_urls: list[str]) -> FastAPI:
    if len(pair_urls) != 2 or len(set(pair_urls)) != 2:
        raise ValueError("exactly two unique pair router URLs are required")
    if any(not value.startswith(("http://", "https://")) for value in pair_urls):
        raise ValueError("pair router URLs must be HTTP(S)")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.clients = [httpx.AsyncClient(base_url=value, timeout=None)
                             for value in pair_urls]
        try:
            yield
        finally:
            for client in app.state.clients:
                await client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"schema": SCHEMA, "ok": True, "pairs": 2}

    @app.get("/tempo/decisions")
    async def decisions() -> dict[str, Any]:
        responses = await __import__("asyncio").gather(*[
            client.get("/tempo/decisions") for client in app.state.clients
        ])
        rows = []
        for response in responses:
            response.raise_for_status()
            value = response.json()
            if value.get("schema") != PAIR_SCHEMA:
                raise HTTPException(status_code=502, detail="pair decision schema mismatch")
            rows.extend(value.get("decisions", []))
        identifiers = [row.get("request_id") for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise HTTPException(status_code=502, detail="duplicate pair decision IDs")
        rows.sort(key=lambda row: str(row.get("request_id")))
        return {"schema": PAIR_SCHEMA, "count": len(rows), "decisions": rows,
                "frontend_schema": SCHEMA}

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_id = request.headers.get("x-tempo-request-id")
        if not request_id:
            raise HTTPException(status_code=400, detail="missing x-tempo-request-id")
        payload = await request.body()
        client = app.state.clients[pair_index(request_id, 2)]
        headers = {"Content-Type": request.headers.get("content-type", "application/json"),
                   "X-Tempo-Request-Id": request_id}
        for name in ("authorization", "x-tempo-remaining-deadline-ms"):
            value = request.headers.get(name)
            if value:
                headers[name] = value
        try:
            upstream_request = client.build_request(
                "POST", "/v1/completions", content=payload, headers=headers)
            upstream = await client.send(upstream_request, stream=True)
            upstream.raise_for_status()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        response_headers = {
            name: upstream.headers[name] for name in _FORWARDED
            if name in upstream.headers
        }

        async def generate():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers=response_headers)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pair-url", action="append", required=True)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(build_app(args.pair_url), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
