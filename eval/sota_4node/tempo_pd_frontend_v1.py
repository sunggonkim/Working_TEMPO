#!/usr/bin/env python3
"""Deterministic two-replica frontend for TEMPO-PD pair routers."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hashlib
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


SCHEMA = "tempo-pd-two-replica-frontend-1"
_SUFFIX = re.compile(r"(\d+)$")
_CONNECT_RETRIES = 2
_KEEPALIVE_EXPIRY_S = 1.0


def _pair_client(base_url: str) -> httpx.AsyncClient:
    # The measured blocks have a two-second cooldown while uvicorn's default
    # keep-alive timeout is five seconds.  Expiring idle client connections
    # well before that server timeout avoids a boundary race on a stale HTTP/1
    # socket.  HTTPX/httpcore's transport retries apply only while establishing
    # TCP (ConnectError/ConnectTimeout); they do not replay a POST after request
    # bytes may have reached the router, preserving exactly-once request IDs.
    transport = httpx.AsyncHTTPTransport(
        retries=_CONNECT_RETRIES,
        limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY_S),
    )
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=None,
        transport=transport,
    )


def pair_index(request_id: str, pair_count: int) -> int:
    if not request_id:
        raise ValueError("request_id must be nonempty")
    if type(pair_count) is not int or pair_count <= 0:
        raise ValueError("pair_count must be positive")
    match = _SUFFIX.search(request_id)
    if match:
        return int(match.group(1)) % pair_count
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % pair_count


def build_app(pair_urls: list[str]) -> FastAPI:
    if len(pair_urls) != 2 or len(set(pair_urls)) != 2:
        raise ValueError("exactly two unique pair router URLs are required")
    if any(not value.startswith(("http://", "https://")) for value in pair_urls):
        raise ValueError("pair router URLs must be HTTP(S)")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.clients = [_pair_client(value) for value in pair_urls]
        try:
            yield
        finally:
            for client in app.state.clients:
                await client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"schema": SCHEMA, "ok": True, "pairs": len(pair_urls)}

    @app.get("/tempo/decisions")
    async def decisions() -> dict[str, Any]:
        responses = await __import__("asyncio").gather(*[
            client.get("/tempo/decisions") for client in app.state.clients
        ])
        rows = []
        for response in responses:
            response.raise_for_status()
            value = response.json()
            if value.get("schema") != "tempo-live-pd-router-1":
                raise HTTPException(status_code=502, detail="pair decision schema mismatch")
            rows.extend(value.get("decisions", []))
        identifiers = [row.get("request_id") for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise HTTPException(status_code=502, detail="duplicate pair decision IDs")
        rows.sort(key=lambda row: str(row.get("request_id")))
        return {"schema": "tempo-live-pd-router-1", "count": len(rows),
                "decisions": rows, "frontend_schema": SCHEMA}

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_id = request.headers.get("x-tempo-request-id")
        if not request_id:
            raise HTTPException(status_code=400, detail="missing x-tempo-request-id")
        payload = await request.body()
        client = app.state.clients[pair_index(request_id, len(pair_urls))]
        headers = {"Content-Type": request.headers.get("content-type", "application/json"),
                   "X-Tempo-Request-Id": request_id}
        authorization = request.headers.get("authorization")
        if authorization:
            headers["Authorization"] = authorization
        try:
            upstream_request = client.build_request(
                "POST", "/v1/completions", content=payload, headers=headers
            )
            upstream = await client.send(upstream_request, stream=True)
            upstream.raise_for_status()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        response_headers = {
            name: upstream.headers[name]
            for name in (
                "x-tempo-pd-schema", "x-tempo-pd-request-id", "x-tempo-pd-mode",
                "x-tempo-pd-route", "x-tempo-pd-reason", "x-tempo-pd-workload",
                "x-tempo-pd-profile", "x-tempo-pd-manifest",
            )
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
