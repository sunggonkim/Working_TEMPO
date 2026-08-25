#!/usr/bin/env python3
"""Elastic-PD router with phase-correct admission-credit lifetimes.

Local-compute credit represents prefill occupancy and remote-KV credit
represents remote handoff occupancy.  Neither credit is decode occupancy.
Consequently a TEMPO reservation is released when the first streamed response
chunk arrives, while the HTTP stream remains active until normal completion.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import hashlib
import json
import logging
import math
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from eval.sota_4node import tempo_pd_elastic_router_v444 as v444
from eval.sota_4node import tempo_pd_elastic_router_v445 as prior
from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_elastic_controller_v443 import ElasticPhase, ElasticRoute
from tempo.pd_elastic_profile_v444 import load_elastic_profile


ROUTER_SCHEMA = "tempo-elastic-pd-router-448"
LOGGER = logging.getLogger(__name__)
NO_DEADLINE_SENTINEL_MS = prior.NO_DEADLINE_SENTINEL_MS


class ElasticPDRouterCore(prior.ElasticPDRouterCore):
    """Separate admission-credit lifetime from response-stream lifetime."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._admission_released_ns: dict[str, int] = {}

    def mark_first_response_chunk(self, request_id: str) -> None:
        now_ns = time.perf_counter_ns()
        with self._lock:
            record = self._get(request_id)
            already_released = request_id in self._admission_released_ns
        if already_released:
            raise ValueError("first response chunk recorded twice")
        if request_id in self._elastic_owned:
            kwargs: dict[str, bool] = {}
            if record.remote_probe:
                elapsed_ms = (now_ns - record.decided_ns) / 1_000_000
                kwargs["remote_probe_success"] = (
                    record.remote_score_ms is not None
                    and elapsed_ms <= record.remote_score_ms
                )
            self.elastic.complete(request_id, **kwargs)
        with self._lock:
            self._admission_released_ns[request_id] = now_ns
        self._replace(
            request_id,
            phase="first_response_credit_released",
            response_started_ns=now_ns,
        )

    def complete(self, request_id: str) -> None:
        finished_ns = time.perf_counter_ns()
        with self._lock:
            record = self._get(request_id)
            released = request_id in self._admission_released_ns
        if request_id in self._elastic_owned and not released:
            kwargs: dict[str, bool] = {}
            if record.remote_probe:
                elapsed_ms = (finished_ns - record.decided_ns) / 1_000_000
                kwargs["remote_probe_success"] = (
                    record.remote_score_ms is not None
                    and elapsed_ms <= record.remote_score_ms
                )
            self.elastic.complete(request_id, **kwargs)
            with self._lock:
                self._admission_released_ns[request_id] = finished_ns
        self._replace(
            request_id, phase=ElasticPhase.COMPLETE.value, finished_ns=finished_ns
        )

    def fail(self, request_id: str, error: str) -> None:
        base._require(bool(error), "error must be nonempty")
        with self._lock:
            record = self._get(request_id)
            released = request_id in self._admission_released_ns
        if (
            request_id in self._elastic_owned
            and not released
            and record.phase not in {
                ElasticPhase.COMPLETE.value,
                ElasticPhase.FAILED.value,
            }
        ):
            self.elastic.fail(request_id)
        self._replace(
            request_id,
            phase=ElasticPhase.FAILED.value,
            finished_ns=time.perf_counter_ns(),
            error=error,
        )

    def records(self) -> list[dict[str, Any]]:
        rows = super().records()
        with self._lock:
            released = dict(self._admission_released_ns)
            elastic_owned = set(self._elastic_owned)
        for row in rows:
            request_id = row["request_id"]
            row["admission_credit_scope"] = (
                "prefill_or_remote_handoff" if request_id in elastic_owned else None
            )
            row["admission_credit_release_event"] = (
                "first_response_chunk" if request_id in released else None
            )
            row["admission_credit_released_ns"] = released.get(request_id)
        return rows


def _headers(record: v444.ElasticRouterRecord) -> dict[str, str]:
    headers = v444._headers(record)
    headers["X-Tempo-PD-Schema"] = ROUTER_SCHEMA
    return headers


def build_app(
    config: base.RouterConfig,
    profile,
    *,
    allow_screen_profile: bool = False,
    queue_wait_ms: float = 100.0,
) -> FastAPI:
    base._require(
        queue_wait_ms >= 0 and math.isfinite(queue_wait_ms),
        "queue_wait_ms must be finite and nonnegative",
    )
    core = ElasticPDRouterCore(
        config, profile, allow_screen_profile=allow_screen_profile
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.local = httpx.AsyncClient(base_url=config.local_url, timeout=None)
        app.state.remote = httpx.AsyncClient(base_url=config.remote_url, timeout=None)
        app.state.tokenizer = httpx.AsyncClient(
            base_url=config.tokenizer_url, timeout=None
        )
        app.state.vllm_metrics = httpx.AsyncClient(
            base_url=config.local_url,
            timeout=httpx.Timeout(5.0),
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        try:
            yield
        finally:
            await app.state.local.aclose()
            await app.state.remote.aclose()
            await app.state.tokenizer.aclose()
            await app.state.vllm_metrics.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.tempo_core = core

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "schema": ROUTER_SCHEMA,
            "ok": True,
            "profile_id": profile.profile_id,
            "profile_fingerprint_sha256": profile.fingerprint_sha256,
            "admission_credit_release_event": "first_response_chunk",
        }

    @app.get("/tempo/decisions")
    async def decisions() -> dict[str, Any]:
        rows = core.records()
        return {"schema": v444.ROUTER_SCHEMA, "count": len(rows), "decisions": rows,
                "runtime_schema": ROUTER_SCHEMA}

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_id = request.headers.get(base.REQUEST_ID_HEADER)
        if not request_id:
            raise HTTPException(
                status_code=400, detail=f"missing {base.REQUEST_ID_HEADER}"
            )
        upstream = None
        try:
            payload = await request.json()
            base._require(isinstance(payload, dict), "request body must be an object")
            base._require(
                payload.get("model") == config.served_model_name,
                "served model mismatch",
            )
            output_tokens = payload.get("max_tokens")
            base._require(
                type(output_tokens) is int and output_tokens >= 2,
                "max_tokens must be at least two",
            )
            prompt = payload.get("prompt")
            if isinstance(prompt, list):
                base._require(
                    bool(prompt) and all(type(value) is int for value in prompt),
                    "token prompt must contain ints",
                )
                prompt_tokens = len(prompt)
                token_ids = list(prompt)
            else:
                base._require(
                    isinstance(prompt, str) and prompt, "prompt must be nonempty"
                )
                tokenized = await app.state.tokenizer.post(
                    "/tokenize", json={"prompt": prompt}
                )
                tokenized.raise_for_status()
                tokens = tokenized.json().get("tokens")
                base._require(
                    isinstance(tokens, list) and tokens,
                    "tokenizer returned no tokens",
                )
                prompt_tokens = len(tokens)
                token_ids = tokens

            deadline_header = request.headers.get(
                "x-tempo-remaining-deadline-ms"
            )
            deadline = (
                float(deadline_header) if deadline_header is not None else None
            )
            prompt_key = hashlib.sha256(json.dumps(
                token_ids, separators=(",", ":")
            ).encode()).hexdigest()
            prepare_namespace = getattr(core, "prepare_prompt_namespace", None)
            if prepare_namespace is not None:
                prepare_namespace(request_id, prompt_key)
            prepare_tokens = getattr(core, "prepare_prompt_tokens", None)
            if prepare_tokens is not None:
                prepare_tokens(request_id, token_ids)
            prepare_semantic_load = getattr(
                core, "prepare_frontend_semantic_load", None
            )
            if prepare_semantic_load is not None:
                prepare_semantic_load(
                    request_id=request_id,
                    pair_index=request.headers.get(
                        "x-tempo-pd-frontend-pair-index"),
                    decode_tokens_before=request.headers.get(
                        "x-tempo-pd-frontend-decode-tokens-before"),
                    active_requests_before=request.headers.get(
                        "x-tempo-pd-frontend-active-requests-before"),
                    max_num_seqs=request.headers.get(
                        "x-tempo-pd-frontend-max-num-seqs"),
                )
            prepare_vllm_load = getattr(
                core, "prepare_vllm_load_snapshot", None
            )
            if prepare_vllm_load is not None:
                await prepare_vllm_load(request_id, app.state.vllm_metrics)
            record = core.decide(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_deadline_ms=deadline,
            )
            service_lane_reservation = getattr(
                core, "service_lane_reservation", lambda _request_id: None
            )(request_id)
            if (
                isinstance(service_lane_reservation, dict)
                and service_lane_reservation.get("status") == "unavailable"
            ):
                core.fail(
                    request_id,
                    "endpoint service-lane reservation unavailable",
                )
                detail = {
                    "code": "tempo_go_service_lane_reservation_unavailable",
                    "reason": service_lane_reservation.get("reason"),
                    "request_id": request_id,
                }
                return JSONResponse(
                    status_code=503,
                    content=detail,
                    headers={
                        "X-Tempo-Service-Lane-Reservation": "unavailable",
                        "X-Tempo-Service-Lane-Reason": str(
                            service_lane_reservation.get("reason")
                        ),
                        "X-Tempo-PD-Request-Id": request_id,
                    },
                )
            queue_started_ns = time.perf_counter_ns()
            queue_wait_for_request_ms = queue_wait_ms
            endpoint_queue_timeout_reason = (
                "endpoint_bounded_queue_lease_timeout"
                if isinstance(service_lane_reservation, dict)
                and service_lane_reservation.get("queue_lease", False)
                else "endpoint_bounded_global_route_timeout"
            )
            global_queue_wait = getattr(core, "global_queue_wait_ms", None)
            if global_queue_wait is not None:
                queue_wait_for_request_ms = global_queue_wait(
                    request_id,
                    default_queue_wait_ms=queue_wait_ms,
                    remaining_deadline_ms=deadline,
                )
            while record.route is ElasticRoute.QUEUE:
                elapsed_ms = (
                    time.perf_counter_ns() - queue_started_ns
                ) / 1_000_000
                if elapsed_ms >= queue_wait_for_request_ms:
                    core.fail(request_id, "bounded ingress queue timeout")
                    raise HTTPException(
                        status_code=503,
                        detail="elastic ingress queue timeout",
                        headers={
                            "X-Tempo-Service-Lane-Reservation": "timeout",
                            "X-Tempo-Service-Lane-Reason": (
                                endpoint_queue_timeout_reason
                            ),
                            "X-Tempo-PD-Request-Id": request_id,
                        },
                    )
                await asyncio.sleep(
                    min(0.001, (queue_wait_for_request_ms - elapsed_ms) / 1000)
                )
                remaining = (
                    deadline - elapsed_ms
                    if deadline is not None
                    else NO_DEADLINE_SENTINEL_MS
                )
                if remaining <= 0:
                    core.fail(
                        request_id, "request deadline expired in ingress queue"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="deadline expired in ingress queue",
                        headers={
                            "X-Tempo-Service-Lane-Reservation": "timeout",
                            "X-Tempo-Service-Lane-Reason": (
                                "endpoint_ingress_queue_deadline_expired"
                            ),
                            "X-Tempo-PD-Request-Id": request_id,
                        },
                    )
                record = core.retry(request_id, remaining)

            prepare_upstream_payload = getattr(
                core, "prepare_upstream_payload", None
            )
            if prepare_upstream_payload is not None:
                payload = prepare_upstream_payload(record, payload)
                base._require(
                    isinstance(payload, dict),
                    "prepared upstream payload must be an object",
                )
            client = (
                app.state.remote
                if record.route is ElasticRoute.REMOTE
                else app.state.local
            )
            forwarded_headers = {"Content-Type": "application/json"}
            for source, target in (
                ("authorization", "Authorization"),
                ("x-tempo-request-id", "X-Request-Id"),
                ("session-id", "session-id"),
            ):
                value = request.headers.get(source)
                if value:
                    forwarded_headers[target] = value
            prepare_upstream_headers = getattr(
                core, "prepare_upstream_headers", None)
            if prepare_upstream_headers is not None:
                forwarded_headers = prepare_upstream_headers(
                    record, forwarded_headers)
                base._require(
                    isinstance(forwarded_headers, dict)
                    and all(
                        isinstance(key, str) and isinstance(value, str)
                        for key, value in forwarded_headers.items()
                    ),
                    "prepared upstream headers must be string pairs",
                )
            upstream_request = client.build_request(
                "POST", "/v1/completions", json=payload, headers=forwarded_headers
            )
            core.mark_upstream_started(request_id)
            upstream = await client.send(upstream_request, stream=True)
            upstream.raise_for_status()
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(
                "request failed before response stream: request_id=%s",
                request_id,
            )
            if request_id and any(
                row["request_id"] == request_id for row in core.records()
            ):
                core.fail(request_id, f"{type(exc).__name__}: {exc}")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        async def generate():
            first_chunk = True
            try:
                assert upstream is not None
                async for chunk in upstream.aiter_raw():
                    observe_chunk = getattr(
                        core, "observe_backend_stream_chunk", None)
                    if observe_chunk is not None:
                        observe_chunk(
                            request_id,
                            route=record.route.value,
                            chunk=chunk,
                        )
                    if first_chunk:
                        core.mark_first_response_chunk(request_id)
                        first_chunk = False
                    yield chunk
                observe_backend = getattr(core, "observe_backend_completion", None)
                if observe_backend is not None:
                    observe_backend(
                        request_id, route=record.route.value,
                        upstream_headers=upstream.headers,
                    )
                core.complete(request_id)
            except BaseException as exc:
                core.fail(request_id, f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if upstream is not None:
                    await upstream.aclose()

        response_headers = _headers(record)
        if service_lane_reservation is not None:
            response_headers.update({
                "X-Tempo-Service-Lane-Reservation": str(
                    service_lane_reservation.get("status")),
                "X-Tempo-Service-Lane-Reason": str(
                    service_lane_reservation.get("reason")),
            })
        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=response_headers
        )

    return app


def main(argv=None) -> int:
    args = prior.prior._parse(argv)
    profile = load_elastic_profile(args.profile.resolve())
    config = base.RouterConfig(
        mode=base.RouterMode.TEMPO_AUTO,
        local_url=args.local_url,
        remote_url=args.remote_url,
        tokenizer_url=args.tokenizer_url,
        served_model_name=args.served_model_name,
        model_id=args.model_id,
        model_revision=args.model_revision,
        topology_id=args.topology_id,
        remote_backend=args.remote_backend,
        classifier_version=args.classifier_version,
        decoder_load_bucket=args.decoder_load_bucket,
        kv_bytes_per_token=args.kv_bytes_per_token,
    )
    import uvicorn

    uvicorn.run(
        build_app(
            config,
            profile,
            allow_screen_profile=args.allow_screen_profile,
            queue_wait_ms=args.queue_wait_ms,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
