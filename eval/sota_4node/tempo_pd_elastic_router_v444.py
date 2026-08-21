#!/usr/bin/env python3
"""Actual ingress router for the TEMPO Elastic-PD controller.

All experiment arms share tokenization, HTTP forwarding, and lifecycle code.
The full controller commits exactly once before either upstream starts.  A
credit miss remains in a bounded ingress queue; it is never hidden as a local
fallback.  REMOTE always targets the existing official LMCache P/D proxy.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import asyncio
import math
import threading
import time
from typing import Any, Callable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_admission import PDRoute
from tempo.pd_elastic_controller_v443 import (
    CacheResidency,
    ElasticDecision,
    ElasticEstimate,
    ElasticPDController,
    ElasticPhase,
    ElasticRequest,
    ElasticRoute,
)
from tempo.pd_elastic_profile_v444 import ElasticPDProfile, ElasticProfileRow, load_elastic_profile


ROUTER_SCHEMA = "tempo-elastic-pd-router-444"


class ElasticExperimentArm(str, Enum):
    ALWAYS_LOCAL = "always_local"
    OFFICIAL_LMCACHE_REMOTE = "official_lmcache_remote"
    PREDICTOR = "predictor"
    TEMPO = "tempo"


_ARM_MARKERS = {
    "local": ElasticExperimentArm.ALWAYS_LOCAL,
    "remote": ElasticExperimentArm.OFFICIAL_LMCACHE_REMOTE,
    "predictor": ElasticExperimentArm.PREDICTOR,
    "tempo": ElasticExperimentArm.TEMPO,
}


@dataclass(frozen=True)
class ElasticRouterRecord:
    request_id: str
    arm: ElasticExperimentArm
    route: ElasticRoute
    reason: str
    phase: str
    prompt_tokens: int
    output_tokens: int
    potential_kv_bytes: int
    decided_ns: int
    attempt: int
    cache_residency: CacheResidency
    profile_id: str
    profile_fingerprint_sha256: str
    regime: str | None
    median_gap_ns: int | None
    local_score_ms: float | None
    remote_score_ms: float | None
    local_compute_used_before_us: int | None
    remote_kv_used_before_bytes: int | None
    remote_probe: bool
    started_ns: int | None = None
    response_started_ns: int | None = None
    finished_ns: int | None = None
    error: str | None = None

    def public_dict(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["arm"] = self.arm.value
        value["route"] = self.route.value
        value["cache_residency"] = self.cache_residency.value
        return value


class ElasticPDRouterCore(base.TempoPDRouterCore):
    def __init__(
        self,
        config: base.RouterConfig,
        profile: ElasticPDProfile,
        *,
        cache_residency: Callable[[str], CacheResidency] | None = None,
        allow_screen_profile: bool = False,
    ) -> None:
        if not isinstance(config, base.RouterConfig):
            raise TypeError("config must be RouterConfig")
        if config.mode is not base.RouterMode.TEMPO_AUTO:
            raise ValueError("elastic router requires tempo_auto mode")
        if not isinstance(profile, ElasticPDProfile):
            raise TypeError("profile must be ElasticPDProfile")
        if profile.deployment_scope == "screen_only" and not allow_screen_profile:
            raise ValueError("screen_only profile requires explicit opt-in")
        profile.validate_identity(
            model_id=config.model_id,
            model_revision=config.model_revision,
            topology_id=config.topology_id,
            remote_backend=config.remote_backend,
            classifier_version=config.classifier_version,
            kv_bytes_per_token=config.kv_bytes_per_token,
        )
        self.config = config
        self.manifest = None
        self.policy = None
        self.ledger = None
        self.profile = profile
        self.elastic = ElasticPDController(profile.controller)
        self._cache_residency = cache_residency or (lambda _request_id: CacheResidency.MISS)
        self._records: dict[str, ElasticRouterRecord] = {}
        self._rows: dict[str, ElasticProfileRow] = {}
        self._elastic_owned: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def arm(request_id: str) -> ElasticExperimentArm:
        for marker, arm in _ARM_MARKERS.items():
            if request_id.startswith(f"epd-{marker}-"):
                return arm
        raise ValueError("request_id must begin with an explicit epd arm")

    @staticmethod
    def _remaining_deadline(value: float | None) -> float:
        if value is None:
            return math.inf
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("remaining_deadline_ms must be numeric")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError("remaining_deadline_ms must be finite and positive")
        return float(value)

    def _estimate(self, row: ElasticProfileRow, remaining_deadline_ms: float | None) -> ElasticEstimate:
        deadline = self._remaining_deadline(remaining_deadline_ms)
        return row.estimate(deadline)

    def _tempo_estimate(
        self, request_id: str, row: ElasticProfileRow,
        remaining_deadline_ms: float | None,
    ) -> ElasticEstimate:
        del request_id
        return self._estimate(row, remaining_deadline_ms)

    @staticmethod
    def _predictor_route(row: ElasticProfileRow, estimate: ElasticEstimate) -> tuple[ElasticRoute, str]:
        local_score = estimate.local_upper_bound_ms + estimate.uncertainty_ms
        remote_score = estimate.remote_upper_bound_ms + estimate.uncertainty_ms
        if (
            row.evidence_safe
            and remote_score <= local_score
            and remote_score <= estimate.remaining_deadline_ms
        ):
            return ElasticRoute.REMOTE, "predictor_remote_lower_bound"
        if estimate.local_tbt_safe and local_score <= estimate.remaining_deadline_ms:
            return ElasticRoute.LOCAL, "predictor_local_safe"
        return ElasticRoute.QUEUE, "predictor_no_deadline_safe_route"

    def decide(
        self, *, request_id: str, prompt_tokens: int, output_tokens: int,
        remaining_deadline_ms: float | None = None,
    ) -> ElasticRouterRecord:
        base._require(isinstance(request_id, str) and request_id.strip(),
                      "request_id must be nonempty")
        experiment_arm = self.arm(request_id)
        _, kv_bytes = self.classify(prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        row = self.profile.exact_row(prompt_tokens, output_tokens)
        if row is None:
            raise ValueError("no exact elastic profile row")
        if row.remote_kv_bytes != kv_bytes:
            raise ValueError("profile/router KV geometry mismatch")
        residency = self._cache_residency(request_id)
        if not isinstance(residency, CacheResidency):
            raise TypeError("cache residency resolver must return CacheResidency")
        estimate = self._estimate(row, remaining_deadline_ms)
        now_ns = time.perf_counter_ns()
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            base._require(len(self._records) < self.config.decision_capacity,
                          "decision capacity exhausted")

        elastic_decision: ElasticDecision | None = None
        if experiment_arm is ElasticExperimentArm.ALWAYS_LOCAL:
            route, reason = ElasticRoute.LOCAL, "fixed_always_local"
        elif experiment_arm is ElasticExperimentArm.OFFICIAL_LMCACHE_REMOTE:
            route, reason = ElasticRoute.REMOTE, "fixed_official_lmcache_remote"
        elif experiment_arm is ElasticExperimentArm.PREDICTOR:
            route, reason = self._predictor_route(row, estimate)
        else:
            estimate = self._tempo_estimate(
                request_id, row, remaining_deadline_ms)
            elastic_decision = self.elastic.submit(
                ElasticRequest(
                    request_id=request_id,
                    arrival_ns=now_ns,
                    cache_residency=residency,
                    local_compute_cost_us=row.local_compute_cost_us,
                    remote_kv_bytes=row.remote_kv_bytes,
                ),
                estimate,
            )
            route, reason = elastic_decision.route, elastic_decision.reason

        record = self._record(
            request_id=request_id,
            arm=experiment_arm,
            route=route,
            reason=reason,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            kv_bytes=kv_bytes,
            residency=residency,
            now_ns=now_ns,
            decision=elastic_decision,
        )
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            self._records[request_id] = record
            self._rows[request_id] = row
            if elastic_decision is not None:
                self._elastic_owned.add(request_id)
        return record

    def retry(self, request_id: str, remaining_deadline_ms: float) -> ElasticRouterRecord:
        with self._lock:
            current = self._get(request_id)
            row = self._rows[request_id]
        if current.arm is not ElasticExperimentArm.TEMPO:
            raise ValueError("only full TEMPO queued requests can retry")
        if current.route is not ElasticRoute.QUEUE:
            raise ValueError("only queued requests can retry")
        decision = self.elastic.retry(
            request_id, self._tempo_estimate(request_id, row, remaining_deadline_ms)
        )
        replacement = self._record(
            request_id=request_id, arm=current.arm, route=decision.route,
            reason=decision.reason, prompt_tokens=current.prompt_tokens,
            output_tokens=current.output_tokens, kv_bytes=current.potential_kv_bytes,
            residency=current.cache_residency, now_ns=current.decided_ns,
            decision=decision,
        )
        with self._lock:
            self._records[request_id] = replacement
        return replacement

    def _record(
        self, *, request_id: str, arm: ElasticExperimentArm, route: ElasticRoute,
        reason: str, prompt_tokens: int, output_tokens: int, kv_bytes: int,
        residency: CacheResidency, now_ns: int, decision: ElasticDecision | None,
    ) -> ElasticRouterRecord:
        return ElasticRouterRecord(
            request_id=request_id, arm=arm, route=route, reason=reason,
            phase=(decision.phase.value if decision else
                   (ElasticPhase.QUEUED.value if route is ElasticRoute.QUEUE else
                    "route_committed")),
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            potential_kv_bytes=kv_bytes, decided_ns=now_ns,
            attempt=decision.attempt if decision else 1,
            cache_residency=residency, profile_id=self.profile.profile_id,
            profile_fingerprint_sha256=self.profile.fingerprint_sha256,
            regime=decision.regime.value if decision else None,
            median_gap_ns=decision.median_gap_ns if decision else None,
            local_score_ms=decision.local_score_ms if decision else None,
            remote_score_ms=decision.remote_score_ms if decision else None,
            local_compute_used_before_us=(
                decision.local_compute_used_before_us if decision else None),
            remote_kv_used_before_bytes=(
                decision.remote_kv_used_before_bytes if decision else None),
            remote_probe=decision.remote_probe if decision else False,
        )

    def mark_upstream_started(self, request_id: str) -> None:
        with self._lock:
            record = self._get(request_id)
        if record.route is ElasticRoute.QUEUE:
            raise ValueError("queued request cannot start upstream")
        if request_id in self._elastic_owned:
            self.elastic.mark_started(request_id)
        self._replace(request_id, phase=ElasticPhase.STARTED.value,
                      started_ns=time.perf_counter_ns())

    def mark_response_started(self, request_id: str) -> None:
        self._replace(request_id, phase="response_started",
                      response_started_ns=time.perf_counter_ns())

    def complete(self, request_id: str) -> None:
        finished_ns = time.perf_counter_ns()
        with self._lock:
            record = self._get(request_id)
        if request_id in self._elastic_owned:
            kwargs: dict[str, bool] = {}
            if record.remote_probe:
                elapsed_ms = (finished_ns - record.decided_ns) / 1_000_000
                kwargs["remote_probe_success"] = (
                    record.remote_score_ms is not None
                    and elapsed_ms <= record.remote_score_ms
                )
            self.elastic.complete(request_id, **kwargs)
        self._replace(request_id, phase=ElasticPhase.COMPLETE.value, finished_ns=finished_ns)

    def fail(self, request_id: str, error: str) -> None:
        base._require(bool(error), "error must be nonempty")
        with self._lock:
            record = self._get(request_id)
        if request_id in self._elastic_owned and record.phase not in {
            ElasticPhase.COMPLETE.value, ElasticPhase.FAILED.value,
        }:
            self.elastic.fail(request_id)
        self._replace(request_id, phase=ElasticPhase.FAILED.value,
                      finished_ns=time.perf_counter_ns(), error=error)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            values = [self._records[key] for key in sorted(self._records)]
        return [value.public_dict() for value in values]

    def _replace(self, request_id: str, **changes: Any) -> None:
        with self._lock:
            self._records[request_id] = replace(self._get(request_id), **changes)

    def _get(self, request_id: str) -> ElasticRouterRecord:
        value = self._records.get(request_id)
        if value is None:
            raise ValueError("unknown request_id")
        return value


def _headers(record: ElasticRouterRecord) -> dict[str, str]:
    return {
        "X-Tempo-PD-Schema": ROUTER_SCHEMA,
        "X-Tempo-PD-Request-Id": record.request_id,
        "X-Tempo-PD-Arm": record.arm.value,
        "X-Tempo-PD-Route": record.route.value,
        "X-Tempo-PD-Reason": record.reason,
        "X-Tempo-PD-Profile": record.profile_id,
        "X-Tempo-PD-Profile-SHA256": record.profile_fingerprint_sha256,
    }


def build_app(
    config: base.RouterConfig,
    profile: ElasticPDProfile,
    *,
    allow_screen_profile: bool = False,
    queue_wait_ms: float = 100.0,
) -> FastAPI:
    base._require(queue_wait_ms >= 0 and math.isfinite(queue_wait_ms),
                  "queue_wait_ms must be finite and nonnegative")
    core = ElasticPDRouterCore(
        config, profile, allow_screen_profile=allow_screen_profile
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.local = httpx.AsyncClient(base_url=config.local_url, timeout=None)
        app.state.remote = httpx.AsyncClient(base_url=config.remote_url, timeout=None)
        app.state.tokenizer = httpx.AsyncClient(base_url=config.tokenizer_url, timeout=None)
        try:
            yield
        finally:
            await app.state.local.aclose()
            await app.state.remote.aclose()
            await app.state.tokenizer.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.tempo_core = core

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "schema": ROUTER_SCHEMA, "ok": True,
            "profile_id": profile.profile_id,
            "profile_fingerprint_sha256": profile.fingerprint_sha256,
        }

    @app.get("/tempo/decisions")
    async def decisions() -> dict[str, Any]:
        rows = core.records()
        return {"schema": ROUTER_SCHEMA, "count": len(rows), "decisions": rows}

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_id = request.headers.get(base.REQUEST_ID_HEADER)
        if not request_id:
            raise HTTPException(status_code=400, detail=f"missing {base.REQUEST_ID_HEADER}")
        upstream = None
        try:
            payload = await request.json()
            base._require(isinstance(payload, dict), "request body must be an object")
            base._require(payload.get("model") == config.served_model_name,
                          "served model mismatch")
            output_tokens = payload.get("max_tokens")
            base._require(type(output_tokens) is int and output_tokens >= 2,
                          "max_tokens must be at least two")
            prompt = payload.get("prompt")
            if isinstance(prompt, list):
                base._require(bool(prompt) and all(type(value) is int for value in prompt),
                              "token prompt must contain ints")
                prompt_tokens = len(prompt)
            else:
                base._require(isinstance(prompt, str) and prompt, "prompt must be nonempty")
                tokenized = await app.state.tokenizer.post("/tokenize", json={"prompt": prompt})
                tokenized.raise_for_status()
                tokens = tokenized.json().get("tokens")
                base._require(isinstance(tokens, list) and tokens,
                              "tokenizer returned no tokens")
                prompt_tokens = len(tokens)

            deadline_header = request.headers.get("x-tempo-remaining-deadline-ms")
            deadline = float(deadline_header) if deadline_header is not None else None
            record = core.decide(
                request_id=request_id, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens, remaining_deadline_ms=deadline,
            )
            queue_started_ns = time.perf_counter_ns()
            while record.route is ElasticRoute.QUEUE:
                elapsed_ms = (time.perf_counter_ns() - queue_started_ns) / 1_000_000
                if elapsed_ms >= queue_wait_ms:
                    core.fail(request_id, "bounded ingress queue timeout")
                    raise HTTPException(status_code=503, detail="elastic ingress queue timeout")
                await asyncio.sleep(min(0.001, (queue_wait_ms - elapsed_ms) / 1000))
                remaining = ((deadline - elapsed_ms) if deadline is not None else math.inf)
                if remaining <= 0:
                    core.fail(request_id, "request deadline expired in ingress queue")
                    raise HTTPException(status_code=503, detail="deadline expired in ingress queue")
                record = core.retry(request_id, remaining)

            client = app.state.remote if record.route is ElasticRoute.REMOTE else app.state.local
            forwarded_headers = {"Content-Type": "application/json"}
            for source, target in (("authorization", "Authorization"), ("session-id", "session-id")):
                value = request.headers.get(source)
                if value:
                    forwarded_headers[target] = value
            upstream_request = client.build_request(
                "POST", "/v1/completions", json=payload, headers=forwarded_headers
            )
            core.mark_upstream_started(request_id)
            upstream = await client.send(upstream_request, stream=True)
            upstream.raise_for_status()
            core.mark_response_started(request_id)
        except HTTPException:
            raise
        except Exception as exc:
            if request_id and any(row["request_id"] == request_id for row in core.records()):
                core.fail(request_id, f"{type(exc).__name__}: {exc}")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        async def generate():
            try:
                assert upstream is not None
                async for chunk in upstream.aiter_raw():
                    yield chunk
                core.complete(request_id)
            except BaseException as exc:
                core.fail(request_id, f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if upstream is not None:
                    await upstream.aclose()

        return StreamingResponse(generate(), media_type="text/event-stream", headers=_headers(record))

    return app


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--local-url", required=True)
    parser.add_argument("--remote-url", required=True)
    parser.add_argument("--tokenizer-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--remote-backend", required=True)
    parser.add_argument("--classifier-version", required=True)
    parser.add_argument("--decoder-load-bucket", required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--allow-screen-profile", action="store_true")
    parser.add_argument("--queue-wait-ms", type=float, default=100.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    profile = load_elastic_profile(args.profile.resolve())
    config = base.RouterConfig(
        mode=base.RouterMode.TEMPO_AUTO,
        local_url=args.local_url, remote_url=args.remote_url,
        tokenizer_url=args.tokenizer_url, served_model_name=args.served_model_name,
        model_id=args.model_id, model_revision=args.model_revision,
        topology_id=args.topology_id, remote_backend=args.remote_backend,
        classifier_version=args.classifier_version,
        decoder_load_bucket=args.decoder_load_bucket,
        kv_bytes_per_token=args.kv_bytes_per_token,
    )
    import uvicorn
    uvicorn.run(
        build_app(config, profile, allow_screen_profile=args.allow_screen_profile,
                  queue_wait_ms=args.queue_wait_ms),
        host=args.host, port=args.port, log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
