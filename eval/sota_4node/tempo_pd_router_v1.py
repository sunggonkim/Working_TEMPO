#!/usr/bin/env python3
"""OpenAI-compatible TEMPO-PD router over local decode and official P/D proxy.

The router owns admission only.  A remote decision forwards the untouched
request to the official LMCache disaggregated-prefill proxy; a local decision
forwards it to the decoder.  Both fixed baselines use the same router and
classification overhead as ``tempo_auto``.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from enum import Enum
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from tempo.pd_admission import (
    PDAdmissionDecision,
    PDAdmissionLedger,
    PDRequestContext,
    PDRequestPhase,
    PDRoute,
    PDWorkloadClass,
)
from tempo.pd_policy_manifest import PDPolicyManifest, load_manifest


ROUTER_SCHEMA = "tempo-live-pd-router-1"
REQUEST_ID_HEADER = "x-tempo-request-id"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class RouterMode(str, Enum):
    FIXED_LOCAL = "fixed_local"
    LMCACHE_ALWAYS_REMOTE = "lmcache_always_remote"
    TEMPO_AUTO = "tempo_auto"


@dataclass(frozen=True)
class RouterConfig:
    mode: RouterMode
    local_url: str
    remote_url: str
    tokenizer_url: str
    served_model_name: str
    model_id: str
    model_revision: str
    topology_id: str
    remote_backend: str
    classifier_version: str
    decoder_load_bucket: str
    kv_bytes_per_token: int
    decision_capacity: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RouterMode):
            raise TypeError("mode must be RouterMode")
        for name in (
            "local_url", "remote_url", "tokenizer_url", "served_model_name",
            "model_id", "model_revision", "topology_id", "remote_backend",
            "classifier_version", "decoder_load_bucket",
        ):
            _require(isinstance(getattr(self, name), str) and getattr(self, name).strip(),
                     f"{name} must be nonempty")
        for name in ("local_url", "remote_url", "tokenizer_url"):
            _require(getattr(self, name).startswith(("http://", "https://")),
                     f"{name} must be HTTP(S)")
        _require(type(self.kv_bytes_per_token) is int and self.kv_bytes_per_token > 0,
                 "kv_bytes_per_token must be a positive int")
        _require(type(self.decision_capacity) is int and self.decision_capacity > 0,
                 "decision_capacity must be a positive int")


@dataclass(frozen=True)
class RouterDecision:
    request_id: str
    mode: RouterMode
    route: PDRoute
    reason: str
    workload: PDWorkloadClass
    profile_id: str | None
    manifest_id: str | None
    policy_epoch: int | None
    remote_advantage_lower_bound_ms: float | None
    prompt_tokens: int
    potential_kv_bytes: int
    decided_ns: int
    phase: str
    finished_ns: int | None = None
    error: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "mode": self.mode.value,
            "route": self.route.value,
            "reason": self.reason,
            "workload": self.workload.canonical_dict(),
            "workload_fingerprint": self.workload.fingerprint,
            "profile_id": self.profile_id,
            "manifest_id": self.manifest_id,
            "policy_epoch": self.policy_epoch,
            "remote_advantage_lower_bound_ms": self.remote_advantage_lower_bound_ms,
            "prompt_tokens": self.prompt_tokens,
            "potential_kv_bytes": self.potential_kv_bytes,
            "decided_ns": self.decided_ns,
            "phase": self.phase,
            "finished_ns": self.finished_ns,
            "error": self.error,
        }


class TempoPDRouterCore:
    """Synchronous policy/state core; HTTP is an adapter around this object."""

    def __init__(
        self,
        config: RouterConfig,
        manifest: PDPolicyManifest | None = None,
        *,
        allow_screen_profiles: bool = False,
    ) -> None:
        if not isinstance(config, RouterConfig):
            raise TypeError("config must be RouterConfig")
        if config.mode is RouterMode.TEMPO_AUTO:
            _require(manifest is not None, "tempo_auto requires a profile manifest")
        else:
            _require(manifest is None, "fixed baselines must not load a policy manifest")
        if manifest is not None:
            _require(manifest.classifier_version == config.classifier_version,
                     "classifier_version mismatch")
            self.policy = manifest.build_policy(
                allow_screen_profiles=allow_screen_profiles
            )
            self.ledger = PDAdmissionLedger(self.policy)
        else:
            self.policy = None
            self.ledger = None
        self.config = config
        self.manifest = manifest
        self._records: dict[str, RouterDecision] = {}
        self._lock = threading.Lock()

    def classify(
        self, *, prompt_tokens: int, output_tokens: int
    ) -> tuple[PDWorkloadClass, int]:
        _require(type(prompt_tokens) is int and prompt_tokens > 0,
                 "prompt_tokens must be positive")
        _require(type(output_tokens) is int and output_tokens >= 2,
                 "output_tokens must be at least two")
        version = self.config.classifier_version
        kv_bytes = prompt_tokens * self.config.kv_bytes_per_token
        workload = PDWorkloadClass(
            model_id=self.config.model_id,
            model_revision=self.config.model_revision,
            topology_id=self.config.topology_id,
            remote_backend=self.config.remote_backend,
            prompt_bucket=f"{version}:prompt_tokens:{prompt_tokens}",
            output_bucket=f"{version}:output_tokens:{output_tokens}",
            decoder_load_bucket=f"{version}:decoder_load:{self.config.decoder_load_bucket}",
            kv_bytes_bucket=f"{version}:kv_bytes:{kv_bytes}",
        )
        return workload, kv_bytes

    def decide(
        self,
        *,
        request_id: str,
        prompt_tokens: int,
        output_tokens: int,
        remaining_deadline_ms: float | None = None,
    ) -> RouterDecision:
        _require(isinstance(request_id, str) and request_id.strip(),
                 "request_id must be nonempty")
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens
        )
        with self._lock:
            _require(request_id not in self._records, "duplicate request_id")
            _require(len(self._records) < self.config.decision_capacity,
                     "decision capacity exhausted")

        core: PDAdmissionDecision | None = None
        if self.config.mode is RouterMode.FIXED_LOCAL:
            route = PDRoute.DECODER_LOCAL
            reason = "fixed_local_baseline"
        elif self.config.mode is RouterMode.LMCACHE_ALWAYS_REMOTE:
            route = PDRoute.REMOTE_PREFILL
            reason = "fixed_official_lmcache_remote_baseline"
        else:
            assert self.ledger is not None and self.manifest is not None
            core = self.ledger.admit(PDRequestContext(
                request_id=request_id,
                workload=workload,
                policy_epoch=self.manifest.policy_epoch,
                remote_backend_available=True,
                remaining_deadline_ms=remaining_deadline_ms,
            ))
            route = core.route
            reason = core.reason.value
        record = RouterDecision(
            request_id=request_id,
            mode=self.config.mode,
            route=route,
            reason=reason,
            workload=workload,
            profile_id=core.profile_id if core else None,
            manifest_id=self.manifest.manifest_id if self.manifest else None,
            policy_epoch=self.manifest.policy_epoch if self.manifest else None,
            remote_advantage_lower_bound_ms=(
                core.remote_advantage_lower_bound_ms if core else None
            ),
            prompt_tokens=prompt_tokens,
            potential_kv_bytes=kv_bytes,
            decided_ns=time.perf_counter_ns(),
            phase=(
                PDRequestPhase.REMOTE_SELECTED.value
                if route is PDRoute.REMOTE_PREFILL
                else PDRequestPhase.LOCAL_SELECTED.value
            ),
        )
        with self._lock:
            # The policy ledger and router registry share a request identity.
            _require(request_id not in self._records, "duplicate request_id")
            self._records[request_id] = record
        return record

    def mark_upstream_started(self, request_id: str) -> None:
        with self._lock:
            record = self._get(request_id)
        if self.ledger is not None and record.route is PDRoute.REMOTE_PREFILL:
            self.ledger.mark_remote_started(request_id)
        phase = (
            PDRequestPhase.REMOTE_STARTED.value
            if record.route is PDRoute.REMOTE_PREFILL
            else record.phase
        )
        self._replace(request_id, phase=phase)

    def mark_response_started(self, request_id: str) -> None:
        with self._lock:
            record = self._get(request_id)
        if self.ledger is not None:
            self.ledger.mark_decode_started(request_id)
        self._replace(request_id, phase=PDRequestPhase.DECODE_STARTED.value)

    def complete(self, request_id: str) -> None:
        if self.ledger is not None:
            self.ledger.complete(request_id)
        self._replace(
            request_id,
            phase=PDRequestPhase.COMPLETE.value,
            finished_ns=time.perf_counter_ns(),
        )

    def fail(self, request_id: str, error: str) -> None:
        _require(bool(error), "error must be nonempty")
        if self.ledger is not None:
            ledger_record = self.ledger.record(request_id)
            if ledger_record.phase not in {PDRequestPhase.COMPLETE, PDRequestPhase.FAILED}:
                self.ledger.fail(request_id, error)
        self._replace(
            request_id,
            phase=PDRequestPhase.FAILED.value,
            finished_ns=time.perf_counter_ns(),
            error=error,
        )

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            values = [self._records[key] for key in sorted(self._records)]
        return [value.public_dict() for value in values]

    def _replace(self, request_id: str, **changes: Any) -> None:
        with self._lock:
            self._records[request_id] = replace(self._get(request_id), **changes)

    def _get(self, request_id: str) -> RouterDecision:
        value = self._records.get(request_id)
        if value is None:
            raise ValueError("unknown request_id")
        return value


def _decision_headers(record: RouterDecision) -> dict[str, str]:
    return {
        "X-Tempo-PD-Schema": ROUTER_SCHEMA,
        "X-Tempo-PD-Request-Id": record.request_id,
        "X-Tempo-PD-Mode": record.mode.value,
        "X-Tempo-PD-Route": record.route.value,
        "X-Tempo-PD-Reason": record.reason,
        "X-Tempo-PD-Workload": record.workload.fingerprint,
        "X-Tempo-PD-Profile": record.profile_id or "none",
        "X-Tempo-PD-Manifest": record.manifest_id or "none",
    }


def build_app(
    config: RouterConfig,
    manifest: PDPolicyManifest | None = None,
    *,
    allow_screen_profiles: bool = False,
) -> FastAPI:
    core = TempoPDRouterCore(
        config, manifest, allow_screen_profiles=allow_screen_profiles
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
            "schema": ROUTER_SCHEMA,
            "ok": True,
            "mode": config.mode.value,
            "manifest_id": manifest.manifest_id if manifest else None,
        }

    @app.get("/tempo/decisions")
    async def decisions() -> dict[str, Any]:
        rows = core.records()
        return {"schema": ROUTER_SCHEMA, "count": len(rows), "decisions": rows}

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_id = request.headers.get(REQUEST_ID_HEADER)
        if not request_id:
            raise HTTPException(status_code=400, detail=f"missing {REQUEST_ID_HEADER}")
        try:
            payload = await request.json()
            _require(isinstance(payload, dict), "request body must be an object")
            _require(payload.get("model") == config.served_model_name, "served model mismatch")
            output_tokens = payload.get("max_tokens")
            _require(type(output_tokens) is int and output_tokens >= 2,
                     "max_tokens must be at least two")
            prompt = payload.get("prompt")
            if isinstance(prompt, list):
                _require(bool(prompt) and all(type(value) is int for value in prompt),
                         "token prompt must contain ints")
                prompt_tokens = len(prompt)
            else:
                _require(isinstance(prompt, str) and prompt, "prompt must be nonempty")
                tokenized = await app.state.tokenizer.post("/tokenize", json={"prompt": prompt})
                tokenized.raise_for_status()
                tokens = tokenized.json().get("tokens")
                _require(isinstance(tokens, list) and tokens, "tokenizer returned no tokens")
                prompt_tokens = len(tokens)
            record = core.decide(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            client = (
                app.state.remote
                if record.route is PDRoute.REMOTE_PREFILL
                else app.state.local
            )
            forwarded_headers = {"Content-Type": "application/json"}
            authorization = request.headers.get("authorization")
            if authorization:
                forwarded_headers["Authorization"] = authorization
            session = request.headers.get("session-id")
            if session:
                forwarded_headers["session-id"] = session
            upstream_request = client.build_request(
                "POST", "/v1/completions", json=payload, headers=forwarded_headers
            )
            core.mark_upstream_started(request_id)
            upstream = await client.send(upstream_request, stream=True)
            upstream.raise_for_status()
            core.mark_response_started(request_id)
        except Exception as exc:
            if request_id and any(row["request_id"] == request_id for row in core.records()):
                core.fail(request_id, f"{type(exc).__name__}: {exc}")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        async def generate():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
                core.complete(request_id)
            except BaseException as exc:
                core.fail(request_id, f"{type(exc).__name__}: {exc}")
                raise
            finally:
                await upstream.aclose()

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=_decision_headers(record)
        )

    return app


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=[value.value for value in RouterMode], required=True)
    parser.add_argument("--local-url", required=True)
    parser.add_argument("--remote-url", required=True)
    parser.add_argument("--tokenizer-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--remote-backend", default="lmcache-ucx")
    parser.add_argument("--classifier-version", required=True)
    parser.add_argument("--decoder-load-bucket", required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-screen-profiles", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    mode = RouterMode(args.mode)
    if mode is RouterMode.TEMPO_AUTO:
        _require(args.manifest is not None, "tempo_auto requires --manifest")
        manifest = load_manifest(args.manifest.resolve())
    else:
        _require(args.manifest is None, "fixed mode forbids --manifest")
        _require(not args.allow_screen_profiles,
                 "fixed mode forbids --allow-screen-profiles")
        manifest = None
    config = RouterConfig(
        mode=mode,
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
    uvicorn.run(build_app(config, manifest, allow_screen_profiles=args.allow_screen_profiles),
                host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
