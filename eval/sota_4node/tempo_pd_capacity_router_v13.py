#!/usr/bin/env python3
"""TEMPO-PD pair router with one conservative remote in-flight credit."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import threading
import time

from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_admission import PDRequestPhase, PDRoute


CAPACITY_SCHEMA = "tempo-pd-remote-credit-router-13"


class CreditCore(base.TempoPDRouterCore):
    """Select remote only when this pair owns an unused request credit."""

    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        if manifest is not None or allow_screen_profiles:
            raise ValueError("credit router forbids calibration manifests")
        self.config = config
        self.manifest = None
        self.policy = None
        self.ledger = None
        self._records = {}
        self._lock = threading.Lock()
        self._remote_owner: str | None = None
        self._reserved: set[str] = set()

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        del remaining_deadline_ms
        base._require(isinstance(request_id, str) and request_id.strip(),
                      "request_id must be nonempty")
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens
        )
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            base._require(len(self._records) < self.config.decision_capacity,
                          "decision capacity exhausted")
            if self._remote_owner is None:
                route = PDRoute.REMOTE_PREFILL
                reason = "remote_credit_acquired_idle_pair"
                self._remote_owner = request_id
                self._reserved.add(request_id)
            else:
                route = PDRoute.DECODER_LOCAL
                reason = "local_fallback_remote_credit_busy"
            record = base.RouterDecision(
                request_id=request_id,
                mode=base.RouterMode.TEMPO_AUTO,
                route=route,
                reason=reason,
                workload=workload,
                profile_id=None,
                manifest_id=None,
                policy_epoch=None,
                remote_advantage_lower_bound_ms=None,
                prompt_tokens=prompt_tokens,
                potential_kv_bytes=kv_bytes,
                decided_ns=time.perf_counter_ns(),
                phase=(PDRequestPhase.REMOTE_SELECTED.value
                       if route is PDRoute.REMOTE_PREFILL
                       else PDRequestPhase.LOCAL_SELECTED.value),
            )
            self._records[request_id] = record
        return record

    def _release(self, request_id: str) -> None:
        with self._lock:
            if request_id in self._reserved:
                base._require(self._remote_owner == request_id,
                              "remote credit owner mismatch")
                self._reserved.remove(request_id)
                self._remote_owner = None

    def complete(self, request_id: str) -> None:
        self._release(request_id)
        self._replace(
            request_id, phase=PDRequestPhase.COMPLETE.value,
            finished_ns=time.perf_counter_ns(),
        )

    def fail(self, request_id: str, error: str) -> None:
        self._release(request_id)
        base._require(bool(error), "error must be nonempty")
        self._replace(
            request_id, phase=PDRequestPhase.FAILED.value,
            finished_ns=time.perf_counter_ns(), error=error,
        )


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)
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
    original = base.TempoPDRouterCore
    base.TempoPDRouterCore = CreditCore
    try:
        import uvicorn
        app = base.build_app(config)
        app.state.tempo_capacity_schema = CAPACITY_SCHEMA
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        base.TempoPDRouterCore = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
