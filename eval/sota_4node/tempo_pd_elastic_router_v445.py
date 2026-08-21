#!/usr/bin/env python3
"""Corrected launch revision of the Elastic-PD ingress router.

The v444 implementation is retained as an auditable first draft.  This
revision maps a missing request deadline to a large finite sentinel (the pure
controller intentionally rejects infinities), including queued retries.
"""

from __future__ import annotations

import math

from eval.sota_4node import tempo_pd_elastic_router_v444 as prior
from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_elastic_profile_v444 import load_elastic_profile


ROUTER_SCHEMA = "tempo-elastic-pd-router-445"
NO_DEADLINE_SENTINEL_MS = 1_000_000_000.0


class ElasticPDRouterCore(prior.ElasticPDRouterCore):
    @staticmethod
    def _remaining_deadline(value: float | None) -> float:
        if value is None:
            return NO_DEADLINE_SENTINEL_MS
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("remaining_deadline_ms must be numeric")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError("remaining_deadline_ms must be finite and positive")
        return float(value)

    def retry(self, request_id: str, remaining_deadline_ms: float):
        if math.isinf(remaining_deadline_ms):
            remaining_deadline_ms = NO_DEADLINE_SENTINEL_MS
        return super().retry(request_id, remaining_deadline_ms)


def build_app(*args, **kwargs):
    # v444 resolves this class only while constructing the app.  Endpoint
    # closures retain the corrected instance after the global is restored.
    original = prior.ElasticPDRouterCore
    prior.ElasticPDRouterCore = ElasticPDRouterCore
    try:
        app = prior.build_app(*args, **kwargs)
    finally:
        prior.ElasticPDRouterCore = original
    app.state.tempo_elastic_schema = ROUTER_SCHEMA
    return app


def main(argv=None) -> int:
    args = prior._parse(argv)
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
            config, profile,
            allow_screen_profile=args.allow_screen_profile,
            queue_wait_ms=args.queue_wait_ms,
        ),
        host=args.host, port=args.port, log_level="info",
    )
    return 0


ElasticExperimentArm = prior.ElasticExperimentArm
ElasticRouterRecord = prior.ElasticRouterRecord


if __name__ == "__main__":
    raise SystemExit(main())
