#!/usr/bin/env python3
"""On-demand, endpoint-local evidence probe for Perlmutter P/D runs.

The probe never polls.  A characterization client explicitly requests a
snapshot at a block boundary.  Each response contains one local vLLM gauge
snapshot and one local Cassini counter delta; timestamps remain owned by the
endpoint that produced them.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from eval.sota_4node.tempo_pd_elastic_router import parse_vllm_load_metrics
from prometheus_client.parser import text_string_to_metric_families
from tempo.cassini_endpoint import CassiniEndpointSampler
from tempo.domain_evidence import CounterSupport
from tempo.pd_endpoint_evidence import (
    PDEndpointIdentity,
    PDEndpointRole,
    PDEndpointSnapshot,
    endpoint_metric_names,
    endpoint_metrics,
)


SCHEMA = "tempo-pd-endpoint-probe-v3"
VLLM_CUMULATIVE_SCHEMA = "tempo-vllm-endpoint-cumulative-v1"
VLLM_CUMULATIVE_METRICS = frozenset({
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_inference_time_seconds_sum",
    "vllm:request_inference_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:request_prefill_kv_computed_tokens_count",
})


def parse_vllm_endpoint_cumulative(
    metrics_text: str, *, served_model_name: str,
) -> dict[str, object]:
    """Aggregate an exact cumulative metric inventory across local engines."""
    if not metrics_text.strip() or not served_model_name:
        raise ValueError("metrics text and served model name must be nonempty")
    values = {name: 0.0 for name in VLLM_CUMULATIVE_METRICS}
    seen = {name: False for name in VLLM_CUMULATIVE_METRICS}
    engines: set[int] = set()
    try:
        families = text_string_to_metric_families(metrics_text)
        for family in families:
            for sample in family.samples:
                if sample.name not in values:
                    continue
                if sample.labels.get("model_name") != served_model_name:
                    continue
                engine = sample.labels.get("engine")
                if not isinstance(engine, str) or not engine.isdigit():
                    raise ValueError("cumulative vLLM metric lacks engine label")
                engine_index = int(engine)
                if str(engine_index) != engine:
                    raise ValueError("cumulative vLLM engine label is invalid")
                value = float(sample.value)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("cumulative vLLM metric is invalid")
                values[sample.name] += value
                seen[sample.name] = True
                engines.add(engine_index)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid cumulative vLLM metrics payload") from exc
    missing = sorted(name for name, present in seen.items() if not present)
    if missing:
        raise ValueError(f"cumulative vLLM metrics are missing: {missing}")
    normalized: dict[str, int | float] = {}
    for name, value in values.items():
        if name.endswith("_total") or name.endswith("_count"):
            if not value.is_integer():
                raise ValueError("counter/count cumulative metric is not integral")
            normalized[name] = int(value)
        else:
            normalized[name] = value
    return {
        "schema": VLLM_CUMULATIVE_SCHEMA,
        "source": "vllm_prometheus_on_demand",
        "model_name": served_model_name,
        "engine_indices": sorted(engines),
        "values": dict(sorted(normalized.items())),
    }


def validate_vllm_endpoint_cumulative(raw: object) -> None:
    if not isinstance(raw, dict):
        raise TypeError("cumulative vLLM evidence must be an object")
    if raw.get("schema") != VLLM_CUMULATIVE_SCHEMA:
        raise ValueError("cumulative vLLM evidence schema mismatch")
    if raw.get("source") != "vllm_prometheus_on_demand":
        raise ValueError("cumulative vLLM evidence source mismatch")
    if not isinstance(raw.get("model_name"), str) or not raw["model_name"]:
        raise ValueError("cumulative vLLM model name is invalid")
    engines = raw.get("engine_indices")
    if not isinstance(engines, list) or not engines:
        raise ValueError("cumulative vLLM engine set is empty")
    if engines != sorted(set(engines)) or any(
        type(engine) is not int or engine < 0 for engine in engines
    ):
        raise ValueError("cumulative vLLM engine set is invalid")
    values = raw.get("values")
    if not isinstance(values, dict) or set(values) != VLLM_CUMULATIVE_METRICS:
        raise ValueError("cumulative vLLM metric inventory is not exact")
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("cumulative vLLM metric must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError("cumulative vLLM metric must be non-negative")
        if (name.endswith("_total") or name.endswith("_count")) and type(
            value
        ) is not int:
            raise TypeError("counter/count cumulative vLLM metric must be int")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--role", choices=[item.value for item in PDEndpointRole],
                        required=True)
    parser.add_argument("--pair-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--vllm-metrics-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--metrics-timeout-s", type=float, default=3.0)
    parser.add_argument("--metrics-attempts", type=int, default=1)
    return parser.parse_args()


class EndpointProbe:
    """Serialize on-demand samples for one local P/D endpoint."""

    def __init__(
        self,
        *,
        identity: PDEndpointIdentity,
        vllm_metrics_url: str,
        served_model_name: str,
        metrics_timeout_s: float,
        metrics_attempts: int = 1,
    ) -> None:
        if not vllm_metrics_url.startswith("http://"):
            raise ValueError("vLLM metrics URL must use explicit HTTP")
        if not served_model_name:
            raise ValueError("served model name must be nonempty")
        if not 0.0 < metrics_timeout_s <= 10.0:
            raise ValueError("metrics timeout must be in (0, 10]")
        if type(metrics_attempts) is not int or not 1 <= metrics_attempts <= 3:
            raise ValueError("metrics attempts must be in [1, 3]")
        self.identity = identity
        self.vllm_metrics_url = vllm_metrics_url.rstrip("/") + "/metrics"
        self.served_model_name = served_model_name
        self.metrics_timeout_s = metrics_timeout_s
        self.metrics_attempts = metrics_attempts
        # The client samples once at each half-block boundary.  Ten seconds is
        # deliberately bounded but covers the 7.5 s half of the frozen run.
        self.cassini = CassiniEndpointSampler(
            identity, min_interval_ms=20.0, max_window_ms=10_000.0)
        self._sequence = 0
        self._lock = threading.Lock()

    def _load_metrics(self) -> tuple[str, dict[str, object]]:
        started_ns = time.perf_counter_ns()
        errors: list[dict[str, object]] = []
        for attempt in range(1, self.metrics_attempts + 1):
            try:
                with urlopen(
                    self.vllm_metrics_url, timeout=self.metrics_timeout_s,
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"vLLM metrics returned HTTP {response.status}")
                    text = response.read().decode("utf-8")
                return text, {
                    "attempts_configured": self.metrics_attempts,
                    "attempts_used": attempt,
                    "timeout_s_per_attempt": self.metrics_timeout_s,
                    "retry_backoff_s": 0.05,
                    "transient_errors": errors,
                    "elapsed_ns": time.perf_counter_ns() - started_ns,
                }
            except (HTTPError, URLError, TimeoutError) as exc:
                errors.append({
                    "attempt": attempt,
                    "error": type(exc).__name__,
                    "http_status": getattr(exc, "code", None),
                    "message": str(exc),
                })
                if attempt == self.metrics_attempts:
                    raise RuntimeError(
                        "vLLM metrics fetch exhausted bounded retries: "
                        f"{errors}") from exc
                time.sleep(0.05)
        raise AssertionError("unreachable metrics retry state")

    def _load_snapshot(
        self,
    ) -> tuple[PDEndpointSnapshot, dict[str, object], dict[str, object]]:
        text, fetch = self._load_metrics()
        parsed = parse_vllm_load_metrics(
            text, served_model_name=self.served_model_name)
        cumulative = parse_vllm_endpoint_cumulative(
            text, served_model_name=self.served_model_name)
        supported = {
            "running_requests": parsed["num_requests_running"],
            "waiting_requests": parsed["num_requests_waiting"],
            "kv_cache_usage_fraction": parsed["kv_cache_usage_perc"],
        }
        unavailable = {
            name: CounterSupport.NOT_COLLECTED
            for name in endpoint_metric_names(self.identity.role)
            if name not in supported
        }
        self._sequence += 1
        snapshot = PDEndpointSnapshot(
            identity=self.identity,
            sequence=self._sequence,
            endpoint_monotonic_ns=time.perf_counter_ns(),
            source="vllm_prometheus_on_demand",
            metrics=endpoint_metrics(
                self.identity.role,
                supported=supported,
                unavailable=unavailable,
            ),
        )
        return snapshot, cumulative, fetch

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            load, cumulative, fetch = self._load_snapshot()
            cassini = self.cassini.sample(force=True)
        return {
            "schema": SCHEMA,
            "endpoint": load.as_dict(),
            "vllm_cumulative": cumulative,
            "vllm_metrics_fetch": fetch,
            "cassini": cassini,
        }


def _handler(probe: EndpointProbe):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, value: dict[str, object]) -> None:
            payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            if self.path == "/health":
                self._json(200, {"schema": SCHEMA, "status": "ok"})
                return
            if self.path != "/snapshot":
                self._json(404, {"schema": SCHEMA, "error": "not_found"})
                return
            try:
                self._json(200, probe.snapshot())
            except Exception as exc:  # fail closed at the HTTP boundary
                self._json(503, {
                    "schema": SCHEMA,
                    "error": type(exc).__name__,
                    "message": str(exc),
                })

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> int:
    args = _parse()
    identity = PDEndpointIdentity(
        endpoint_id=args.endpoint_id,
        role=PDEndpointRole(args.role),
        pair_index=args.pair_index,
    )
    probe = EndpointProbe(
        identity=identity,
        vllm_metrics_url=args.vllm_metrics_url,
        served_model_name=args.served_model_name,
        metrics_timeout_s=args.metrics_timeout_s,
        metrics_attempts=args.metrics_attempts,
    )
    server = ThreadingHTTPServer((args.host, args.port), _handler(probe))
    server.serve_forever(poll_interval=0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
