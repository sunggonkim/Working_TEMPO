#!/usr/bin/env python3
"""Canonical actual-vLLM ingress router for TEMPO Elastic-PD."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from fastapi import HTTPException
from prometheus_client.parser import text_string_to_metric_families


from tempo.cassini_pressure import CassiniPressureSampler
from tempo.pd_decoder_cache_evidence import (
    DEFAULT_BLOCK_SIZE as DECODER_CACHE_BLOCK_SIZE,
    VLLMDecoderCacheSSEParser,
    full_prefix_hit_tokens,
)
from eval.sota_4node import tempo_pd_cache_reuse as cache_reuse
from eval.sota_4node import tempo_pd_elastic_router_v444 as wire
from eval.sota_4node import tempo_pd_elastic_router_v448 as runtime
from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_elastic_controller import (
    CacheResidency, CacheResidencyCatalog, ElasticPDController, ElasticPhase,
    ElasticRoute,
)
from tempo.pd_elastic_profile import load_elastic_profile, require_replicated_profile
from tempo.pd_endpoint_controller import (
    EndpointFeedbackController,
    EndpointRequest,
    EndpointRoute,
    EndpointWork,
)
from tempo.pd_endpoint_profile import (
    SCHEMA_V2 as ENDPOINT_PROFILE_V2_SCHEMA,
    SEMANTIC_EPOCH_POLICY as PROFILE_SEMANTIC_EPOCH_POLICY,
    load_endpoint_service_profile,
)


ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"
ElasticExperimentArm = wire.ElasticExperimentArm
DECODER_INDEX_HEADER = "X-Tempo-PD-Decoder-Index"
TRANSFER_EVIDENCE_COMPLETE = "complete"
TRANSFER_EVIDENCE_OVERLAPPED = "eof_complete_after_control_overlap"
ElasticRouterRecord = wire.ElasticRouterRecord
VLLM_LOAD_SNAPSHOT_SCHEMA = "tempo-vllm-load-snapshot-v1"
VLLM_LOAD_DECISION_MODE = "observe_only"
VLLM_LOAD_DISABLED_MODE = "disabled"
VLLM_LOAD_DISABLED_SOURCE = "explicitly_disabled_no_request_rpc"
VLLM_LOAD_SNAPSHOT_MODE_ENV = "TEMPO_VLLM_LOAD_SNAPSHOT_MODE"
COLD_MEASURED_ENV = "TEMPO_PD_BENCHMARK_COLD_MEASURED"
P_ONLY_MEASURED_MARKER = "-cache-p-only-measured-"
D_ONLY_MEASURED_MARKER = "-cache-d-only-measured-"
BOTH_MEASURED_MARKER = "-cache-both-measured-"
MISS_MEASURED_MARKER = "-cache-miss-measured-"
D_CACHE_SEED_MARKER = "-cache-d-seed-"
D_CACHE_PROBE_MARKER = "-cache-d-probe-"
VLLM_SKIP_LOCAL_PREFIX_READ_XARG = "tempo_skip_local_prefix_cache_read"
PROXY_DECODER_SKIP_LOCAL_PREFIX_READ_FIELD = (
    "tempo_decoder_skip_local_prefix_cache_read"
)
PRESSURE_SCHEMA = "tempo-elastic-pd-dual-pressure-v1"
PRESSURE_MODE_ENV = "TEMPO_PD_PRESSURE_MODE"
PRESSURE_DISABLED_MODE = "disabled"
PRESSURE_OBSERVE_MODE = "observe_only"
PRESSURE_ADAPTIVE_MODE = "adaptive"
FRONTEND_SEMANTIC_LOAD_SCHEMA = "tempo-frontend-semantic-load-v1"
ENDPOINT_FEEDBACK_MODE_ENV = "TEMPO_PD_ENDPOINT_FEEDBACK_MODE"
ENDPOINT_FEEDBACK_DISABLED_MODE = "disabled"
ENDPOINT_FEEDBACK_ADAPTIVE_MODE = "adaptive"
ENDPOINT_ROUTING_POLICY_ENV = "TEMPO_PD_ENDPOINT_ROUTING_POLICY"
ENDPOINT_INSTANT_SCORE_POLICY = "instant_score_v1"
ENDPOINT_SEMANTIC_EPOCH_POLICY = "semantic_epoch_v1"
SEMANTIC_EPOCH_SCHEMA = "tempo-pd-semantic-epoch-v1"
ENDPOINT_SERVICE_PROFILE_ENV = "TEMPO_PD_ENDPOINT_SERVICE_PROFILE"
ENDPOINT_WORKLOAD_MANIFEST_SHA256_ENV = (
    "TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256"
)
ENDPOINT_PASSIVE_FEEDBACK_ENV = "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK"
ENDPOINT_PASSIVE_MARKER = "-endpoint-observed-"
REMOTE_PRESSURE_MS_PER_PROMPT_TOKEN_ENV = (
    "TEMPO_PD_REMOTE_PRESSURE_MS_PER_PROMPT_TOKEN")
LOCAL_PRESSURE_MS_PER_PROMPT_TOKEN_ENV = (
    "TEMPO_PD_LOCAL_PRESSURE_MS_PER_PROMPT_TOKEN")
FABRIC_PAUSE_FLOOR = 0.01
FABRIC_PAUSE_CEILING = 0.20
HOST_BLOCKED_FLOOR = 4.0
HOST_BLOCKED_CEILING = 24.0
FABRIC_EWMA_ALPHA = 0.35
FABRIC_CONGESTION_ENTER = 0.35
FABRIC_CONGESTION_EXIT = 0.15
_VLLM_LOAD_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)


def explicit_cache_contract(request_id: str) -> str | None:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be nonempty")
    matches = [
        name for marker, name in (
            (P_ONLY_MEASURED_MARKER, "p_only"),
            (D_ONLY_MEASURED_MARKER, "d_only"),
            (BOTH_MEASURED_MARKER, "both"),
            (MISS_MEASURED_MARKER, "miss"),
        )
        if marker in request_id
    ]
    if len(matches) > 1:
        raise ValueError("request_id has conflicting cache contracts")
    return matches[0] if matches else None


def enforce_explicit_cache_contract(
    request_id: str, observed: CacheResidency,
) -> CacheResidency:
    """Resolve preregistered state only when physical evidence permits it.

    P_ONLY/D_ONLY/BOTH must already exist in the completion-backed catalog
    before route commitment.  MISS is the sole constructive state: its exact
    prompt namespace must be unseen, after which completion still has to
    report zero source/APC hits.  This gives endpoint-profile lookup a concrete
    MISS row without allowing a request-ID label to invent warm residency.
    """

    if not isinstance(observed, CacheResidency):
        raise TypeError("observed cache residency must be CacheResidency")
    contract = explicit_cache_contract(request_id)
    if contract is None:
        return observed
    if contract == "miss":
        if observed is not CacheResidency.UNKNOWN:
            raise ValueError(
                "explicit MISS request namespace was previously observed")
        return CacheResidency.MISS
    expected = {
        "p_only": CacheResidency.P_ONLY,
        "d_only": CacheResidency.D_ONLY,
        "both": CacheResidency.BOTH,
    }[contract]
    if observed is not expected:
        raise ValueError(
            f"explicit {contract.upper()} request lacks completed cache "
            f"evidence: observed={observed.value}"
        )
    return observed


def parse_vllm_load_metrics(
    metrics_text: str, *, served_model_name: str,
) -> dict[str, Any]:
    """Parse one strict, engine-complete vLLM Prometheus snapshot."""
    if not isinstance(metrics_text, str) or not metrics_text.strip():
        raise ValueError("vLLM metrics payload must be nonempty text")
    if not isinstance(served_model_name, str) or not served_model_name:
        raise ValueError("served_model_name must be nonempty")
    values: dict[str, dict[int, float]] = {
        metric: {} for metric in _VLLM_LOAD_METRICS
    }
    try:
        families = text_string_to_metric_families(metrics_text)
        for family in families:
            for sample in family.samples:
                if sample.name not in values:
                    continue
                if sample.labels.get("model_name") != served_model_name:
                    continue
                engine = sample.labels.get("engine")
                if not isinstance(engine, str) or not engine:
                    raise ValueError("vLLM load metric lacks engine label")
                try:
                    engine_index = int(engine)
                except ValueError as exc:
                    raise ValueError(
                        "vLLM load metric engine label is not an integer"
                    ) from exc
                if engine_index < 0 or str(engine_index) != engine:
                    raise ValueError("vLLM load metric engine label is invalid")
                metric_values = values[sample.name]
                if engine_index in metric_values:
                    raise ValueError(
                        "duplicate vLLM load metric for model and engine"
                    )
                value = float(sample.value)
                if not math.isfinite(value):
                    raise ValueError("vLLM load metric must be finite")
                metric_values[engine_index] = value
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid vLLM Prometheus metrics payload") from exc

    engine_sets = [set(values[metric]) for metric in _VLLM_LOAD_METRICS]
    if any(not engines for engines in engine_sets):
        raise ValueError("required vLLM load metric is missing")
    if any(engines != engine_sets[0] for engines in engine_sets[1:]):
        raise ValueError("vLLM load metric engine label sets differ")
    for metric in _VLLM_LOAD_METRICS[:2]:
        if any(
            value < 0 or not value.is_integer()
            for value in values[metric].values()
        ):
            raise ValueError("vLLM request-count metric is invalid")
    kv_values = values["vllm:kv_cache_usage_perc"].values()
    if any(value < 0 or value > 1 for value in kv_values):
        raise ValueError("vLLM KV-cache usage metric is outside [0, 1]")
    return {
        "schema": VLLM_LOAD_SNAPSHOT_SCHEMA,
        "source": "local_decoder_prometheus_request_start",
        "decision_mode": VLLM_LOAD_DECISION_MODE,
        "model_name": served_model_name,
        "engine_indices": sorted(engine_sets[0]),
        "num_requests_running": sum(
            int(value)
            for value in values["vllm:num_requests_running"].values()
        ),
        "num_requests_waiting": sum(
            int(value)
            for value in values["vllm:num_requests_waiting"].values()
        ),
        "kv_cache_usage_perc": max(kv_values),
    }


def _unit_interval(value: float, low: float, high: float) -> float:
    if not all(math.isfinite(item) for item in (value, low, high)):
        raise ValueError("pressure normalization values must be finite")
    if high <= low:
        raise ValueError("pressure normalization high must exceed low")
    return min(1.0, max(0.0, (value - low) / (high - low)))


def pressure_penalties_ms(
    *, prompt_tokens: int, local_pressure: float, fabric_pressure: float,
    local_ms_per_prompt_token: float, remote_ms_per_prompt_token: float,
) -> tuple[float, float]:
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    for name, value in (
        ("local_pressure", local_pressure),
        ("fabric_pressure", fabric_pressure),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    for name, value in (
        ("local_ms_per_prompt_token", local_ms_per_prompt_token),
        ("remote_ms_per_prompt_token", remote_ms_per_prompt_token),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 0.25:
            raise ValueError(f"{name} must be in [0, 0.25]")
    return (
        prompt_tokens * local_pressure * local_ms_per_prompt_token,
        prompt_tokens * fabric_pressure * remote_ms_per_prompt_token,
    )


def _coefficient(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 0.25:
        raise ValueError(f"{name} must be in [0, 0.25]")
    return value


class ElasticPDRouterCore(runtime.ElasticPDRouterCore):
    def __init__(self, config, profile, *, cache_catalog=None,
                 cache_residency=None, allow_screen_profile=False):
        self.cache_catalog = cache_catalog or CacheResidencyCatalog()
        self._request_cache_namespaces = {}
        self._request_prompt_keys = {}
        self._request_token_ids = {}
        self._request_upstream_priorities = {}
        self._request_remote_decoder_indices = {}
        self._request_decision_cache_residencies = {}
        self._request_vllm_load_snapshots = {}
        self._request_pressure_snapshots = {}
        self._request_endpoint_decisions = {}
        self._request_endpoint_feedback = {}
        self._endpoint_requests = {}
        self._passive_endpoint_requests = {}
        self._request_frontend_semantic_loads = {}
        self._request_semantic_epoch_decisions = {}
        self._semantic_epoch_route = EndpointRoute.LOCAL
        self._semantic_epoch_generation = 0
        self._semantic_high_streak = 0
        self._semantic_low_streak = 0
        self._served_model_name = config.served_model_name
        self.vllm_load_snapshot_mode = os.environ.get(
            VLLM_LOAD_SNAPSHOT_MODE_ENV, VLLM_LOAD_DISABLED_MODE)
        if self.vllm_load_snapshot_mode not in (
            VLLM_LOAD_DISABLED_MODE, VLLM_LOAD_DECISION_MODE,
        ):
            raise ValueError(
                f"{VLLM_LOAD_SNAPSHOT_MODE_ENV} must be disabled or observe_only")
        raw_cold_measured = os.environ.get(COLD_MEASURED_ENV, "0")
        if raw_cold_measured not in ("0", "1"):
            raise ValueError(
                f"{COLD_MEASURED_ENV} must be 0 or 1")
        self.cold_measured = raw_cold_measured == "1"
        raw_decoder_prefix_caching = os.environ.get(
            "TEMPO_VLLM_DECODER_PREFIX_CACHING", "0")
        if raw_decoder_prefix_caching not in ("0", "1"):
            raise ValueError(
                "TEMPO_VLLM_DECODER_PREFIX_CACHING must be 0 or 1")
        self.decoder_prefix_caching = raw_decoder_prefix_caching == "1"
        self.decoder_reuse_items = cache_reuse.parse_reuse_items(
            os.environ.get("TEMPO_PD_DECODER_REUSE_ITEMS", "all"))
        raw_forward_token_ids = os.environ.get(
            "TEMPO_PD_FORWARD_TOKEN_IDS", "0")
        if raw_forward_token_ids not in ("0", "1"):
            raise ValueError("TEMPO_PD_FORWARD_TOKEN_IDS must be 0 or 1")
        self.forward_token_ids = raw_forward_token_ids == "1"
        raw_control_overlap = os.environ.get(
            "TEMPO_PD_PROXY_KV_CONTROL_OVERLAP", "0")
        if raw_control_overlap not in ("0", "1"):
            raise ValueError(
                "TEMPO_PD_PROXY_KV_CONTROL_OVERLAP must be 0 or 1")
        self.proxy_kv_control_overlap = raw_control_overlap == "1"
        self.remote_decode_placement = os.environ.get(
            "TEMPO_PD_REMOTE_DECODE_PLACEMENT", "paired")
        if self.remote_decode_placement not in (
            "paired", "cross", "long_decode_cross",
        ):
            raise ValueError(
                "TEMPO_PD_REMOTE_DECODE_PLACEMENT must be "
                "paired, cross, or long_decode_cross")
        raw_decoder_index = os.environ.get(
            "TEMPO_PD_LOCAL_DECODER_INDEX")
        self.local_decoder_index = None
        if raw_decoder_index is not None:
            try:
                self.local_decoder_index = int(raw_decoder_index)
            except ValueError as exc:
                raise ValueError(
                    "TEMPO_PD_LOCAL_DECODER_INDEX must be 0 or 1") from exc
        if (
            self.local_decoder_index not in (None, 0, 1)
            or self.remote_decode_placement == "long_decode_cross"
            and self.local_decoder_index is None
        ):
            raise ValueError(
                "TEMPO_PD_LOCAL_DECODER_INDEX must be 0 or 1")

        raw_priority = os.environ.get(
            "TEMPO_PD_REMOTE_CATCHUP_PRIORITY", "0")
        try:
            self.remote_catchup_priority = int(raw_priority)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_REMOTE_CATCHUP_PRIORITY must be 0, -1, or -2") from exc
        if self.remote_catchup_priority not in (0, -1, -2):
            raise ValueError(
                "TEMPO_PD_REMOTE_CATCHUP_PRIORITY must be 0, -1, or -2")
        raw_strong_catchup = os.environ.get(
            "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY", "0")
        try:
            self.strong_remote_catchup_priority = int(
                raw_strong_catchup)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY must be "
                "0, -1, or -2") from exc
        if self.strong_remote_catchup_priority not in (0, -1, -2):
            raise ValueError(
                "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY must be "
                "0, -1, or -2")
        raw_long_catchup = os.environ.get(
            "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY", "0")
        try:
            self.long_remote_catchup_priority = int(raw_long_catchup)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY must be 0, -1, or -2"
            ) from exc
        if self.long_remote_catchup_priority not in (0, -1, -2):
            raise ValueError(
                "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY must be 0, -1, or -2")
        raw_long_min_prompt = os.environ.get(
            "TEMPO_PD_LONG_REMOTE_CATCHUP_MIN_PROMPT_TOKENS", "0")
        try:
            self.long_remote_catchup_min_prompt_tokens = int(
                raw_long_min_prompt)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_LONG_REMOTE_CATCHUP_MIN_PROMPT_TOKENS must be "
                "0, 512, 1230, 2048, or 4094") from exc
        if self.long_remote_catchup_min_prompt_tokens not in (
            0, 512, 1230, 2048, 4094,
        ):
            raise ValueError(
                "TEMPO_PD_LONG_REMOTE_CATCHUP_MIN_PROMPT_TOKENS must be "
                "0, 512, 1230, 2048, or 4094")
        raw_median_guard = os.environ.get(
            "TEMPO_PD_MEDIAN_GUARD_PRIORITY", "0")
        try:
            self.median_guard_priority = int(raw_median_guard)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_MEDIAN_GUARD_PRIORITY must be 0, -1, or -2") from exc
        if self.median_guard_priority not in (0, -1, -2):
            raise ValueError(
                "TEMPO_PD_MEDIAN_GUARD_PRIORITY must be 0, -1, or -2")
        raw_medium_catchup = os.environ.get(
            "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY", "0")
        try:
            self.medium_remote_catchup_priority = int(raw_medium_catchup)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY must be 0, -1, or -2"
            ) from exc
        if self.medium_remote_catchup_priority not in (0, -1, -2):
            raise ValueError(
                "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY must be 0, -1, or -2")
        raw_min_output = os.environ.get(
            "TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS", "256")
        try:
            self.remote_catchup_min_output_tokens = int(raw_min_output)
        except ValueError as exc:
            raise ValueError(
                "TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS must be 16, 128, or 256"
            ) from exc
        if self.remote_catchup_min_output_tokens not in (16, 128, 256):
            raise ValueError(
                "TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS must be 16, 128, or 256")
        scheduling_policy = os.environ.get(
            "TEMPO_VLLM_SCHEDULING_POLICY", "fcfs")
        priority_enabled = (
            self.remote_catchup_priority
            or self.strong_remote_catchup_priority
            or self.long_remote_catchup_priority or self.median_guard_priority
            or self.medium_remote_catchup_priority)
        if priority_enabled and scheduling_policy != "priority":
            raise ValueError(
                "TEMPO request priorities require vLLM priority scheduling")

        self._request_source_cached_tokens = {}
        self._request_decoder_cache_parsers = {}
        self._request_decoder_cache_evidence = {}
        def resolve(request_id):
            if cache_residency is not None:
                value = cache_residency(request_id)
                if not isinstance(value, CacheResidency):
                    raise TypeError("cache residency resolver must return CacheResidency")
            else:
                namespace = self._request_cache_namespaces.get(request_id)
                value = (
                    self.cache_catalog.classify(namespace)
                    if namespace is not None else CacheResidency.UNKNOWN)
            return enforce_explicit_cache_contract(request_id, value)

        super().__init__(config, profile, cache_residency=resolve,
                         allow_screen_profile=allow_screen_profile)
        self.elastic = ElasticPDController(profile.controller)
        self.pressure_mode = os.environ.get(
            PRESSURE_MODE_ENV, PRESSURE_DISABLED_MODE)
        if self.pressure_mode not in (
            PRESSURE_DISABLED_MODE, PRESSURE_OBSERVE_MODE,
            PRESSURE_ADAPTIVE_MODE,
        ):
            raise ValueError(
                f"{PRESSURE_MODE_ENV} must be disabled, observe_only, or adaptive")
        if (
            self.pressure_mode == PRESSURE_ADAPTIVE_MODE
            and not config.remote_backend.endswith("-libfabric-cxi")
        ):
            raise ValueError("adaptive pressure mode requires LIBFABRIC/CXI")
        self.remote_pressure_ms_per_prompt_token = _coefficient(
            REMOTE_PRESSURE_MS_PER_PROMPT_TOKEN_ENV, 0.07)
        self.local_pressure_ms_per_prompt_token = _coefficient(
            LOCAL_PRESSURE_MS_PER_PROMPT_TOKEN_ENV, 0.04)
        raw_max_seqs = os.environ.get("TEMPO_VLLM_MAX_NUM_SEQS", "8")
        try:
            self.pressure_max_num_seqs = int(raw_max_seqs)
        except ValueError as exc:
            raise ValueError("TEMPO_VLLM_MAX_NUM_SEQS must be 8 or 16") from exc
        if self.pressure_max_num_seqs not in (8, 16):
            raise ValueError("TEMPO_VLLM_MAX_NUM_SEQS must be 8 or 16")
        self._fabric_pressure_ewma = None
        self._fabric_congested = False
        self._fabric_sequence = None
        self.endpoint_feedback_mode = os.environ.get(
            ENDPOINT_FEEDBACK_MODE_ENV, ENDPOINT_FEEDBACK_DISABLED_MODE)
        if self.endpoint_feedback_mode not in (
            ENDPOINT_FEEDBACK_DISABLED_MODE,
            ENDPOINT_FEEDBACK_ADAPTIVE_MODE,
        ):
            raise ValueError(
                f"{ENDPOINT_FEEDBACK_MODE_ENV} must be disabled or adaptive"
            )
        raw_passive_feedback = os.environ.get(
            ENDPOINT_PASSIVE_FEEDBACK_ENV, "0")
        if raw_passive_feedback not in ("0", "1"):
            raise ValueError(
                f"{ENDPOINT_PASSIVE_FEEDBACK_ENV} must be 0 or 1")
        self.endpoint_passive_feedback = raw_passive_feedback == "1"
        if (
            self.endpoint_passive_feedback
            and self.endpoint_feedback_mode != ENDPOINT_FEEDBACK_ADAPTIVE_MODE
        ):
            raise ValueError(
                "passive endpoint feedback requires adaptive endpoint mode")
        self.endpoint_routing_policy = os.environ.get(
            ENDPOINT_ROUTING_POLICY_ENV, ENDPOINT_INSTANT_SCORE_POLICY)
        if self.endpoint_routing_policy not in {
            ENDPOINT_INSTANT_SCORE_POLICY,
            ENDPOINT_SEMANTIC_EPOCH_POLICY,
        }:
            raise ValueError(
                f"{ENDPOINT_ROUTING_POLICY_ENV} must be instant_score_v1 or "
                "semantic_epoch_v1"
            )
        if (
            self.endpoint_routing_policy == ENDPOINT_SEMANTIC_EPOCH_POLICY
            and (
                self.endpoint_feedback_mode != ENDPOINT_FEEDBACK_ADAPTIVE_MODE
                or not self.endpoint_passive_feedback
                or self.local_decoder_index not in (0, 1)
            )
        ):
            raise ValueError(
                "semantic_epoch_v1 requires adaptive endpoint feedback, "
                "passive feedback, and a local decoder index"
            )
        if (
            self.endpoint_feedback_mode == ENDPOINT_FEEDBACK_ADAPTIVE_MODE
            and self.pressure_mode != PRESSURE_DISABLED_MODE
        ):
            raise ValueError(
                "endpoint feedback forbids the scalar Cassini pressure policy"
            )
        self._cassini_pressure = (
            CassiniPressureSampler()
            if self.pressure_mode != PRESSURE_DISABLED_MODE else None
        )
        self.endpoint_service_profile = None
        self.endpoint_feedback = None
        self.semantic_epoch_policy = None
        self._endpoint_controller_generation = 0
        if self.endpoint_feedback_mode == ENDPOINT_FEEDBACK_ADAPTIVE_MODE:
            if self.vllm_load_snapshot_mode != VLLM_LOAD_DISABLED_MODE:
                raise ValueError(
                    "endpoint feedback forbids synchronous request-start /metrics"
                )
            if priority_enabled:
                raise ValueError(
                    "endpoint feedback forbids shape-specific priority exceptions"
                )
            raw_profile = os.environ.get(ENDPOINT_SERVICE_PROFILE_ENV)
            if not raw_profile:
                raise ValueError(
                    f"{ENDPOINT_SERVICE_PROFILE_ENV} is required in adaptive mode"
                )
            endpoint_profile = load_endpoint_service_profile(
                Path(raw_profile).resolve()
            )
            if (
                endpoint_profile.elastic_profile_fingerprint_sha256
                != self.profile.fingerprint_sha256
            ):
                raise ValueError(
                    "endpoint and Elastic-PD profile fingerprints differ"
                )
            workload_sha = os.environ.get(
                ENDPOINT_WORKLOAD_MANIFEST_SHA256_ENV)
            if workload_sha != endpoint_profile.workload_manifest_sha256:
                raise ValueError(
                    "endpoint service profile workload binding differs"
                )
            if (
                endpoint_profile.deployment_scope == "calibration_only"
                and not allow_screen_profile
            ):
                raise ValueError(
                    "calibration-only endpoint profile requires explicit opt-in"
                )
            if self.endpoint_routing_policy == ENDPOINT_SEMANTIC_EPOCH_POLICY:
                if (
                    endpoint_profile.schema != ENDPOINT_PROFILE_V2_SCHEMA
                    or endpoint_profile.routing_policy is None
                    or endpoint_profile.routing_policy.policy
                    != PROFILE_SEMANTIC_EPOCH_POLICY
                ):
                    raise ValueError(
                        "semantic_epoch_v1 requires a profile-bound v2 "
                        "routing policy"
                    )
                self.semantic_epoch_policy = endpoint_profile.routing_policy
            elif endpoint_profile.routing_policy is not None:
                raise ValueError(
                    "profile-bound semantic routing policy requires "
                    "semantic_epoch_v1"
                )
            self.endpoint_service_profile = endpoint_profile
            self.endpoint_feedback = EndpointFeedbackController(
                endpoint_profile.controller)

    def prepare_prompt_namespace(self, request_id, prompt_key):
        if not isinstance(prompt_key, str) or not prompt_key:
            raise ValueError("prompt_key must be nonempty")
        with self._lock:
            self._request_prompt_keys[request_id] = prompt_key

    def prepare_prompt_tokens(self, request_id, token_ids):
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(type(value) is not int for value in token_ids)
        ):
            raise ValueError("prompt token IDs must be a nonempty integer list")
        with self._lock:
            prior = self._request_token_ids.get(request_id)
            if prior is not None and prior != token_ids:
                raise ValueError("prompt token IDs changed")
            self._request_token_ids[request_id] = list(token_ids)

    @staticmethod
    def _parse_frontend_nonnegative_int(name, raw_value):
        if (
            not isinstance(raw_value, str)
            or not raw_value
            or not raw_value.isascii()
            or not raw_value.isdecimal()
        ):
            raise ValueError(f"{name} must be a canonical non-negative integer")
        value = int(raw_value)
        if str(value) != raw_value:
            raise ValueError(f"{name} must be a canonical non-negative integer")
        return value

    def prepare_frontend_semantic_load(
        self, *, request_id, pair_index, decode_tokens_before,
        active_requests_before, max_num_seqs,
    ):
        """Record the frontend's pair-local full-stream pressure evidence.

        The frontend owns this ledger from request admission through HTTP EOF.
        Missing headers are permitted for direct pair-router maintenance traffic;
        partial or inconsistent evidence fails closed.
        """
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be nonempty")
        raw = {
            "pair_index": pair_index,
            "decode_tokens_before": decode_tokens_before,
            "active_requests_before": active_requests_before,
            "max_num_seqs": max_num_seqs,
        }
        present = {name: value is not None for name, value in raw.items()}
        if not any(present.values()):
            return None
        if not all(present.values()):
            raise ValueError("frontend semantic-load headers are incomplete")
        parsed = {
            name: self._parse_frontend_nonnegative_int(name, value)
            for name, value in raw.items()
        }
        if parsed["pair_index"] not in (0, 1):
            raise ValueError("frontend semantic-load pair index must be 0 or 1")
        if parsed["max_num_seqs"] not in (8, 16):
            raise ValueError("frontend semantic-load max_num_seqs must be 8 or 16")
        if parsed["max_num_seqs"] != self.pressure_max_num_seqs:
            raise ValueError("frontend/router max_num_seqs mismatch")
        if (
            self.local_decoder_index is not None
            and parsed["pair_index"] != self.local_decoder_index
        ):
            raise ValueError("frontend semantic-load pair/router mismatch")
        evidence = {
            "schema": FRONTEND_SEMANTIC_LOAD_SCHEMA,
            "source": "frontend_pair_ledger_request_start_to_http_eof",
            **parsed,
            "occupancy_ratio_before": (
                parsed["active_requests_before"]
                / parsed["max_num_seqs"]
            ),
        }
        with self._lock:
            if request_id in self._request_frontend_semantic_loads:
                raise ValueError("frontend semantic-load evidence recorded twice")
            self._request_frontend_semantic_loads[request_id] = evidence
        return dict(evidence)

    async def prepare_vllm_load_snapshot(self, request_id, local_client):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be nonempty")
        if self.vllm_load_snapshot_mode == VLLM_LOAD_DISABLED_MODE:
            snapshot = {
                "schema": VLLM_LOAD_SNAPSHOT_SCHEMA,
                "source": VLLM_LOAD_DISABLED_SOURCE,
                "decision_mode": VLLM_LOAD_DISABLED_MODE,
                "endpoint": None,
                "model_name": self._served_model_name,
                "engine_indices": [],
                "sampled_ns": time.perf_counter_ns(),
                "fetch_ms": 0.0,
                "num_requests_running": None,
                "num_requests_waiting": None,
                "kv_cache_usage_perc": None,
            }
        else:
            started_ns = time.perf_counter_ns()
            response = await local_client.get("/metrics")
            response.raise_for_status()
            snapshot = parse_vllm_load_metrics(
                response.text, served_model_name=self._served_model_name)
            sampled_ns = time.perf_counter_ns()
            snapshot.update({
                "endpoint": "/metrics",
                "sampled_ns": sampled_ns,
                "fetch_ms": (sampled_ns - started_ns) / 1_000_000,
            })
        with self._lock:
            if request_id in self._request_vllm_load_snapshots:
                raise ValueError("vLLM load snapshot recorded twice")
            self._request_vllm_load_snapshots[request_id] = snapshot
        self._prepare_pressure_snapshot(request_id)
        return dict(snapshot)

    def _active_decode_pressure(self) -> dict[str, Any]:
        with self._lock:
            active = [
                record for record in self._records.values()
                if record.phase in {"started", "response_started"}
            ]
        active_requests = len(active)
        active_output_tokens = sum(record.output_tokens for record in active)
        active_local_prefill_tokens = sum(
            record.prompt_tokens for record in active
            if record.route is ElasticRoute.LOCAL)
        active_remote_kv_bytes = sum(
            record.potential_kv_bytes for record in active
            if record.route is ElasticRoute.REMOTE)
        request_pressure = min(
            1.0, active_requests / self.pressure_max_num_seqs)
        token_pressure = min(
            1.0,
            active_output_tokens / (self.pressure_max_num_seqs * 256),
        )
        return {
            "active_requests": active_requests,
            "active_output_tokens": active_output_tokens,
            "active_local_prefill_tokens": active_local_prefill_tokens,
            "active_remote_kv_bytes": active_remote_kv_bytes,
            "local_pressure": max(request_pressure, token_pressure),
        }

    def _prepare_pressure_snapshot(self, request_id: str) -> dict[str, Any]:
        active = self._active_decode_pressure()
        if self.pressure_mode == PRESSURE_DISABLED_MODE:
            cassini = None
            raw_fabric_pressure = 0.0
            effective_fabric_pressure = 0.0
            congested = False
        else:
            if self._cassini_pressure is None:
                raise RuntimeError("Cassini pressure sampler is missing")
            cassini = self._cassini_pressure.sample()
            if cassini["valid"]:
                pause = max(
                    cassini["rx_pause_fraction_max"],
                    cassini["tx_pause_fraction_max"],
                )
                raw_fabric_pressure = max(
                    _unit_interval(
                        pause, FABRIC_PAUSE_FLOOR, FABRIC_PAUSE_CEILING),
                    _unit_interval(
                        cassini["host_blocked_cycles_per_packet_max"],
                        HOST_BLOCKED_FLOOR, HOST_BLOCKED_CEILING),
                )
                with self._lock:
                    if cassini["sequence"] != self._fabric_sequence:
                        prior = self._fabric_pressure_ewma
                        self._fabric_pressure_ewma = (
                            raw_fabric_pressure if prior is None
                            else FABRIC_EWMA_ALPHA * raw_fabric_pressure
                            + (1.0 - FABRIC_EWMA_ALPHA) * prior
                        )
                        self._fabric_sequence = cassini["sequence"]
                        if (
                            not self._fabric_congested
                            and self._fabric_pressure_ewma
                            >= FABRIC_CONGESTION_ENTER
                        ):
                            self._fabric_congested = True
                        elif (
                            self._fabric_congested
                            and self._fabric_pressure_ewma
                            <= FABRIC_CONGESTION_EXIT
                        ):
                            self._fabric_congested = False
                    ewma = self._fabric_pressure_ewma
                    congested = self._fabric_congested
                effective_fabric_pressure = float(ewma or 0.0)
                if congested:
                    effective_fabric_pressure = max(
                        effective_fabric_pressure, FABRIC_CONGESTION_ENTER)
            else:
                raw_fabric_pressure = 0.0
                with self._lock:
                    effective_fabric_pressure = float(
                        self._fabric_pressure_ewma or 0.0)
                    congested = self._fabric_congested
        local_per_token, remote_per_token = pressure_penalties_ms(
            prompt_tokens=1,
            local_pressure=active["local_pressure"],
            fabric_pressure=effective_fabric_pressure,
            local_ms_per_prompt_token=self.local_pressure_ms_per_prompt_token,
            remote_ms_per_prompt_token=self.remote_pressure_ms_per_prompt_token,
        )
        snapshot = {
            "schema": PRESSURE_SCHEMA,
            "mode": self.pressure_mode,
            "sampled_ns": time.perf_counter_ns(),
            "cassini": dict(cassini) if cassini is not None else None,
            "fabric_pressure_raw": raw_fabric_pressure,
            "fabric_pressure_ewma": effective_fabric_pressure,
            "fabric_congested": congested,
            **active,
            "local_penalty_ms_per_prompt_token": local_per_token,
            "remote_penalty_ms_per_prompt_token": remote_per_token,
        }
        with self._lock:
            if request_id in self._request_pressure_snapshots:
                raise ValueError("pressure snapshot recorded twice")
            self._request_pressure_snapshots[request_id] = snapshot
        return dict(snapshot)

    def vllm_load_snapshot(self, request_id):
        with self._lock:
            snapshot = self._request_vllm_load_snapshots.get(request_id)
        if snapshot is None:
            raise KeyError(request_id)
        return dict(snapshot)

    def _remember_decision_cache_residency(self, record):
        base._require(
            isinstance(record.cache_residency, CacheResidency),
            "decision cache residency is invalid",
        )
        value = record.cache_residency.value
        with self._lock:
            prior = self._request_decision_cache_residencies.get(
                record.request_id)
            if prior is not None and prior != value:
                raise ValueError("decision cache residency changed")
            self._request_decision_cache_residencies[
                record.request_id] = value
        return record

    def _decide_fixed_profile_independent(
        self, *, request_id, prompt_tokens, output_tokens,
    ):
        """Commit fixed controls without requiring a predictor profile row.

        Contention calibration must be able to run short-output and other
        diagnostic geometries that are intentionally absent from the frozen
        predictor profile.  Only fixed arms use this path; predictor and
        TEMPO requests retain exact-profile fail-closed behavior.
        """

        experiment_arm = self.arm(request_id)
        if experiment_arm is ElasticExperimentArm.ALWAYS_LOCAL:
            route = ElasticRoute.LOCAL
            reason = "fixed_always_local"
        elif experiment_arm is ElasticExperimentArm.OFFICIAL_LMCACHE_REMOTE:
            route = ElasticRoute.REMOTE
            reason = "fixed_official_lmcache_remote"
        else:
            raise ValueError("profile-independent decision requires a fixed arm")
        _, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        residency = self._cache_residency(request_id)
        if not isinstance(residency, CacheResidency):
            raise TypeError("cache residency resolver must return CacheResidency")
        now_ns = time.perf_counter_ns()
        with self._lock:
            base._require(
                request_id not in self._records, "duplicate request_id")
            base._require(
                len(self._records) < self.config.decision_capacity,
                "decision capacity exhausted",
            )
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
            decision=None,
        )
        passive = None
        if (
            self.endpoint_passive_feedback
            and ENDPOINT_PASSIVE_MARKER in request_id
        ):
            if self.endpoint_service_profile is None or self.endpoint_feedback is None:
                raise RuntimeError(
                    "passive endpoint feedback controller is unavailable")
            endpoint_route = (
                EndpointRoute.LOCAL
                if route is ElasticRoute.LOCAL else EndpointRoute.REMOTE)
            service_proxy = self.endpoint_service_profile.external_credit_proxy(
                prompt_tokens,
                output_tokens,
                residency,
                route=endpoint_route,
                cold_unknown_as_miss=self.cold_measured,
            )
            service = service_proxy.row
            external_credit = (
                self.endpoint_routing_policy
                == ENDPOINT_SEMANTIC_EPOCH_POLICY)
            passive = {
                "route": endpoint_route,
                "prior_ttft_ms": service.local_ttft_prior_ms
                if endpoint_route is EndpointRoute.LOCAL
                else service.remote_ttft_prior_ms,
                "cache_residency": residency,
                "external_credit": external_credit,
                "service_lookup_mode": service_proxy.lookup_mode,
                "service_source_prompt_tokens": service.prompt_tokens,
                "service_source_output_tokens": service.output_tokens,
                "service_source_cache_residency": service.cache_residency,
            }
            if external_credit:
                self.endpoint_feedback.observe_external_start(
                    request_id,
                    route=endpoint_route,
                    work=EndpointWork(
                        local_token_ms=service.local_token_ms,
                        remote_prefill_token_ms=(
                            service.remote_prefill_token_ms),
                        remote_kv_bytes=kv_bytes,
                        remote_semantic_ops=1,
                    ),
                    prior_ttft_ms=float(passive["prior_ttft_ms"]),
                    e2e_deadline_ms=float(
                        self.endpoint_service_profile.default_e2e_deadline_ms),
                    now_ns=now_ns,
                )
        with self._lock:
            base._require(
                request_id not in self._records, "duplicate request_id")
            self._records[request_id] = record
            if passive is not None:
                base._require(
                    request_id not in self._passive_endpoint_requests,
                    "passive endpoint request registered twice",
                )
                self._passive_endpoint_requests[request_id] = passive
        return record

    def _decide_cache_aware_predictor(
        self, *, request_id, prompt_tokens, output_tokens,
        remaining_deadline_ms,
    ):
        """Apply the frozen static predictor without violating decoder KV.

        The calibrated predictor is intentionally non-adaptive.  It compares
        only the two frozen C0 route bounds, except that D_ONLY/BOTH cannot be
        sent to the remote decoder because that would abandon confirmed local
        decoder residency.  This is the same baseline used by offline replay.
        """

        row = self.profile.exact_row(prompt_tokens, output_tokens)
        if row is None:
            raise ValueError("no exact elastic profile row")
        _, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        if row.remote_kv_bytes != kv_bytes:
            raise ValueError("profile/router KV geometry mismatch")
        residency = self._cache_residency(request_id)
        if not isinstance(residency, CacheResidency):
            raise TypeError("cache residency resolver must return CacheResidency")
        estimate = self._estimate(row, remaining_deadline_ms)
        if residency in {CacheResidency.D_ONLY, CacheResidency.BOTH}:
            local_score = (
                estimate.local_upper_bound_ms + estimate.uncertainty_ms)
            if (
                estimate.local_tbt_safe
                and local_score <= estimate.remaining_deadline_ms
            ):
                route = ElasticRoute.LOCAL
                reason = "predictor_decoder_residency_local"
            else:
                route = ElasticRoute.QUEUE
                reason = "predictor_decoder_residency_no_safe_local"
        else:
            route, reason = self._predictor_route(row, estimate)
        now_ns = time.perf_counter_ns()
        with self._lock:
            base._require(request_id not in self._records,
                          "duplicate request_id")
            base._require(
                len(self._records) < self.config.decision_capacity,
                "decision capacity exhausted",
            )
        record = self._record(
            request_id=request_id,
            arm=ElasticExperimentArm.PREDICTOR,
            route=route,
            reason=reason,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            kv_bytes=kv_bytes,
            residency=residency,
            now_ns=now_ns,
            decision=None,
        )
        with self._lock:
            base._require(request_id not in self._records,
                          "duplicate request_id")
            self._records[request_id] = record
            self._rows[request_id] = row
        return record

    @staticmethod
    def _endpoint_decision_dict(decision):
        return {
            "schema": decision.schema,
            "request_id": decision.request_id,
            "route": decision.route.value,
            "reason": decision.reason,
            "decided_ns": decision.decided_ns,
            "local_score_ms": decision.local_score_ms,
            "remote_score_ms": decision.remote_score_ms,
            "local_multiplier": decision.local_multiplier,
            "remote_multiplier": decision.remote_multiplier,
            "local_state": decision.local_state.value,
            "remote_state": decision.remote_state.value,
            "probe": decision.probe,
            "resource_used_before": dict(decision.resource_used_before),
        }

    def endpoint_controller_state(self):
        profile = self.endpoint_service_profile
        controller = self.endpoint_feedback
        if profile is None or controller is None:
            return {
                "schema": ROUTER_SCHEMA,
                "endpoint_feedback_mode": self.endpoint_feedback_mode,
                "endpoint_passive_feedback": self.endpoint_passive_feedback,
                "endpoint_service_profile": None,
                "controller": None,
                "controller_generation": self._endpoint_controller_generation,
                "queued_requests": 0,
                "passive_registered_requests": 0,
            }
        with self._lock:
            queued_requests = sum(
                record.route is ElasticRoute.QUEUE
                and record.phase not in {
                    ElasticPhase.COMPLETE.value,
                    ElasticPhase.FAILED.value,
                }
                for record in self._records.values()
                if record.request_id in self._endpoint_requests
            )
            passive_registered = len(self._passive_endpoint_requests)
        return {
            "schema": ROUTER_SCHEMA,
            "endpoint_feedback_mode": self.endpoint_feedback_mode,
            "endpoint_routing_policy": self.endpoint_routing_policy,
            "endpoint_passive_feedback": self.endpoint_passive_feedback,
            "endpoint_service_profile": {
                "schema": profile.schema,
                "profile_id": profile.profile_id,
                "fingerprint_sha256": profile.fingerprint_sha256,
                "elastic_profile_fingerprint_sha256": (
                    profile.elastic_profile_fingerprint_sha256
                ),
                "workload_manifest_sha256": (
                    profile.workload_manifest_sha256
                ),
                "deployment_scope": profile.deployment_scope,
                "routing_policy": (
                    profile.routing_policy.as_dict()
                    if profile.routing_policy is not None else None
                ),
            },
            "controller": controller.snapshot(
                now_ns=time.perf_counter_ns()),
            "controller_generation": self._endpoint_controller_generation,
            "queued_requests": queued_requests,
            "passive_registered_requests": passive_registered,
        }

    def reset_endpoint_controller(self):
        profile = self.endpoint_service_profile
        controller = self.endpoint_feedback
        if profile is None or controller is None:
            raise ValueError("endpoint feedback controller is unavailable")
        with self._lock:
            snapshot = controller.snapshot(now_ns=time.perf_counter_ns())
            queued_requests = sum(
                record.route is ElasticRoute.QUEUE
                and record.phase not in {
                    ElasticPhase.COMPLETE.value,
                    ElasticPhase.FAILED.value,
                }
                for record in self._records.values()
                if record.request_id in self._endpoint_requests
            )
            if (
                snapshot["inflight"] != 0
                or snapshot.get("external_inflight", 0) != 0
                or queued_requests != 0
                or any(snapshot["resources"].values())
            ):
                raise ValueError("endpoint controller reset is not quiescent")
            self.endpoint_feedback = EndpointFeedbackController(
                profile.controller)
            self._endpoint_controller_generation += 1
            self._semantic_epoch_route = EndpointRoute.LOCAL
            self._semantic_epoch_generation = 0
            self._semantic_high_streak = 0
            self._semantic_low_streak = 0
            generation = self._endpoint_controller_generation
        return {
            "schema": ROUTER_SCHEMA,
            "success": True,
            "controller_generation": generation,
            "profile_id": profile.profile_id,
            "profile_fingerprint_sha256": profile.fingerprint_sha256,
            "controller": self.endpoint_feedback.snapshot(
                now_ns=time.perf_counter_ns()),
        }

    def _endpoint_record(
        self, *, request, decision, prompt_tokens, output_tokens,
        kv_bytes, residency,
    ):
        route = {
            EndpointRoute.LOCAL: ElasticRoute.LOCAL,
            EndpointRoute.REMOTE: ElasticRoute.REMOTE,
            EndpointRoute.QUEUE: ElasticRoute.QUEUE,
        }[decision.route]
        record = self._record(
            request_id=request.request_id,
            arm=ElasticExperimentArm.TEMPO,
            route=route,
            reason=decision.reason,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            kv_bytes=kv_bytes,
            residency=residency,
            now_ns=decision.decided_ns,
            decision=None,
        )
        return replace(
            record,
            phase=(
                ElasticPhase.QUEUED.value
                if route is ElasticRoute.QUEUE
                else "route_committed"
            ),
            regime="endpoint_feedback",
            local_score_ms=decision.local_score_ms,
            remote_score_ms=decision.remote_score_ms,
            remote_probe=decision.probe,
        )

    def _semantic_epoch_request(self, request, *, now_ns):
        if self.endpoint_feedback is None:
            raise RuntimeError("semantic epoch lacks endpoint controller")
        policy = self.semantic_epoch_policy
        profile = self.endpoint_service_profile
        if policy is None or profile is None:
            raise RuntimeError("semantic epoch lacks a profile-bound policy")
        with self._lock:
            load = self._request_frontend_semantic_loads.get(
                request.request_id)
        if load is None:
            raise ValueError(
                "semantic_epoch_v1 requires frontend semantic-load evidence")
        snapshot = self.endpoint_feedback.snapshot(now_ns=now_ns)
        external = snapshot.get("external_resources")
        routes = snapshot.get("routes")
        if not isinstance(external, dict) or not isinstance(routes, dict):
            raise RuntimeError("endpoint snapshot lacks semantic resources")
        config = self.endpoint_feedback.config
        local_external_utilization = (
            int(external["local_token_ms"])
            / config.local_token_ms_window
        )
        remote_external_utilization = max(
            int(external["remote_prefill_token_ms"])
            / config.remote_prefill_token_ms_window,
            int(external["remote_kv_bytes"])
            / config.remote_kv_bytes_window,
            int(external["remote_semantic_ops"])
            / config.remote_semantic_ops_window,
        )
        remote = routes.get(EndpointRoute.REMOTE.value)
        if not isinstance(remote, dict):
            raise RuntimeError("endpoint snapshot lacks remote route")
        remote_state = remote.get("state")
        remote_multiplier = float(remote.get("service_multiplier"))
        remote_available = (
            request.remote_allowed
            and remote_state == "good"
            and remote_multiplier <= policy.remote_overload_service_stretch
            and remote_external_utilization
            < policy.remote_external_credit_close_fraction
        )
        active = int(load["active_requests_before"])
        capacity = int(load["max_num_seqs"])
        decoder_high = (
            active * policy.decoder_high_water_denominator
            >= capacity * policy.decoder_high_water_numerator
        )
        decoder_low = (
            active * policy.decoder_low_water_denominator
            < capacity * policy.decoder_low_water_numerator
        )
        credit_epoch = policy.uses_local_external_credit_epoch
        local_external_credit_pressure = local_external_utilization > 0.0
        route_high = (
            local_external_credit_pressure if credit_epoch else decoder_high)
        route_low = (
            not local_external_credit_pressure if credit_epoch else decoder_low)
        if credit_epoch:
            reason_names = {
                "local_default": "semantic_credit_epoch_local_default",
                "close_unavailable": (
                    "semantic_credit_epoch_close_remote_unavailable"),
                "close_low": (
                    "semantic_credit_epoch_close_local_credit_idle"),
                "remote_low_confirmation": (
                    "semantic_credit_epoch_remote_idle_confirmation"),
                "remote_latched": "semantic_credit_epoch_remote_latched",
                "open_high": (
                    "semantic_credit_epoch_open_remote_local_credit"),
                "local_high_confirmation": (
                    "semantic_credit_epoch_local_credit_confirmation"),
                "local_unavailable": (
                    "semantic_credit_epoch_local_remote_unavailable"),
            }
        else:
            reason_names = {
                "local_default": "semantic_epoch_local_default",
                "close_unavailable": (
                    "semantic_epoch_close_remote_unavailable"),
                "close_low": "semantic_epoch_close_decoder_low_water",
                "remote_low_confirmation": (
                    "semantic_epoch_remote_low_water_confirmation"),
                "remote_latched": "semantic_epoch_remote_latched",
                "open_high": "semantic_epoch_open_remote_high_water",
                "local_high_confirmation": (
                    "semantic_epoch_local_high_water_confirmation"),
                "local_unavailable": (
                    "semantic_epoch_local_remote_unavailable"),
            }
        with self._lock:
            route_before = self._semantic_epoch_route
            reason = reason_names["local_default"]
            if self._semantic_epoch_route is EndpointRoute.REMOTE:
                if not remote_available:
                    self._semantic_epoch_route = EndpointRoute.LOCAL
                    self._semantic_epoch_generation += 1
                    self._semantic_high_streak = 0
                    self._semantic_low_streak = 0
                    reason = reason_names["close_unavailable"]
                elif route_low:
                    self._semantic_low_streak += 1
                    if (
                        self._semantic_low_streak
                        >= policy.epoch_confirmation_requests
                    ):
                        self._semantic_epoch_route = EndpointRoute.LOCAL
                        self._semantic_epoch_generation += 1
                        self._semantic_high_streak = 0
                        self._semantic_low_streak = 0
                        reason = reason_names["close_low"]
                    else:
                        reason = reason_names["remote_low_confirmation"]
                else:
                    self._semantic_low_streak = 0
                    reason = reason_names["remote_latched"]
            else:
                self._semantic_low_streak = 0
                if route_high and remote_available:
                    self._semantic_high_streak += 1
                    if (
                        self._semantic_high_streak
                        >= policy.epoch_confirmation_requests
                    ):
                        self._semantic_epoch_route = EndpointRoute.REMOTE
                        self._semantic_epoch_generation += 1
                        self._semantic_high_streak = 0
                        reason = reason_names["open_high"]
                    else:
                        reason = reason_names["local_high_confirmation"]
                else:
                    self._semantic_high_streak = 0
                    if route_high and not remote_available:
                        reason = reason_names["local_unavailable"]
            selected = self._semantic_epoch_route
            if selected is EndpointRoute.REMOTE and not request.remote_allowed:
                raise RuntimeError("semantic epoch selected a forbidden remote route")
            evidence = {
                "schema": SEMANTIC_EPOCH_SCHEMA,
                "policy": ENDPOINT_SEMANTIC_EPOCH_POLICY,
                "profile_fingerprint_sha256": profile.fingerprint_sha256,
                "route_before": route_before.value,
                "route_after": selected.value,
                "reason": reason,
                "generation": self._semantic_epoch_generation,
                "decoder_high_water": decoder_high,
                "decoder_low_water": decoder_low,
                "decision_basis": (
                    "local_external_credit_nonzero"
                    if credit_epoch else "frontend_decoder_watermarks"),
                "local_external_credit_pressure": (
                    local_external_credit_pressure),
                "local_external_credit_opens_epoch": (
                    policy.local_external_credit_opens_epoch),
                "frontend_decoder_watermarks_policy_input": (
                    policy.frontend_decoder_watermarks_policy_input),
                "active_requests_before": active,
                "decode_tokens_before": int(load["decode_tokens_before"]),
                "max_num_seqs": capacity,
                "high_streak_after": self._semantic_high_streak,
                "low_streak_after": self._semantic_low_streak,
                "remote_state": remote_state,
                "remote_multiplier": remote_multiplier,
                "remote_available": remote_available,
                "local_external_utilization": local_external_utilization,
                "remote_external_utilization": remote_external_utilization,
                "decoder_high_water_numerator": (
                    policy.decoder_high_water_numerator),
                "decoder_high_water_denominator": (
                    policy.decoder_high_water_denominator),
                "decoder_low_water_numerator": (
                    policy.decoder_low_water_numerator),
                "decoder_low_water_denominator": (
                    policy.decoder_low_water_denominator),
                "confirmation_requests": policy.epoch_confirmation_requests,
                "overload_multiplier": (
                    policy.remote_overload_service_stretch),
                "remote_external_credit_close_fraction": (
                    policy.remote_external_credit_close_fraction),
            }
            if request.request_id in self._request_semantic_epoch_decisions:
                raise ValueError("semantic epoch decision recorded twice")
            self._request_semantic_epoch_decisions[
                request.request_id] = evidence
        return replace(
            request,
            local_allowed=selected is EndpointRoute.LOCAL,
            remote_allowed=selected is EndpointRoute.REMOTE,
        ), evidence

    def _decide_endpoint_feedback(
        self, *, request_id, prompt_tokens, output_tokens,
        remaining_deadline_ms,
    ):
        if self.endpoint_service_profile is None or self.endpoint_feedback is None:
            raise RuntimeError("endpoint feedback controller is unavailable")
        row = self.profile.exact_row(prompt_tokens, output_tokens)
        if row is None:
            raise ValueError("no exact elastic profile row")
        _, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        if row.remote_kv_bytes != kv_bytes:
            raise ValueError("profile/router KV geometry mismatch")
        residency = self._cache_residency(request_id)
        if not isinstance(residency, CacheResidency):
            raise TypeError("cache residency resolver must return CacheResidency")
        service = self.endpoint_service_profile.exact_row(
            prompt_tokens,
            output_tokens,
            residency,
            cold_unknown_as_miss=self.cold_measured,
        )
        estimate = self._estimate(row, remaining_deadline_ms)
        deadline = estimate.remaining_deadline_ms
        if remaining_deadline_ms is None:
            deadline = float(
                self.endpoint_service_profile.default_e2e_deadline_ms)
        local_allowed = True
        remote_allowed = residency not in {
            CacheResidency.D_ONLY, CacheResidency.BOTH,
        }
        request = EndpointRequest(
            request_id=request_id,
            local_e2e_prior_ms=estimate.local_upper_bound_ms,
            remote_e2e_prior_ms=estimate.remote_upper_bound_ms,
            local_ttft_prior_ms=service.local_ttft_prior_ms,
            remote_ttft_prior_ms=service.remote_ttft_prior_ms,
            uncertainty_ms=estimate.uncertainty_ms,
            e2e_deadline_ms=deadline,
            work=EndpointWork(
                local_token_ms=service.local_token_ms,
                remote_prefill_token_ms=service.remote_prefill_token_ms,
                remote_kv_bytes=kv_bytes,
                remote_semantic_ops=1,
            ),
            local_allowed=local_allowed,
            remote_allowed=remote_allowed,
        )
        now_ns = time.perf_counter_ns()
        semantic_epoch = None
        if self.endpoint_routing_policy == ENDPOINT_SEMANTIC_EPOCH_POLICY:
            request, semantic_epoch = self._semantic_epoch_request(
                request, now_ns=now_ns)
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            base._require(
                len(self._records) < self.config.decision_capacity,
                "decision capacity exhausted",
            )
        decision = self.endpoint_feedback.submit(request, now_ns=now_ns)
        record = self._endpoint_record(
            request=request,
            decision=decision,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            kv_bytes=kv_bytes,
            residency=residency,
        )
        if semantic_epoch is not None:
            record = replace(
                record,
                reason=str(semantic_epoch["reason"]),
                regime=ENDPOINT_SEMANTIC_EPOCH_POLICY,
            )
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            self._records[request_id] = record
            self._rows[request_id] = row
            self._endpoint_requests[request_id] = request
            self._request_endpoint_decisions[request_id] = [
                self._endpoint_decision_dict(decision)
            ]
        return record

    def decide(self, *, request_id, prompt_tokens, output_tokens,
               remaining_deadline_ms=None):
        experiment_arm = self.arm(request_id)
        arm = experiment_arm.value
        with self._lock:
            prompt_key = self._request_prompt_keys.get(request_id, request_id)
            namespace = self.cache_catalog.namespace(
                arm=arm, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens, item=prompt_key)
            self._request_cache_namespaces[request_id] = namespace
        if (
            "-warm-" in request_id
            or D_CACHE_SEED_MARKER in request_id
            or D_CACHE_PROBE_MARKER in request_id
        ):
            record = self._decide_cache_prepare(
                request_id=request_id, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens)
        elif experiment_arm in {
            ElasticExperimentArm.ALWAYS_LOCAL,
            ElasticExperimentArm.OFFICIAL_LMCACHE_REMOTE,
        }:
            record = self._decide_fixed_profile_independent(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        elif experiment_arm is ElasticExperimentArm.PREDICTOR:
            record = self._decide_cache_aware_predictor(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_deadline_ms=remaining_deadline_ms,
            )
        elif (
            experiment_arm is ElasticExperimentArm.TEMPO
            and self.endpoint_feedback_mode == ENDPOINT_FEEDBACK_ADAPTIVE_MODE
        ):
            record = self._decide_endpoint_feedback(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_deadline_ms=remaining_deadline_ms,
            )
        else:
            if experiment_arm is ElasticExperimentArm.TEMPO:
                self.elastic.register_request_geometry(
                    request_id, prompt_tokens, output_tokens)
            record = super().decide(
                request_id=request_id, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_deadline_ms=remaining_deadline_ms)
        return self._remember_decision_cache_residency(record)

    def retry(self, request_id, remaining_deadline_ms):
        with self._lock:
            endpoint_request = self._endpoint_requests.get(request_id)
            current = self._records.get(request_id)
            semantic_epoch = self._request_semantic_epoch_decisions.get(
                request_id)
        if endpoint_request is None:
            return super().retry(request_id, remaining_deadline_ms)
        if current is None or current.route is not ElasticRoute.QUEUE:
            raise ValueError("only queued endpoint requests can retry")
        if self.endpoint_feedback is None:
            raise RuntimeError("endpoint feedback controller is unavailable")
        deadline = float(remaining_deadline_ms)
        if not math.isfinite(deadline) or deadline <= 0.0:
            raise ValueError("remaining endpoint deadline must be finite and positive")
        # The generic wire loop uses a very large finite sentinel when the
        # client supplied no deadline.  Retrying must never relax the frozen
        # profile deadline or an earlier client deadline.
        deadline = min(deadline, endpoint_request.e2e_deadline_ms)
        request = replace(endpoint_request, e2e_deadline_ms=deadline)
        decision = self.endpoint_feedback.submit(
            request, now_ns=time.perf_counter_ns())
        record = self._endpoint_record(
            request=request,
            decision=decision,
            prompt_tokens=current.prompt_tokens,
            output_tokens=current.output_tokens,
            kv_bytes=current.potential_kv_bytes,
            residency=current.cache_residency,
        )
        # Preserve the original ingress decision timestamp for E2E evidence;
        # endpoint service feedback starts at ``started_ns`` instead.
        record = replace(record, decided_ns=current.decided_ns,
                         attempt=current.attempt + 1)
        # A queued semantic-epoch request retains its ingress route contract
        # across retries.  Preserve the matching policy provenance as well;
        # the endpoint controller's retry reason remains available in the
        # append-only endpoint decision history.
        if semantic_epoch is not None:
            record = replace(
                record,
                reason=str(semantic_epoch["reason"]),
                regime=ENDPOINT_SEMANTIC_EPOCH_POLICY,
            )
        with self._lock:
            self._records[request_id] = record
            self._endpoint_requests[request_id] = request
            self._request_endpoint_decisions[request_id].append(
                self._endpoint_decision_dict(decision)
            )
        return record

    def _observe_passive_endpoint_first_response(
        self, request_id, *, now_ns, event,
    ):
        with self._lock:
            passive = self._passive_endpoint_requests.get(request_id)
            record = self._records.get(request_id)
            prior_event = self._request_endpoint_feedback.get(request_id)
        if passive is None:
            return False
        if prior_event is not None:
            raise ValueError("passive endpoint feedback recorded twice")
        if record is None or record.started_ns is None:
            raise ValueError("passive endpoint request has no start evidence")
        if self.endpoint_feedback is None:
            raise RuntimeError("passive endpoint controller is unavailable")
        observed_ttft_ms = (now_ns - record.started_ns) / 1_000_000
        prior_ttft_ms = float(passive["prior_ttft_ms"])
        external_credit = passive.get("external_credit") is True
        if external_credit:
            accepted = self.endpoint_feedback.observe_external_first_response(
                request_id,
                observed_ttft_ms=observed_ttft_ms,
                now_ns=now_ns,
            )
        else:
            accepted = self.endpoint_feedback.observe_passive_first_response(
                request_id,
                route=passive["route"],
                observed_ttft_ms=observed_ttft_ms,
                prior_ttft_ms=prior_ttft_ms,
                now_ns=now_ns,
            )
        controller_snapshot = self.endpoint_feedback.snapshot(now_ns=now_ns)
        with self._lock:
            self._request_endpoint_feedback[request_id] = {
                "event": (
                    f"external_credit_{event}" if external_credit else event),
                "observed_ttft_ms": observed_ttft_ms,
                "prior_ttft_ms": prior_ttft_ms,
                "service_stretch": observed_ttft_ms / prior_ttft_ms,
                "passive": True,
                "external_credit": external_credit,
                "accepted": accepted,
                "released_ns": None,
                "controller": controller_snapshot,
            }
        return True

    def _fail_passive_endpoint(self, request_id, *, now_ns):
        with self._lock:
            passive = self._passive_endpoint_requests.get(request_id)
            prior_event = self._request_endpoint_feedback.get(request_id)
        if passive is None or prior_event is not None:
            return False
        if self.endpoint_feedback is None:
            raise RuntimeError("passive endpoint controller is unavailable")
        external_credit = passive.get("external_credit") is True
        if external_credit:
            self.endpoint_feedback.fail_external(
                request_id, now_ns=now_ns)
        else:
            self.endpoint_feedback.fail_passive(
                request_id, route=passive["route"], now_ns=now_ns)
        controller_snapshot = self.endpoint_feedback.snapshot(now_ns=now_ns)
        with self._lock:
            self._request_endpoint_feedback[request_id] = {
                "event": (
                    "external_credit_upstream_failure"
                    if external_credit else "passive_upstream_failure"),
                "observed_ttft_ms": None,
                "prior_ttft_ms": float(passive["prior_ttft_ms"]),
                "service_stretch": None,
                "passive": True,
                "external_credit": external_credit,
                "accepted": True,
                "released_ns": None,
                "controller": controller_snapshot,
            }
        return True

    def mark_first_response_chunk(self, request_id):
        with self._lock:
            endpoint_request = self._endpoint_requests.get(request_id)
        if endpoint_request is None:
            self._observe_passive_endpoint_first_response(
                request_id,
                now_ns=time.perf_counter_ns(),
                event="passive_first_response_chunk",
            )
            return super().mark_first_response_chunk(request_id)
        if self.endpoint_feedback is None:
            raise RuntimeError("endpoint feedback controller is unavailable")
        now_ns = time.perf_counter_ns()
        with self._lock:
            record = self._get(request_id)
            already_released = request_id in self._admission_released_ns
        if already_released:
            raise ValueError("first response chunk recorded twice")
        if record.started_ns is None:
            raise ValueError("endpoint request has no upstream-start evidence")
        observed_ttft_ms = (now_ns - record.started_ns) / 1_000_000
        endpoint_route = (
            EndpointRoute.LOCAL
            if record.route is ElasticRoute.LOCAL else EndpointRoute.REMOTE)
        prior_ttft_ms = endpoint_request.ttft_prior_ms(endpoint_route)
        accepted = self.endpoint_feedback.observe_first_response(
            request_id,
            observed_ttft_ms=observed_ttft_ms,
            now_ns=now_ns,
        )
        controller_snapshot = self.endpoint_feedback.snapshot(now_ns=now_ns)
        with self._lock:
            self._admission_released_ns[request_id] = now_ns
            self._request_endpoint_feedback[request_id] = {
                "event": "first_response_chunk",
                "observed_ttft_ms": observed_ttft_ms,
                "prior_ttft_ms": prior_ttft_ms,
                "service_stretch": observed_ttft_ms / prior_ttft_ms,
                "passive": False,
                "accepted": accepted,
                "released_ns": now_ns,
                "controller": controller_snapshot,
            }
        self._replace(
            request_id,
            phase="first_response_credit_released",
            response_started_ns=now_ns,
        )

    def complete(self, request_id):
        with self._lock:
            endpoint_request = self._endpoint_requests.get(request_id)
            released = request_id in self._admission_released_ns
            record = self._records.get(request_id)
        if endpoint_request is None:
            with self._lock:
                passive_pending = (
                    request_id in self._passive_endpoint_requests
                    and request_id not in self._request_endpoint_feedback)
            if passive_pending:
                self._observe_passive_endpoint_first_response(
                    request_id,
                    now_ns=time.perf_counter_ns(),
                    event="passive_stream_completion_fallback",
                )
            return super().complete(request_id)
        if record is None:
            raise ValueError("unknown request_id")
        if record.phase in {
            ElasticPhase.COMPLETE.value,
            ElasticPhase.FAILED.value,
        }:
            raise ValueError("endpoint request is already terminal")
        finished_ns = time.perf_counter_ns()
        if not released:
            if record.route is ElasticRoute.QUEUE:
                raise ValueError("queued endpoint request cannot complete")
            if record.started_ns is None or self.endpoint_feedback is None:
                raise ValueError("endpoint completion lacks service ownership")
            observed_ttft_ms = (finished_ns - record.started_ns) / 1_000_000
            endpoint_route = (
                EndpointRoute.LOCAL
                if record.route is ElasticRoute.LOCAL else EndpointRoute.REMOTE)
            prior_ttft_ms = endpoint_request.ttft_prior_ms(endpoint_route)
            accepted = self.endpoint_feedback.observe_first_response(
                request_id,
                observed_ttft_ms=observed_ttft_ms,
                now_ns=finished_ns,
            )
            controller_snapshot = self.endpoint_feedback.snapshot(
                now_ns=finished_ns)
            with self._lock:
                self._admission_released_ns[request_id] = finished_ns
                self._request_endpoint_feedback[request_id] = {
                    "event": "stream_completion_fallback",
                    "observed_ttft_ms": observed_ttft_ms,
                    "prior_ttft_ms": prior_ttft_ms,
                    "service_stretch": observed_ttft_ms / prior_ttft_ms,
                    "passive": False,
                    "accepted": accepted,
                    "released_ns": finished_ns,
                    "controller": controller_snapshot,
                }
        self._replace(
            request_id, phase=ElasticPhase.COMPLETE.value,
            finished_ns=finished_ns)

    def fail(self, request_id, error):
        base._require(bool(error), "error must be nonempty")
        with self._lock:
            endpoint_request = self._endpoint_requests.get(request_id)
            released = request_id in self._admission_released_ns
            record = self._records.get(request_id)
        if endpoint_request is None:
            self._fail_passive_endpoint(
                request_id, now_ns=time.perf_counter_ns())
            return super().fail(request_id, error)
        if record is None:
            raise ValueError("unknown request_id")
        if record.phase in {
            ElasticPhase.COMPLETE.value,
            ElasticPhase.FAILED.value,
        }:
            raise ValueError("endpoint request is already terminal")
        failed_ns = time.perf_counter_ns()
        if (
            not released
            and record.route is not ElasticRoute.QUEUE
            and self.endpoint_feedback is not None
        ):
            self.endpoint_feedback.fail(request_id, now_ns=failed_ns)
            controller_snapshot = self.endpoint_feedback.snapshot(
                now_ns=failed_ns)
            with self._lock:
                self._admission_released_ns[request_id] = failed_ns
                self._request_endpoint_feedback[request_id] = {
                    "event": "upstream_failure",
                    "observed_ttft_ms": None,
                    "prior_ttft_ms": None,
                    "service_stretch": None,
                    "passive": False,
                    "accepted": True,
                    "released_ns": failed_ns,
                    "controller": controller_snapshot,
                }
        elif record.route is ElasticRoute.QUEUE:
            controller_snapshot = (
                self.endpoint_feedback.snapshot(now_ns=failed_ns)
                if self.endpoint_feedback is not None else None
            )
            with self._lock:
                self._request_endpoint_feedback[request_id] = {
                    "event": "queue_failure_no_reservation",
                    "observed_ttft_ms": None,
                    "prior_ttft_ms": None,
                    "service_stretch": None,
                    "passive": False,
                    "accepted": False,
                    "released_ns": None,
                    "controller": controller_snapshot,
                }
        self._replace(
            request_id, phase=ElasticPhase.FAILED.value,
            finished_ns=failed_ns, error=error)

    def _tempo_estimate(self, request_id, row, remaining_deadline_ms):
        estimate = super()._tempo_estimate(
            request_id, row, remaining_deadline_ms)
        with self._lock:
            pressure = self._request_pressure_snapshots.get(request_id)
        if pressure is None:
            if self.pressure_mode != PRESSURE_DISABLED_MODE:
                raise ValueError(
                    "TEMPO decision lacks request-start pressure snapshot")
            pressure = self._prepare_pressure_snapshot(request_id)
        local_penalty, remote_penalty = pressure_penalties_ms(
            prompt_tokens=row.prompt_tokens,
            local_pressure=pressure["local_pressure"],
            fabric_pressure=pressure["fabric_pressure_ewma"],
            local_ms_per_prompt_token=self.local_pressure_ms_per_prompt_token,
            remote_ms_per_prompt_token=self.remote_pressure_ms_per_prompt_token,
        )
        apply_pressure = self.pressure_mode == PRESSURE_ADAPTIVE_MODE
        adjusted = replace(
            estimate,
            local_upper_bound_ms=(
                estimate.local_upper_bound_ms
                + (local_penalty if apply_pressure else 0.0)),
            remote_upper_bound_ms=(
                estimate.remote_upper_bound_ms
                + (remote_penalty if apply_pressure else 0.0)),
        )
        evidence = dict(pressure)
        evidence.update({
            "prompt_tokens": row.prompt_tokens,
            "static_local_upper_bound_ms": estimate.local_upper_bound_ms,
            "static_remote_upper_bound_ms": estimate.remote_upper_bound_ms,
            "local_pressure_penalty_ms": local_penalty,
            "remote_pressure_penalty_ms": remote_penalty,
            "pressure_applied": apply_pressure,
            "adjusted_local_upper_bound_ms": adjusted.local_upper_bound_ms,
            "adjusted_remote_upper_bound_ms": adjusted.remote_upper_bound_ms,
        })
        with self._lock:
            self._request_pressure_snapshots[request_id] = evidence
        return adjusted

    def _skip_local_prefix_cache_read(self, record):
        """Select vLLM's existing request-level local APC read control."""
        contract = explicit_cache_contract(record.request_id)
        if D_CACHE_SEED_MARKER in record.request_id:
            return True
        if D_CACHE_PROBE_MARKER in record.request_id:
            return False
        if "-warm-" in record.request_id:
            # P-side preparation must not be satisfied by a decoder-local hit.
            return True
        if contract in {"p_only", "miss"}:
            return True
        if contract in {"d_only", "both"}:
            return False
        if self.cold_measured and "-measured-" in record.request_id:
            return True
        if "-measured-" in record.request_id:
            return not cache_reuse.reuses_decoder_cache(
                record.request_id, self.decoder_reuse_items)
        return False


    def prepare_upstream_payload(self, record, payload):
        """Give delayed remote long decodes bounded admission catch-up."""
        requested_priority = payload.get("priority", 0)
        base._require(
            type(requested_priority) is int,
            "completion priority must be an integer",
        )
        tempo_measured = (
            record.arm is ElasticExperimentArm.TEMPO
            and "-measured-" in record.request_id
        )
        with self._lock:
            pressure = self._request_pressure_snapshots.get(
                record.request_id)
        fabric_congested = bool(
            pressure
            and pressure.get("fabric_congested") is True
        )
        suppress_remote_priority = (
            tempo_measured
            and self.pressure_mode == PRESSURE_ADAPTIVE_MODE
            and fabric_congested
        )
        remote_candidate = (
            self.remote_catchup_priority < 0
            and tempo_measured
            and record.route is ElasticRoute.REMOTE
            and record.output_tokens >= self.remote_catchup_min_output_tokens
        )
        strong_remote_candidate = (
            self.strong_remote_catchup_priority < 0
            and tempo_measured
            and record.route is ElasticRoute.REMOTE
            and (
                getattr(record, "prompt_tokens", 0) > 2048
                or (
                    getattr(record, "prompt_tokens", 0) <= 512
                    and record.output_tokens >= 128
                )
            )
        )
        long_remote_candidate = (
            self.long_remote_catchup_priority < 0
            and tempo_measured
            and record.route is ElasticRoute.REMOTE
            and record.output_tokens >= 256
            and getattr(record, "prompt_tokens", 0)
            >= self.long_remote_catchup_min_prompt_tokens
        )
        median_guard_eligible = (
            self.median_guard_priority < 0
            and tempo_measured
            and record.output_tokens == 64
        )
        medium_remote_candidate = (
            self.medium_remote_catchup_priority < 0
            and tempo_measured
            and record.route is ElasticRoute.REMOTE
            and record.output_tokens == 128
            and record.prompt_tokens <= 2048
        )
        remote_eligible = remote_candidate and not suppress_remote_priority
        strong_remote_eligible = (
            strong_remote_candidate and not suppress_remote_priority)
        long_remote_eligible = (
            long_remote_candidate and not suppress_remote_priority)
        medium_remote_eligible = (
            medium_remote_candidate and not suppress_remote_priority)
        priorities = [requested_priority]
        if remote_eligible:
            priorities.append(self.remote_catchup_priority)
        if strong_remote_eligible:
            priorities.append(self.strong_remote_catchup_priority)
        if long_remote_eligible:
            priorities.append(self.long_remote_catchup_priority)
        if medium_remote_eligible:
            priorities.append(self.medium_remote_catchup_priority)
        if median_guard_eligible:
            priorities.append(self.median_guard_priority)
        effective_priority = min(priorities)
        priority_class = (
            "strong_remote_catchup"
            if strong_remote_eligible
            and effective_priority == self.strong_remote_catchup_priority
            else "long_remote_catchup"
            if long_remote_eligible
            and effective_priority == self.long_remote_catchup_priority
            else "median_guard"
            if median_guard_eligible
            and effective_priority == self.median_guard_priority
            else "medium_remote_catchup"
            if medium_remote_eligible
            and effective_priority == self.medium_remote_catchup_priority
            else "remote_catchup"
            if remote_eligible
            and effective_priority == self.remote_catchup_priority
            else "request_default"
        )
        prepared = dict(payload)
        cache_salt = None
        skip_local_prefix_read = None
        if self.decoder_prefix_caching:
            with self._lock:
                prompt_key = self._request_prompt_keys.get(record.request_id)
            base._require(
                prompt_key is not None,
                "decoder prefix caching requires an exact prompt key",
            )
            cache_salt = cache_reuse.namespace_cache_salt(
                arm=record.arm.value, prompt_key=prompt_key)
            prepared["cache_salt"] = cache_salt
            skip_local_prefix_read = self._skip_local_prefix_cache_read(record)
            raw_xargs = prepared.get("vllm_xargs")
            base._require(
                raw_xargs is None or isinstance(raw_xargs, dict),
                "vllm_xargs must be an object",
            )
            xargs = dict(raw_xargs or {})
            base._require(
                VLLM_SKIP_LOCAL_PREFIX_READ_XARG not in xargs,
                "client must not set TEMPO's prefix-cache read control",
            )
            base._require(
                PROXY_DECODER_SKIP_LOCAL_PREFIX_READ_FIELD not in prepared,
                "client must not set TEMPO's proxy decoder cache control",
            )
            if record.route is ElasticRoute.REMOTE:
                # The official proxy consumes this before producer prefill and
                # maps it to vllm_xargs only for the downstream decoder call.
                prepared[PROXY_DECODER_SKIP_LOCAL_PREFIX_READ_FIELD] = int(
                    skip_local_prefix_read)
            else:
                xargs[VLLM_SKIP_LOCAL_PREFIX_READ_XARG] = int(
                    skip_local_prefix_read)
                prepared["vllm_xargs"] = xargs
        if effective_priority != 0 or "priority" in prepared:
            prepared["priority"] = effective_priority
        token_ids_forwarded = False
        if self.forward_token_ids:
            with self._lock:
                token_ids = self._request_token_ids.get(record.request_id)
            base._require(
                token_ids is not None, "prompt token IDs are unavailable")
            prepared["prompt"] = list(token_ids)
            token_ids_forwarded = True
        decoder_cache_parser = None
        if self.decoder_prefix_caching:
            base._require(
                prepared.get("stream") is True,
                "decoder cache evidence requires an upstream SSE stream",
            )
            stream_options = prepared.get("stream_options")
            base._require(
                isinstance(stream_options, dict)
                and stream_options.get("include_usage") is True,
                "decoder cache evidence requires final stream usage",
            )
            decoder_cache_parser = VLLMDecoderCacheSSEParser()
        with self._lock:
            if decoder_cache_parser is not None:
                base._require(
                    record.request_id not in self._request_decoder_cache_parsers,
                    "decoder cache evidence parser registered twice",
                )
                self._request_decoder_cache_parsers[
                    record.request_id] = decoder_cache_parser
            self._request_upstream_priorities[record.request_id] = {
                "requested": requested_priority,
                "effective": effective_priority,
                "eligible": remote_eligible,
                "applied": effective_priority != requested_priority,
                "remote_applied": (
                    remote_eligible
                    and effective_priority == self.remote_catchup_priority),
                "strong_remote_eligible": strong_remote_eligible,
                "strong_remote_applied": (
                    priority_class == "strong_remote_catchup"),
                "long_remote_eligible": long_remote_eligible,
                "long_remote_applied": (
                    priority_class == "long_remote_catchup"),
                "medium_remote_eligible": medium_remote_eligible,
                "medium_remote_applied": priority_class == "medium_remote_catchup",
                "median_guard_eligible": median_guard_eligible,
                "median_guard_applied": priority_class == "median_guard",
                "priority_class": priority_class,
                "fabric_congested": fabric_congested,
                "fabric_congestion_suppressed": (
                    suppress_remote_priority
                    and any((remote_candidate, strong_remote_candidate,
                             long_remote_candidate,
                             medium_remote_candidate))),
                "token_ids_forwarded": token_ids_forwarded,
                "cache_salt": cache_salt,
                "skip_local_prefix_read": skip_local_prefix_read,
            }
        return prepared

    def prepare_upstream_headers(self, record, headers):
        prepared = dict(headers)
        if (
            self.remote_decode_placement == "long_decode_cross"
            and record.route is ElasticRoute.REMOTE
        ):
            base._require(
                self.local_decoder_index in (0, 1),
                "local decoder index is unavailable",
            )
            decoder_index = (
                1 - self.local_decoder_index
                if record.output_tokens >= 256
                else self.local_decoder_index
            )
            prepared[DECODER_INDEX_HEADER] = str(decoder_index)
            with self._lock:
                self._request_remote_decoder_indices[
                    record.request_id] = decoder_index
        return prepared


    def _decide_cache_prepare(
        self, *, request_id, prompt_tokens, output_tokens,
    ):
        """Force unmeasured P- or D-residency probes through a fixed path."""
        base._require(
            isinstance(request_id, str) and request_id.strip(),
            "request_id must be nonempty",
        )
        experiment_arm = self.arm(request_id)
        _, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        with self._lock:
            namespace = self._request_cache_namespaces.get(request_id)
            base._require(namespace is not None, "cache namespace missing")
            base._require(
                request_id not in self._records, "duplicate request_id")
            base._require(
                len(self._records) < self.config.decision_capacity,
                "decision capacity exhausted",
            )
        now_ns = time.perf_counter_ns()
        residency = self.cache_catalog.classify(namespace)
        d_seed = D_CACHE_SEED_MARKER in request_id
        d_probe = D_CACHE_PROBE_MARKER in request_id
        base._require(
            not (d_seed and d_probe),
            "cache preparation request has conflicting D-cache markers",
        )
        if d_seed:
            route = ElasticRoute.LOCAL
            reason = "unmeasured_d_cache_seed_local"
        elif d_probe:
            route = ElasticRoute.LOCAL
            reason = "unmeasured_d_cache_hit_probe_local"
        else:
            route = ElasticRoute.REMOTE
            reason = (
                "unmeasured_p_only_seed_remote"
                if "-warm-seed-" in request_id
                else "unmeasured_p_only_hit_probe_remote"
            )
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
            decision=None,
        )
        with self._lock:
            base._require(
                request_id not in self._records, "duplicate request_id")
            self._records[request_id] = record
        return record

    def observe_cache_completion(self, request_id, *, prefill_resident,
                                 decode_resident, actual_kv_bytes=None,
                                 completed_ns=None):
        with self._lock:
            namespace = self._request_cache_namespaces.get(request_id)
        if namespace is None:
            raise ValueError("unknown cache namespace")
        event = self.cache_catalog.record_completion(
            namespace, prefill_resident=prefill_resident,
            decode_resident=decode_resident, actual_kv_bytes=actual_kv_bytes,
            completed_ns=completed_ns)
        self._replace(request_id, cache_residency=event.residency)
        return event

    def observe_backend_stream_chunk(self, request_id, *, route, chunk):
        """Observe upstream bytes without changing the forwarded stream."""
        with self._lock:
            record = self._get(request_id)
            parser = self._request_decoder_cache_parsers.get(request_id)
        if route != record.route.value:
            raise ValueError("streamed backend route differs from committed route")
        if not self.decoder_prefix_caching:
            if parser is not None:
                raise ValueError(
                    "decoder cache parser exists for an ineligible route")
            return
        if parser is None:
            raise ValueError("decoder cache evidence parser is missing")
        parser.feed(chunk)

    def _finish_decoder_cache_evidence(self, request_id, record):
        if not self.decoder_prefix_caching:
            return None
        with self._lock:
            parser = self._request_decoder_cache_parsers.get(request_id)
            prior = self._request_decoder_cache_evidence.get(request_id)
        if parser is None:
            raise ValueError("decoder cache evidence parser is missing")
        if prior is not None:
            raise ValueError("decoder cache evidence was finalized twice")
        remote = record.route is ElasticRoute.REMOTE
        expected_prompt_tokens = record.prompt_tokens + int(remote)
        evidence = parser.finish(
            expected_prompt_tokens=expected_prompt_tokens)
        # D residency is established by the exact local preparation probe on
        # the original P-token prompt.  A remote decoder later sees P+1 after
        # the producer appends its first token, but that extra geometry cannot
        # manufacture another local APC block without contaminating producer
        # state.  Require the exact preparation-proven prefix on both routes.
        expected_local_full = full_prefix_hit_tokens(
            record.prompt_tokens, block_size=DECODER_CACHE_BLOCK_SIZE)
        if remote:
            # The producer computes/transfers the original P-token prompt and
            # appends its first token, so the decoder receives P+1 tokens and
            # must obtain exactly P cached tokens in total.  The source split
            # proves how much came from the preparation-proven decoder APC
            # prefix and how much came from external KV transfer.
            if evidence.cached_tokens != record.prompt_tokens:
                raise ValueError(
                    "remote decoder did not receive an exact full P/D cache hit: "
                    f"cached={evidence.cached_tokens} "
                    f"expected={record.prompt_tokens}"
                )
            if evidence.local_cached_tokens not in (0, expected_local_full):
                raise ValueError(
                    "remote decoder local APC evidence is neither an exact miss "
                    "nor an exact full hit: "
                    f"local={evidence.local_cached_tokens} "
                    f"expected_full={expected_local_full}"
                )
        else:
            if evidence.external_cached_tokens != 0:
                raise ValueError(
                    "local decoder route unexpectedly consumed external KV")
            if (
                evidence.cached_tokens != evidence.local_cached_tokens
                or evidence.local_cached_tokens not in (0, expected_local_full)
            ):
                raise ValueError(
                    "local decoder APC evidence is neither an exact miss nor "
                    "an exact full hit: "
                    f"local={evidence.local_cached_tokens} "
                    f"total={evidence.cached_tokens} "
                    f"expected_full={expected_local_full}"
                )
        with self._lock:
            base._require(
                request_id not in self._request_decoder_cache_evidence,
                "decoder cache evidence stored twice",
            )
            self._request_decoder_cache_evidence[request_id] = evidence
        return evidence

    def observe_backend_completion(self, request_id, *, route, upstream_headers):
        with self._lock:
            record = self._get(request_id)
            namespace = self._request_cache_namespaces.get(request_id)
        if namespace is None:
            raise ValueError("unknown cache namespace")
        if route != record.route.value:
            raise ValueError("completed backend route differs from committed route")
        decoder_evidence = self._finish_decoder_cache_evidence(
            request_id, record)
        if route == "official_lmcache_remote_prefill":
            if self.remote_decode_placement == "long_decode_cross":
                with self._lock:
                    expected_decoder_index = (
                        self._request_remote_decoder_indices.get(request_id))
                if expected_decoder_index is None:
                    raise ValueError(
                        "remote decoder placement commitment is missing")
                try:
                    observed_decoder_index = int(upstream_headers.get(
                        "X-Tempo-LMCache-PD-Decoder-Index", ""))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "remote decoder placement evidence is invalid") from exc
                if observed_decoder_index != expected_decoder_index:
                    raise ValueError(
                        "remote decoder placement evidence mismatch: "
                        f"expected={expected_decoder_index} "
                        f"observed={observed_decoder_index}")
            expected_transfer_evidence = (
                TRANSFER_EVIDENCE_OVERLAPPED
                if self.proxy_kv_control_overlap
                else TRANSFER_EVIDENCE_COMPLETE
            )
            if upstream_headers.get(
                "X-Tempo-LMCache-PD-Transfer"
            ) != expected_transfer_evidence:
                raise ValueError("missing completed LMCache P/D transfer evidence")
            if upstream_headers.get("X-Tempo-LMCache-PD-Request-Id") != request_id:
                raise ValueError("LMCache transfer evidence request ID mismatch")
            try:
                prompt_tokens = int(upstream_headers.get(
                    "X-Tempo-LMCache-PD-Prompt-Tokens", ""))
                cached_tokens = int(upstream_headers.get(
                    "X-Tempo-LMCache-PD-Cached-Tokens", ""))
                actual_kv_bytes = int(upstream_headers.get(
                    "X-Tempo-LMCache-PD-KV-Bytes", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "invalid LMCache transfer/cache geometry evidence") from exc
            expected_transfer_tokens = record.prompt_tokens + 1
            expected_transfer_bytes = (
                record.potential_kv_bytes + self.config.kv_bytes_per_token)
            if (
                prompt_tokens != expected_transfer_tokens
                or actual_kv_bytes != expected_transfer_bytes
            ):
                raise ValueError(
                    "LMCache transfer geometry evidence mismatch: "
                    f"expected_tokens={expected_transfer_tokens} "
                    f"header_tokens={prompt_tokens} "
                    f"expected_bytes={expected_transfer_bytes} "
                    f"header_bytes={actual_kv_bytes}"
                )
            # vLLM intentionally recomputes the final source token on a full
            # external-cache hit. LMCache therefore reports exactly L-1
            # reusable KV tokens for the proxy's L-token source prompt.
            full_cacheable_tokens = prompt_tokens - 1
            if not 0 <= cached_tokens <= full_cacheable_tokens:
                raise ValueError("LMCache cached-token evidence outside prompt")
            with self._lock:
                self._request_source_cached_tokens[request_id] = cached_tokens
            if (
                self.decoder_prefix_caching
                and "-warm-" in request_id
                and (
                    decoder_evidence is None
                    or decoder_evidence.local_cached_tokens != 0
                )
            ):
                raise ValueError(
                    "P-side preparation was contaminated by decoder APC")
            if "-warm-seed-" in request_id:
                if cached_tokens != 0:
                    raise ValueError(
                        "P-only seed was contaminated by a prior source hit")
                return None

            prior_event = self.cache_catalog.event(namespace)
            cache_contract = explicit_cache_contract(request_id)
            if self.decoder_prefix_caching:
                if decoder_evidence is None:
                    raise ValueError(
                        "remote route lacks decoder cache-source evidence")
                expected_local = (
                    full_prefix_hit_tokens(
                        record.prompt_tokens,
                        block_size=DECODER_CACHE_BLOCK_SIZE,
                    )
                    if cache_contract in {"d_only", "both"}
                    else 0
                )
                if (
                    cache_contract is not None
                    and decoder_evidence.local_cached_tokens != expected_local
                ):
                    raise ValueError(
                        "remote decoder APC evidence differs from explicit "
                        "cache state: "
                        f"state={cache_contract} "
                        f"local={decoder_evidence.local_cached_tokens} "
                        f"expected={expected_local}"
                    )
            p_only_measured = cache_contract == "p_only"
            if p_only_measured:
                if (
                    prior_event is None
                    or prior_event.residency is not CacheResidency.P_ONLY
                ):
                    raise ValueError(
                        "P-only measured remote request lacks confirmed "
                        "prefill residency"
                    )
                if cached_tokens != full_cacheable_tokens:
                    raise ValueError(
                        "P-only measured remote request lost its full source hit: "
                        f"cached={cached_tokens} prompt={prompt_tokens}"
                    )
                return self.observe_cache_completion(
                    request_id,
                    prefill_resident=True,
                    decode_resident=False,
                    actual_kv_bytes=actual_kv_bytes,
                )
            if "-warm-" in request_id:
                if cached_tokens != full_cacheable_tokens:
                    raise ValueError(
                        "P-only warm probe did not observe a full source hit: "
                        f"cached={cached_tokens} prompt={prompt_tokens}"
                    )
            else:
                if self.cold_measured or cache_contract == "miss":
                    if prior_event is not None:
                        raise ValueError(
                            "cold measured remote request has prior residency")
                    if cached_tokens != 0:
                        raise ValueError(
                            "cold measured remote request observed a source hit")
                    return self.observe_cache_completion(
                        request_id, prefill_resident=True,
                        decode_resident=False, actual_kv_bytes=actual_kv_bytes)
                if (
                    prior_event is None
                    or prior_event.residency not in {
                        CacheResidency.P_ONLY,
                        CacheResidency.D_ONLY,
                        CacheResidency.BOTH,
                    }
                ):
                    raise ValueError(
                        "measured remote request lacks confirmed cache residency")
                expected_contract_residency = {
                    "d_only": CacheResidency.D_ONLY,
                    "both": CacheResidency.BOTH,
                }.get(cache_contract)
                if (
                    expected_contract_residency is not None
                    and prior_event.residency is not expected_contract_residency
                ):
                    raise ValueError(
                        "measured remote cache residency differs from its "
                        "explicit contract")
                if (
                    record.arm is ElasticExperimentArm.TEMPO
                    and prior_event.decode_resident
                ):
                    raise ValueError(
                        "TEMPO remotely admitted a decoder-resident prompt")
                expected_cached_tokens = (
                    full_cacheable_tokens
                    if prior_event.prefill_resident else 0)
                if cached_tokens != expected_cached_tokens:
                    raise ValueError(
                        "measured remote source-cache evidence differs from "
                        "residency: "
                        f"cached={cached_tokens} "
                        f"expected={expected_cached_tokens}"
                    )
            return self.observe_cache_completion(
                request_id, prefill_resident=True,
                decode_resident=(
                    prior_event.decode_resident
                    if prior_event is not None else False),
                actual_kv_bytes=actual_kv_bytes)

        if route == "decoder_local_chunked_prefill":
            if upstream_headers.get("X-Tempo-LMCache-PD-Transfer") is not None:
                raise ValueError("local route received remote transfer evidence")
            prior_event = self.cache_catalog.event(namespace)
            cache_contract = explicit_cache_contract(request_id)
            p_only_measured = cache_contract == "p_only"
            d_cache_seed = D_CACHE_SEED_MARKER in request_id
            d_cache_probe = D_CACHE_PROBE_MARKER in request_id
            measured_apc = (
                self.decoder_prefix_caching
                and "-measured-" in request_id
                and cache_reuse.reuses_decoder_cache(
                    request_id, self.decoder_reuse_items)
            )
            decoder_cached_tokens = (
                decoder_evidence.local_cached_tokens
                if decoder_evidence is not None else None)
            expected_full_hit = (
                full_prefix_hit_tokens(
                    record.prompt_tokens,
                    block_size=DECODER_CACHE_BLOCK_SIZE,
                )
                if self.decoder_prefix_caching else None)
            decoder_full_hit = (
                expected_full_hit is not None
                and expected_full_hit > 0
                and decoder_cached_tokens == expected_full_hit)
            decoder_miss = decoder_cached_tokens == 0

            if d_cache_seed:
                if decoder_evidence is None:
                    raise ValueError(
                        "D-cache seed requires vLLM decoder cache evidence")
                if prior_event is not None and prior_event.decode_resident:
                    raise ValueError(
                        "D-cache seed already has decoder residency")
                if not decoder_miss:
                    raise ValueError(
                        "D-cache seed was contaminated by a decoder hit")
                if prior_event is not None:
                    self._replace(
                        request_id, cache_residency=prior_event.residency)
                return prior_event

            if d_cache_probe:
                if decoder_evidence is None or not decoder_full_hit:
                    raise ValueError(
                        "D-cache probe did not observe an exact full APC hit")
                return self.observe_cache_completion(
                    request_id,
                    prefill_resident=(
                        prior_event.prefill_resident
                        if prior_event is not None else False),
                    decode_resident=True,
                    actual_kv_bytes=0,
                )

            if p_only_measured:
                if (
                    prior_event is None
                    or prior_event.residency is not CacheResidency.P_ONLY
                ):
                    raise ValueError(
                        "P-only measured local request lacks confirmed "
                        "prefill residency"
                    )
                if decoder_evidence is not None and not decoder_miss:
                    raise ValueError(
                        "P-only measured local request observed a decoder hit")
                self._replace(
                    request_id, cache_residency=CacheResidency.P_ONLY)
                return prior_event
            if cache_contract == "miss":
                if prior_event is not None:
                    raise ValueError(
                        "explicit MISS request has prior cache residency")
                if decoder_evidence is not None and not decoder_miss:
                    raise ValueError(
                        "explicit MISS request observed a decoder hit")
                return self.observe_cache_completion(
                    request_id, prefill_resident=False,
                    decode_resident=False, actual_kv_bytes=0)
            if "-measured-" in request_id:
                if self.cold_measured:
                    if prior_event is not None:
                        raise ValueError(
                            "cold measured local request has prior residency")
                    if decoder_evidence is not None and not decoder_miss:
                        raise ValueError(
                            "cold measured local request observed a decoder hit")
                    return self.observe_cache_completion(
                        request_id, prefill_resident=False,
                        decode_resident=False, actual_kv_bytes=0)
                allowed_residencies = {
                    CacheResidency.P_ONLY,
                    CacheResidency.D_ONLY,
                    CacheResidency.BOTH,
                }
                if (
                    prior_event is None
                    or prior_event.residency not in allowed_residencies
                ):
                    raise ValueError(
                        "measured local request lacks confirmed cache residency")
                expected_contract_residency = {
                    "d_only": CacheResidency.D_ONLY,
                    "both": CacheResidency.BOTH,
                }.get(cache_contract)
                if (
                    expected_contract_residency is not None
                    and prior_event.residency is not expected_contract_residency
                ):
                    raise ValueError(
                        "measured local cache residency differs from its "
                        "explicit contract")
                if prior_event.decode_resident:
                    if decoder_evidence is None or not decoder_full_hit:
                        raise ValueError(
                            "decoder-resident measured request lost its full APC hit")
                    return self.observe_cache_completion(
                        request_id,
                        prefill_resident=prior_event.prefill_resident,
                        decode_resident=True, actual_kv_bytes=0)
                if measured_apc:
                    if decoder_evidence is None or not decoder_full_hit:
                        raise ValueError(
                            "planned decoder reuse lacked an exact full APC hit")
                    return self.observe_cache_completion(
                        request_id, prefill_resident=True,
                        decode_resident=True, actual_kv_bytes=0)
                if decoder_evidence is not None and not decoder_miss:
                    raise ValueError(
                        "non-reuse measured request observed a decoder hit")
                self._replace(
                    request_id, cache_residency=CacheResidency.P_ONLY)
                return prior_event
            if prior_event is not None:
                if (
                    prior_event.decode_resident
                    and (decoder_evidence is None or not decoder_full_hit)
                ):
                    raise ValueError(
                        "decoder-resident local request lost its full APC hit")
                self._replace(
                    request_id, cache_residency=prior_event.residency)
                return prior_event
            if decoder_full_hit:
                return self.observe_cache_completion(
                    request_id, prefill_resident=False,
                    decode_resident=True, actual_kv_bytes=0)
            if decoder_evidence is not None and not decoder_miss:
                raise ValueError("local request has ambiguous decoder cache evidence")
            return self.observe_cache_completion(
                request_id, prefill_resident=False, decode_resident=False,
                actual_kv_bytes=0)
        raise ValueError(f"unsupported backend route: {route}")

    def records(self):
        rows = super().records()
        with self._lock:
            namespaces = dict(self._request_cache_namespaces)
            source_cached_tokens = dict(self._request_source_cached_tokens)
            decoder_cache_evidence = dict(
                self._request_decoder_cache_evidence)
            upstream_priorities = dict(self._request_upstream_priorities)
            remote_decoder_indices = dict(
                self._request_remote_decoder_indices)
            decision_cache_residencies = dict(
                self._request_decision_cache_residencies)
            vllm_load_snapshots = dict(
                self._request_vllm_load_snapshots)
            pressure_snapshots = dict(
                self._request_pressure_snapshots)
            endpoint_decisions = {
                request_id: [dict(value) for value in values]
                for request_id, values
                in self._request_endpoint_decisions.items()
            }
            endpoint_feedback = dict(self._request_endpoint_feedback)
            semantic_epoch_decisions = {
                request_id: dict(value) for request_id, value
                in self._request_semantic_epoch_decisions.items()
            }
            endpoint_requests = dict(self._endpoint_requests)
            passive_endpoint_requests = dict(
                self._passive_endpoint_requests)
            frontend_semantic_loads = {
                request_id: dict(value) for request_id, value
                in self._request_frontend_semantic_loads.items()
            }
        endpoint_profile = self.endpoint_service_profile
        reuse_items = (
            "all" if self.decoder_reuse_items is None
            else sorted(self.decoder_reuse_items)
        )
        for row in rows:
            request_id = row["request_id"]
            namespace = namespaces.get(request_id)
            cached_tokens = source_cached_tokens.get(request_id)
            decoder_evidence = decoder_cache_evidence.get(request_id)
            priority = upstream_priorities.get(request_id)
            decoder_index = remote_decoder_indices.get(
                request_id)
            request_credit = self.elastic.request_credit_evidence(
                request_id)
            load_snapshot = vllm_load_snapshots.get(request_id)
            pressure = pressure_snapshots.get(request_id)
            endpoint_history = endpoint_decisions.get(request_id, [])
            endpoint_decision = (
                endpoint_history[-1] if endpoint_history else None)
            endpoint_event = endpoint_feedback.get(request_id)
            semantic_epoch = semantic_epoch_decisions.get(request_id)
            endpoint_request = endpoint_requests.get(request_id)
            passive_endpoint_request = passive_endpoint_requests.get(
                request_id)
            semantic_load = frontend_semantic_loads.get(request_id)
            endpoint_snapshot = (
                endpoint_event.get("controller")
                if endpoint_event is not None else None
            )
            endpoint_resources_after = (
                endpoint_snapshot.get("resources", {})
                if endpoint_snapshot is not None else {}
            )
            endpoint_owned_resources_after = (
                endpoint_snapshot.get("owned_resources", {})
                if endpoint_snapshot is not None else {}
            )
            endpoint_external_resources_after = (
                endpoint_snapshot.get("external_resources", {})
                if endpoint_snapshot is not None else {}
            )
            endpoint_routes_after = (
                endpoint_snapshot.get("routes", {})
                if endpoint_snapshot is not None else {}
            )
            endpoint_local_after = endpoint_routes_after.get(
                EndpointRoute.LOCAL.value, {})
            endpoint_remote_after = endpoint_routes_after.get(
                EndpointRoute.REMOTE.value, {})
            measured_reuse = (
                self.decoder_prefix_caching
                and "-measured-" in request_id
                and cache_reuse.reuses_decoder_cache(
                    request_id, self.decoder_reuse_items)
            )
            row["decision_cache_residency"] = (
                decision_cache_residencies.get(request_id))
            row["completion_cache_residency"] = row["cache_residency"]
            row["vllm_load_snapshot_schema"] = (
                load_snapshot.get("schema") if load_snapshot else None)
            row["vllm_load_snapshot_source"] = (
                load_snapshot.get("source") if load_snapshot else None)
            row["vllm_load_decision_mode"] = (
                load_snapshot.get("decision_mode") if load_snapshot else None)
            row["vllm_load_endpoint"] = (
                load_snapshot.get("endpoint") if load_snapshot else None)
            row["vllm_load_model_name"] = (
                load_snapshot.get("model_name") if load_snapshot else None)
            row["vllm_load_engine_indices"] = (
                load_snapshot.get("engine_indices") if load_snapshot else None)
            row["vllm_load_sampled_ns"] = (
                load_snapshot.get("sampled_ns") if load_snapshot else None)
            row["vllm_load_fetch_ms"] = (
                load_snapshot.get("fetch_ms") if load_snapshot else None)
            row["vllm_num_requests_running"] = (
                load_snapshot.get("num_requests_running")
                if load_snapshot else None)
            row["vllm_num_requests_waiting"] = (
                load_snapshot.get("num_requests_waiting")
                if load_snapshot else None)
            row["vllm_kv_cache_usage_perc"] = (
                load_snapshot.get("kv_cache_usage_perc")
                if load_snapshot else None)
            row["pressure_schema"] = (
                pressure.get("schema") if pressure else None)
            row["pressure_mode"] = (
                pressure.get("mode") if pressure else None)
            row["pressure_cassini_valid"] = (
                pressure.get("cassini", {}).get("valid")
                if pressure and pressure.get("cassini") else None)
            row["pressure_cassini_read_ms"] = (
                pressure.get("cassini", {}).get("read_ms")
                if pressure and pressure.get("cassini") else None)
            row["pressure_cassini_sequence"] = (
                pressure.get("cassini", {}).get("sequence")
                if pressure and pressure.get("cassini") else None)
            row["pressure_rx_pause_fraction_max"] = (
                pressure.get("cassini", {}).get("rx_pause_fraction_max")
                if pressure and pressure.get("cassini") else None)
            row["pressure_tx_pause_fraction_max"] = (
                pressure.get("cassini", {}).get("tx_pause_fraction_max")
                if pressure and pressure.get("cassini") else None)
            row["pressure_host_blocked_cycles_per_packet_max"] = (
                pressure.get("cassini", {}).get(
                    "host_blocked_cycles_per_packet_max")
                if pressure and pressure.get("cassini") else None)
            for field in (
                "fabric_pressure_raw", "fabric_pressure_ewma",
                "fabric_congested", "active_requests",
                "active_output_tokens", "active_local_prefill_tokens",
                "active_remote_kv_bytes", "local_pressure",
                "static_local_upper_bound_ms",
                "static_remote_upper_bound_ms",
                "local_pressure_penalty_ms",
                "remote_pressure_penalty_ms", "pressure_applied",
                "adjusted_local_upper_bound_ms",
                "adjusted_remote_upper_bound_ms",
            ):
                row[f"pressure_{field}"] = (
                    pressure.get(field) if pressure else None)
            row["endpoint_feedback_mode"] = self.endpoint_feedback_mode
            row["endpoint_routing_policy"] = self.endpoint_routing_policy
            row["frontend_semantic_load_schema"] = (
                semantic_load.get("schema") if semantic_load else None)
            row["frontend_semantic_load_source"] = (
                semantic_load.get("source") if semantic_load else None)
            for field in (
                "pair_index", "decode_tokens_before",
                "active_requests_before", "max_num_seqs",
                "occupancy_ratio_before",
            ):
                row[f"frontend_semantic_{field}"] = (
                    semantic_load.get(field) if semantic_load else None)
            row["endpoint_policy_applied"] = endpoint_request is not None
            row["semantic_epoch_applied"] = semantic_epoch is not None
            for field in (
                "schema", "policy", "route_before", "route_after", "reason",
                "profile_fingerprint_sha256", "generation",
                "decoder_high_water", "decoder_low_water",
                "decision_basis", "local_external_credit_pressure",
                "local_external_credit_opens_epoch",
                "frontend_decoder_watermarks_policy_input",
                "active_requests_before", "decode_tokens_before",
                "max_num_seqs", "high_streak_after", "low_streak_after",
                "remote_state", "remote_multiplier", "remote_available",
                "local_external_utilization",
                "remote_external_utilization",
                "decoder_high_water_numerator",
                "decoder_high_water_denominator",
                "decoder_low_water_numerator",
                "decoder_low_water_denominator", "confirmation_requests",
                "overload_multiplier",
                "remote_external_credit_close_fraction",
            ):
                row[f"semantic_epoch_{field}"] = (
                    semantic_epoch.get(field) if semantic_epoch else None)
            row["endpoint_service_profile_schema"] = (
                endpoint_profile.schema if endpoint_profile else None)
            row["endpoint_service_profile_id"] = (
                endpoint_profile.profile_id if endpoint_profile else None)
            row["endpoint_service_profile_fingerprint_sha256"] = (
                endpoint_profile.fingerprint_sha256
                if endpoint_profile else None)
            row["endpoint_service_profile_elastic_fingerprint_sha256"] = (
                endpoint_profile.elastic_profile_fingerprint_sha256
                if endpoint_profile else None)
            row["endpoint_service_profile_workload_manifest_sha256"] = (
                endpoint_profile.workload_manifest_sha256
                if endpoint_profile else None)
            row["endpoint_service_profile_deployment_scope"] = (
                endpoint_profile.deployment_scope if endpoint_profile else None)
            row["endpoint_default_e2e_deadline_ms"] = (
                endpoint_profile.default_e2e_deadline_ms
                if endpoint_profile else None)
            row["endpoint_decision_attempts"] = len(endpoint_history)
            row["endpoint_decision_history"] = endpoint_history
            for field in (
                "schema", "route", "reason", "decided_ns", "local_score_ms",
                "remote_score_ms", "local_multiplier", "remote_multiplier",
                "local_state", "remote_state", "probe",
            ):
                row[f"endpoint_decision_{field}"] = (
                    endpoint_decision.get(field)
                    if endpoint_decision else None)
            endpoint_resources_before = (
                endpoint_decision.get("resource_used_before", {})
                if endpoint_decision else {})
            for field in (
                "local_token_ms", "remote_prefill_token_ms",
                "remote_kv_bytes", "remote_semantic_ops",
            ):
                row[f"endpoint_resource_{field}_used_before"] = (
                    endpoint_resources_before.get(field)
                    if endpoint_decision else None)
                row[f"endpoint_resource_{field}_used_after_feedback"] = (
                    endpoint_resources_after.get(field)
                    if endpoint_snapshot else None)
                row[f"endpoint_owned_resource_{field}_used_after_feedback"] = (
                    endpoint_owned_resources_after.get(field)
                    if endpoint_snapshot else None)
                row[f"endpoint_external_resource_{field}_used_after_feedback"] = (
                    endpoint_external_resources_after.get(field)
                    if endpoint_snapshot else None)
            if endpoint_request is None:
                row["endpoint_request_local_e2e_prior_ms"] = None
                row["endpoint_request_remote_e2e_prior_ms"] = None
                row["endpoint_request_local_ttft_prior_ms"] = None
                row["endpoint_request_remote_ttft_prior_ms"] = None
                row["endpoint_request_uncertainty_ms"] = None
                row["endpoint_request_e2e_deadline_ms"] = None
                row["endpoint_request_local_allowed"] = None
                row["endpoint_request_remote_allowed"] = None
                row["endpoint_work_local_token_ms"] = None
                row["endpoint_work_remote_prefill_token_ms"] = None
                row["endpoint_work_remote_kv_bytes"] = None
                row["endpoint_work_remote_semantic_ops"] = None
            else:
                row["endpoint_request_local_e2e_prior_ms"] = (
                    endpoint_request.local_e2e_prior_ms)
                row["endpoint_request_remote_e2e_prior_ms"] = (
                    endpoint_request.remote_e2e_prior_ms)
                row["endpoint_request_local_ttft_prior_ms"] = (
                    endpoint_request.local_ttft_prior_ms)
                row["endpoint_request_remote_ttft_prior_ms"] = (
                    endpoint_request.remote_ttft_prior_ms)
                row["endpoint_request_uncertainty_ms"] = (
                    endpoint_request.uncertainty_ms)
                row["endpoint_request_e2e_deadline_ms"] = (
                    endpoint_request.e2e_deadline_ms)
                row["endpoint_request_local_allowed"] = (
                    endpoint_request.local_allowed)
                row["endpoint_request_remote_allowed"] = (
                    endpoint_request.remote_allowed)
                row["endpoint_work_local_token_ms"] = (
                    endpoint_request.work.local_token_ms)
                row["endpoint_work_remote_prefill_token_ms"] = (
                    endpoint_request.work.remote_prefill_token_ms)
                row["endpoint_work_remote_kv_bytes"] = (
                    endpoint_request.work.remote_kv_bytes)
                row["endpoint_work_remote_semantic_ops"] = (
                    endpoint_request.work.remote_semantic_ops)
            row["endpoint_feedback_event"] = (
                endpoint_event.get("event") if endpoint_event else None)
            row["endpoint_feedback_passive"] = (
                endpoint_event.get("passive", False)
                if endpoint_event else False)
            row["endpoint_feedback_passive_accepted"] = (
                endpoint_event.get("accepted")
                if endpoint_event
                and endpoint_event.get("passive") is True else None)
            row["endpoint_feedback_accepted"] = (
                endpoint_event.get("accepted") if endpoint_event else None)
            row["endpoint_feedback_observed_ttft_ms"] = (
                endpoint_event.get("observed_ttft_ms")
                if endpoint_event else None)
            row["endpoint_feedback_prior_ttft_ms"] = (
                endpoint_event.get("prior_ttft_ms")
                if endpoint_event else None)
            row["endpoint_feedback_service_stretch"] = (
                endpoint_event.get("service_stretch")
                if endpoint_event else None)
            row["endpoint_feedback_released_ns"] = (
                endpoint_event.get("released_ns") if endpoint_event else None)
            row["endpoint_feedback_local_state_after"] = (
                endpoint_local_after.get("state")
                if endpoint_snapshot else None)
            row["endpoint_feedback_remote_state_after"] = (
                endpoint_remote_after.get("state")
                if endpoint_snapshot else None)
            row["endpoint_feedback_local_multiplier_after"] = (
                endpoint_local_after.get("service_multiplier")
                if endpoint_snapshot else None)
            row["endpoint_feedback_remote_multiplier_after"] = (
                endpoint_remote_after.get("service_multiplier")
                if endpoint_snapshot else None)
            row["endpoint_feedback_local_count_after"] = (
                endpoint_local_after.get("feedback_count")
                if endpoint_snapshot else None)
            row["endpoint_feedback_remote_count_after"] = (
                endpoint_remote_after.get("feedback_count")
                if endpoint_snapshot else None)
            row["endpoint_passive_feedback_enabled"] = (
                self.endpoint_passive_feedback)
            row["endpoint_passive_registered"] = (
                passive_endpoint_request is not None)
            row["endpoint_external_credit_registered"] = (
                passive_endpoint_request is not None
                and passive_endpoint_request.get("external_credit") is True)
            row["endpoint_passive_route"] = (
                passive_endpoint_request["route"].value
                if passive_endpoint_request is not None else None)
            row["endpoint_passive_prior_ttft_ms"] = (
                passive_endpoint_request["prior_ttft_ms"]
                if passive_endpoint_request is not None else None)
            row["endpoint_passive_service_lookup_mode"] = (
                passive_endpoint_request["service_lookup_mode"]
                if passive_endpoint_request is not None else None)
            row["endpoint_passive_service_source_prompt_tokens"] = (
                passive_endpoint_request["service_source_prompt_tokens"]
                if passive_endpoint_request is not None else None)
            row["endpoint_passive_service_source_output_tokens"] = (
                passive_endpoint_request["service_source_output_tokens"]
                if passive_endpoint_request is not None else None)
            row["endpoint_passive_service_source_cache_residency"] = (
                passive_endpoint_request[
                    "service_source_cache_residency"].value
                if passive_endpoint_request is not None else None)
            if endpoint_request is not None:
                row["admission_credit_scope"] = (
                    "endpoint_prefill_or_remote_handoff"
                    if endpoint_decision
                    and endpoint_decision["route"]
                    != EndpointRoute.QUEUE.value
                    else None
                )
                row["admission_credit_release_event"] = (
                    endpoint_event.get("event")
                    if endpoint_event is not None else None)
                row["admission_credit_released_ns"] = (
                    endpoint_event.get("released_ns")
                    if endpoint_event is not None else None)
            elif passive_endpoint_request is not None:
                row["admission_credit_scope"] = None
                row["admission_credit_release_event"] = None
                row["admission_credit_released_ns"] = None
            row["decoder_prefix_caching"] = self.decoder_prefix_caching
            row["decoder_prefix_cache_block_size"] = (
                DECODER_CACHE_BLOCK_SIZE
                if self.decoder_prefix_caching else None)
            row["decoder_prefix_cached_tokens"] = (
                decoder_evidence.local_cached_tokens
                if decoder_evidence is not None else None)
            row["decoder_total_cached_tokens"] = (
                decoder_evidence.cached_tokens
                if decoder_evidence is not None else None)
            row["decoder_external_cached_tokens"] = (
                decoder_evidence.external_cached_tokens
                if decoder_evidence is not None else None)
            row["decoder_prefix_usage_prompt_tokens"] = (
                decoder_evidence.prompt_tokens
                if decoder_evidence is not None else None)
            row["decoder_prefix_expected_full_hit_tokens"] = (
                full_prefix_hit_tokens(
                    row["prompt_tokens"],
                    block_size=DECODER_CACHE_BLOCK_SIZE,
                )
                if self.decoder_prefix_caching else None)
            row["decoder_prefix_full_hit_observed"] = (
                decoder_evidence.local_cached_tokens == full_prefix_hit_tokens(
                    row["prompt_tokens"],
                    block_size=DECODER_CACHE_BLOCK_SIZE,
                )
                and decoder_evidence.local_cached_tokens > 0
                if decoder_evidence is not None else None)
            row["decoder_prefix_cache_evidence_source"] = (
                decoder_evidence.source
                if decoder_evidence is not None else None)
            row["decoder_cache_reuse_enabled_for_request"] = measured_reuse
            row["decoder_cache_reuse_items"] = reuse_items
            row["benchmark_cold_measured"] = self.cold_measured
            row["request_cache_contract"] = (
                explicit_cache_contract(request_id)
                or (
                    "miss"
                    if self.cold_measured and "-measured-" in request_id
                    else None
                )
            )
            row["upstream_cache_salt"] = (
                priority["cache_salt"] if priority is not None else None)
            row["decoder_prefix_read_skipped"] = (
                priority["skip_local_prefix_read"]
                if priority is not None else None)
            row["remote_decode_placement"] = (
                self.remote_decode_placement)
            row["proxy_kv_control_overlap"] = (
                self.proxy_kv_control_overlap)
            row["local_decoder_index"] = self.local_decoder_index
            row["remote_decoder_index"] = decoder_index
            row["remote_decoder_crossed"] = (
                decoder_index is not None
                and self.local_decoder_index is not None
                and decoder_index != self.local_decoder_index
            )
            row["remote_catchup_priority_configured"] = (
                self.remote_catchup_priority)
            row["strong_remote_catchup_priority_configured"] = (
                self.strong_remote_catchup_priority)
            row["long_remote_catchup_priority_configured"] = (
                self.long_remote_catchup_priority)
            row["long_remote_catchup_min_prompt_tokens"] = (
                self.long_remote_catchup_min_prompt_tokens)
            row["remote_catchup_min_output_tokens"] = (
                self.remote_catchup_min_output_tokens)
            row["median_guard_priority_configured"] = (
                self.median_guard_priority)
            row["medium_remote_catchup_priority_configured"] = (
                self.medium_remote_catchup_priority)
            row["externality_spill_budget_ms"] = (
                self.elastic.externality_spill_budget_ms)
            row["profile_remote_kv_budget_bytes"] = (
                self.elastic.profile_remote_kv_budget_bytes)
            row["effective_remote_kv_budget_bytes"] = (
                self.elastic.effective_remote_kv_budget_bytes)
            row["remote_headroom_kv_budget_bytes"] = (
                self.elastic.remote_headroom_kv_budget_bytes)
            row["profile_remote_backend"] = (
                self.profile.identity.remote_backend)
            row["remote_request_budget"] = (
                request_credit.get("remote_request_budget"))
            row["remote_requests_used_before"] = (
                request_credit.get("remote_requests_used_before"))
            row["remote_request_credit_available"] = (
                request_credit.get("remote_request_credit_available"))
            row["remote_headroom_request_budget"] = (
                request_credit.get("remote_headroom_request_budget"))
            row["remote_headroom_requests_used_before"] = request_credit.get(
                "remote_headroom_requests_used_before")
            row["remote_headroom_request_credit_available"] = (
                request_credit.get(
                    "remote_headroom_request_credit_available"))
            row["remote_headroom_eligible"] = (
                request_credit.get("remote_headroom_eligible"))
            row["remote_headroom_credit_consumed"] = (
                request_credit.get("remote_headroom_credit_consumed"))
            row["cold_unknown_remote_candidate"] = (
                request_credit.get("cold_unknown_remote_candidate"))
            row["cold_unknown_remote_admitted"] = (
                request_credit.get("cold_unknown_remote_admitted"))
            row["cold_high_load_headroom_candidate"] = (
                request_credit.get(
                    "cold_high_load_headroom_candidate"))
            row["cold_high_load_headroom_consumed"] = (
                request_credit.get(
                    "cold_high_load_headroom_consumed"))
            row["short_remote_min_advantage_ms"] = (
                self.elastic.short_remote_min_advantage_ms)
            row["headroom_medium_min_output_tokens"] = (
                self.elastic.headroom_medium_min_output_tokens)
            row["upstream_priority_requested"] = (
                priority["requested"] if priority is not None else None)
            row["upstream_priority_effective"] = (
                priority["effective"] if priority is not None else None)
            row["remote_catchup_priority_eligible"] = (
                priority["eligible"] if priority is not None else None)
            row["remote_catchup_priority_applied"] = (
                priority["remote_applied"] if priority is not None else None)
            row["strong_remote_catchup_priority_eligible"] = (
                priority["strong_remote_eligible"]
                if priority is not None else None)
            row["strong_remote_catchup_priority_applied"] = (
                priority["strong_remote_applied"]
                if priority is not None else None)
            row["long_remote_catchup_priority_eligible"] = (
                priority["long_remote_eligible"]
                if priority is not None else None)
            row["long_remote_catchup_priority_applied"] = (
                priority["long_remote_applied"]
                if priority is not None else None)
            row["medium_remote_catchup_priority_eligible"] = (
                priority["medium_remote_eligible"] if priority is not None else None)
            row["medium_remote_catchup_priority_applied"] = (
                priority["medium_remote_applied"] if priority is not None else None)
            row["median_guard_priority_eligible"] = (
                priority["median_guard_eligible"] if priority is not None else None)
            row["median_guard_priority_applied"] = (
                priority["median_guard_applied"] if priority is not None else None)
            row["upstream_priority_class"] = (
                priority["priority_class"] if priority is not None else None)
            row["remote_priority_fabric_congested"] = (
                priority["fabric_congested"]
                if priority is not None else None)
            row["remote_priority_fabric_congestion_suppressed"] = (
                priority["fabric_congestion_suppressed"]
                if priority is not None else None)
            row["upstream_token_ids_forwarded"] = (
                priority["token_ids_forwarded"]
                if priority is not None else None)
            row["lmcache_source_cached_tokens"] = cached_tokens
            row["lmcache_source_full_hit_observed"] = (
                cached_tokens == row["prompt_tokens"]
                if cached_tokens is not None
                else None)
            row["cache_namespace"] = namespace
            row["cache_residency_source"] = (
                "confirmed_completion_event"
                if namespace and self.cache_catalog.event(namespace) is not None
                else "unknown_until_completion_event")
        return rows


def _headers(record):
    return {
        "X-Tempo-PD-Schema": ROUTER_SCHEMA,
        "X-Tempo-PD-Request-Id": record.request_id,
        "X-Tempo-PD-Arm": record.arm.value,
        "X-Tempo-PD-Route": record.route.value,
        "X-Tempo-PD-Reason": record.reason,
        "X-Tempo-PD-Profile": record.profile_id,
        "X-Tempo-PD-Profile-SHA256": record.profile_fingerprint_sha256,
    }


class _CanonicalWireMiddleware:
    """Keep canonical wire identity app-local while runtime globals are restored."""

    _JSON_PATHS = frozenset((
        "/health",
        "/tempo/decisions",
        "/tempo/endpoint_controller",
        "/tempo/reset_endpoint_controller",
    ))

    def __init__(self, app):
        self.app = app

    @staticmethod
    def _schema_headers(headers, *, content_length=None):
        result = [
            (key, value) for key, value in headers
            if key.lower() not in {b"x-tempo-pd-schema", b"content-length"}
        ]
        result.append((b"x-tempo-pd-schema", ROUTER_SCHEMA.encode()))
        if content_length is not None:
            result.append((b"content-length", str(content_length).encode()))
        return result

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path")
        start_message = None
        body_parts = []

        async def send_canonical(message):
            nonlocal start_message
            if message["type"] == "http.response.start":
                if path in self._JSON_PATHS:
                    start_message = message
                else:
                    value = dict(message)
                    value["headers"] = self._schema_headers(message.get("headers", []))
                    await send(value)
                return
            if message["type"] == "http.response.body" and path in self._JSON_PATHS:
                body_parts.append(message.get("body", b""))
                if message.get("more_body", False):
                    return
                body = b"".join(body_parts)
                try:
                    payload = json.loads(body)
                    if isinstance(payload, dict):
                        payload["schema"] = ROUTER_SCHEMA
                        if "runtime_schema" in payload:
                            payload["runtime_schema"] = ROUTER_SCHEMA
                        body = json.dumps(payload, separators=(",", ":")).encode()
                except (TypeError, ValueError):
                    pass
                if start_message is not None:
                    value = dict(start_message)
                    value["headers"] = self._schema_headers(
                        start_message.get("headers", []), content_length=len(body))
                    await send(value)
                await send({"type": "http.response.body", "body": body,
                            "more_body": False})
                return
            await send(message)

        await self.app(scope, receive, send_canonical)


def build_app(config, profile, *, allow_screen_profile=False,
              cache_catalog=None, queue_wait_ms=100.0):
    original_core = runtime.ElasticPDRouterCore
    original_headers = runtime._headers
    original_schema = runtime.ROUTER_SCHEMA
    original_wire_schema = wire.ROUTER_SCHEMA

    def factory(config_value, profile_value, *, allow_screen_profile=False):
        return ElasticPDRouterCore(
            config_value, profile_value, allow_screen_profile=allow_screen_profile,
            cache_catalog=cache_catalog)

    runtime.ElasticPDRouterCore = factory
    runtime._headers = _headers
    runtime.ROUTER_SCHEMA = ROUTER_SCHEMA
    wire.ROUTER_SCHEMA = ROUTER_SCHEMA
    try:
        app = runtime.build_app(config, profile,
                                allow_screen_profile=allow_screen_profile,
                                queue_wait_ms=queue_wait_ms)
        app.add_middleware(_CanonicalWireMiddleware)

        @app.get("/tempo/endpoint_controller")
        async def endpoint_controller():
            return app.state.tempo_core.endpoint_controller_state()

        @app.post("/tempo/reset_endpoint_controller")
        async def reset_endpoint_controller():
            try:
                return app.state.tempo_core.reset_endpoint_controller()
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        @app.post("/tempo/reset_decoder_prefix_cache")
        async def reset_decoder_prefix_cache():
            if os.environ.get(
                "TEMPO_PD_BENCHMARK_RESET_DECODER_APC", "0"
            ) != "1":
                raise HTTPException(
                    status_code=403,
                    detail="decoder APC reset is disabled",
                )
            response = await app.state.local.post(
                "/reset_prefix_cache",
                params={"reset_external": "false"},
            )
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict) or value.get("success") is not True:
                raise HTTPException(
                    status_code=409,
                    detail="decoder prefix cache reset was not quiescent",
                )
            return {
                "schema": ROUTER_SCHEMA,
                "success": True,
                "external_cache_reset": False,
            }

    except BaseException:
        runtime.ElasticPDRouterCore = original_core
        runtime._headers = original_headers
        runtime.ROUTER_SCHEMA = original_schema
        wire.ROUTER_SCHEMA = original_wire_schema
        raise
    runtime.ElasticPDRouterCore = original_core
    runtime._headers = original_headers
    runtime.ROUTER_SCHEMA = original_schema
    wire.ROUTER_SCHEMA = original_wire_schema
    app.state.tempo_elastic_schema = ROUTER_SCHEMA
    return app


def _parse(argv=None):
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
    parser.add_argument("--require-replicated-profile", action="store_true")
    parser.add_argument("--queue-wait-ms", type=float, default=100.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    profile = load_elastic_profile(args.profile.resolve())
    if args.require_replicated_profile:
        require_replicated_profile(profile)
    config = base.RouterConfig(
        mode=base.RouterMode.TEMPO_AUTO, local_url=args.local_url,
        remote_url=args.remote_url, tokenizer_url=args.tokenizer_url,
        served_model_name=args.served_model_name, model_id=args.model_id,
        model_revision=args.model_revision, topology_id=args.topology_id,
        remote_backend=args.remote_backend,
        classifier_version=args.classifier_version,
        decoder_load_bucket=args.decoder_load_bucket,
        kv_bytes_per_token=args.kv_bytes_per_token)
    import uvicorn
    uvicorn.run(build_app(config, profile,
                          allow_screen_profile=args.allow_screen_profile,
                          queue_wait_ms=args.queue_wait_ms),
                host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
