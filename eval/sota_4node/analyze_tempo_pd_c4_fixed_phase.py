#!/usr/bin/env python3
"""Validate and summarize the frozen C4 fixed-phase characterization.

The analyzer is intentionally fail closed.  It accepts the node-0 result,
revalidates every provenance edge down to the four measured block files, and
keeps all endpoint arithmetic endpoint-local.  The output may authorize a
calibration-only profile fit; it never authorizes a performance or physical
switch-bottleneck claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from eval.sota_4node import build_tempo_pd_c4_phase_manifest as manifest_builder
from eval.sota_4node import run_tempo_pd_c4_fixed_phase_client as client
from eval.sota_4node import tempo_pd_endpoint_probe as endpoint_probe
from eval.sota_4node import verify_tempo_pd_c4_implementation as implementation
from tempo.cassini_endpoint import validate_cassini_endpoint_sample
from tempo.domain_evidence import CounterSupport
from tempo.pd_contention_workload import (
    CacheState,
    ForegroundArm,
    LoadSelection,
    Tenant,
    TrafficShape,
    VALIDATION_FOREGROUND_GEOMETRIES,
    build_schedule,
    semantic_schedule_sha256,
)
from tempo.pd_endpoint_evidence import (
    EndpointMetric,
    PDEndpointIdentity,
    PDEndpointRole,
    PDEndpointSnapshot,
    endpoint_metric_names,
)


SCHEMA = "tempo-pd-c4-fixed-phase-analysis-v1"
NODE_SCHEMA = "tempo-pd-c4-fixed-phase-node-v1"
STREAM_SCHEMA = "tempo-pd-stream-metrics-raw-1"
REPO_ROOT = Path(__file__).resolve().parents[2]

_EXPECTED_BLOCKS = (
    ("00_local_r0", ForegroundArm.LOCAL, 0),
    ("01_remote_r0", ForegroundArm.REMOTE, 0),
    ("02_remote_r1", ForegroundArm.REMOTE, 1),
    ("03_local_r1", ForegroundArm.LOCAL, 1),
)
_EXPECTED_ENDPOINTS = (
    ("pair0-prefill", PDEndpointRole.PREFILL, 0),
    ("pair0-decoder", PDEndpointRole.DECODER, 0),
    ("pair1-prefill", PDEndpointRole.PREFILL, 1),
    ("pair1-decoder", PDEndpointRole.DECODER, 1),
)
_NODE_KEYS = frozenset({
    "schema", "raw", "raw_sha256", "phase_manifest",
    "phase_manifest_sha256", "phase_manifest_fingerprint_sha256",
    "implementation_contract", "implementation_contract_sha256",
    "implementation_fingerprint_sha256", "implementation_file_count",
    "implementation_git_heads", "implementation_environment_versions",
    "fixed_runtime_environment", "transport_environment", "elastic_profile",
    "elastic_profile_sha256", "source_workload", "source_workload_sha256",
    "slurm_job_id", "startup_readiness_timeout_s", "block_count",
    "paired_output_count", "phase_service_row_count",
    "phase_route_summary_count", "cache_state_protocol_completion_backed",
    "decoder_cache_source_breakdown_exact", "phase_aligned_endpoint_evidence",
    "decoder_residency_basis", "characterization_gate_pass",
    "controller_tuning_allowed", "performance_claim_allowed",
    "physical_switch_bottleneck_claim_allowed", "unchanged_pd_data_plane",
    "transport",
})
_CLIENT_KEYS = frozenset({
    "schema", "run_id", "manifest", "manifest_sha256",
    "manifest_fingerprint_sha256", "cache_plan", "cache_plan_sha256",
    "cache_runtime_evidence", "cache_runtime_evidence_sha256", "block_order",
    "artifacts", "contracts", "gate", "performance_claim_allowed",
    "controller_tuning_allowed",
})
_BLOCK_CONTRACT_KEYS = frozenset({
    "schema", "sequence", "foreground_arm", "replicate",
    "semantic_schedule_sha256", "request_index", "all_requests_valid",
    "decision_cache_states_exact", "completion_cache_evidence_exact",
    "workload_start_marker_exact", "phase_aligned_endpoint_evidence",
    "preparation_outside_measurement", "actual_inference_background_only",
    "cross_endpoint_clock_subtraction_allowed",
})
_REQUEST_METADATA_KEYS = frozenset({
    "phase", "tenant", "arrival_offset_ms", "prompt_tokens",
    "output_tokens", "cache_state", "ordinal", "arm",
    "prompt_token_sha256", "terminal_item", "pair_key",
})
_GATE_KEYS = frozenset({
    "all_blocks_valid", "paired_semantic_schedules_exact",
    "paired_output_digests_exact", "phase_aligned_endpoint_evidence",
    "phase_geometry_cells_complete", "paired_output_count", "service_rows",
    "phase_service_rows", "phase_route_summaries",
    "authorizes_endpoint_profile_fit", "performance_claim_allowed",
})
_PHASE_SERVICE_KEYS = frozenset({
    "phase", "prompt_tokens", "output_tokens", "cache_state",
    "local_ttft_median_ms", "local_e2e_median_ms", "local_tpot_median_ms",
    "remote_ttft_median_ms", "remote_e2e_median_ms",
    "remote_tpot_median_ms", "remote_minus_local_ttft_median_ms",
    "remote_minus_local_e2e_median_ms",
    "remote_minus_local_tpot_median_ms", "remote_e2e_faster_fraction",
    "paired_samples",
})
_PHASE_ROUTE_KEYS = frozenset({
    "phase", "remote_minus_local_ttft_median_ms",
    "remote_minus_local_e2e_median_ms",
    "remote_minus_local_tpot_median_ms", "remote_e2e_faster_fraction",
    "paired_samples",
})
_SERVICE_KEYS = frozenset({
    "prompt_tokens", "output_tokens", "cache_state",
    "local_ttft_median_ms", "local_e2e_median_ms", "local_tpot_median_ms",
    "remote_ttft_median_ms", "remote_e2e_median_ms",
    "remote_tpot_median_ms", "samples_local", "samples_remote",
})
_ENDPOINT_KEYS = frozenset({
    "schema", "endpoint_id", "role", "pair_index", "sequence",
    "endpoint_monotonic_ns", "source", "metrics",
})
_SNAPSHOT_WRAPPER_KEYS = frozenset({
    "source_url", "client_fetch_started_monotonic_ns",
    "client_received_monotonic_ns", "probe",
})
_PROBE_KEYS = frozenset({
    "schema", "endpoint", "vllm_cumulative", "vllm_metrics_fetch",
    "cassini",
})
_FETCH_KEYS = frozenset({
    "attempts_configured", "attempts_used", "timeout_s_per_attempt",
    "retry_backoff_s", "transient_errors", "elapsed_ns",
})
_MEAN_PREFIXES = (
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_prefill_kv_computed_tokens",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be lowercase SHA-256",
    )
    return value


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    _require(path.is_file(), f"{name} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {name}") from exc
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _bound_absolute_path(
    raw_path: object, raw_sha: object, *, name: str,
    within: Path | None = None,
) -> Path:
    _canonical_sha(raw_sha, name=f"{name}.sha256")
    _require(type(raw_path) is str and Path(raw_path).is_absolute(),
             f"{name} path must be absolute")
    path = Path(raw_path).resolve()
    if within is not None:
        try:
            path.relative_to(within.resolve())
        except ValueError as exc:
            raise ValueError(f"{name} escapes its bound result root") from exc
    _require(path.is_file(), f"{name} is missing")
    _require(_sha256(path) == raw_sha, f"{name} digest differs")
    return path


def _manifest_entry_path(entry: object, *, name: str) -> Path:
    _require(isinstance(entry, dict) and set(entry) == {"path", "sha256"},
             f"manifest {name} binding is not exact")
    raw_path = entry["path"]
    _require(type(raw_path) is str and raw_path, f"manifest {name} path missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    _require(path.is_file(), f"manifest {name} is missing")
    _require(_sha256(path) == entry["sha256"],
             f"manifest {name} digest differs")
    return path


def _validate_manifest(path: Path, result: Mapping[str, object]) -> dict[str, object]:
    value = _load_object(path, name="C4 phase manifest")
    _require(
        value.get("schema") == manifest_builder.SCHEMA
        and value.get("fingerprint_sha256")
        == manifest_builder.manifest_fingerprint(value)
        == result["phase_manifest_fingerprint_sha256"],
        "C4 phase manifest fingerprint differs",
    )
    _require(
        value.get("phase_order") == [phase.value for phase in manifest_builder.PHASES]
        and value.get("performance_claim_allowed") is False
        and value.get("controller_tuning_allowed") is False
        and value.get("physical_switch_bottleneck_claim_allowed") is False
        and value.get("transport") == "LMCacheConnectorV1:UCX",
        "C4 phase manifest claim or phase contract differs",
    )
    endpoint = value.get("endpoint_evidence_contract")
    _require(
        isinstance(endpoint, dict)
        and endpoint.get("phase_boundary_samples") == 7
        and endpoint.get("phase_midpoint_samples") == 6
        and endpoint.get("cassini_phase_windows")
        == "two_nonoverlapping_half_phase_endpoint_deltas"
        and endpoint.get("vllm_phase_windows")
        == "boundary_to_boundary_cumulative_deltas"
        and endpoint.get("cross_host_clock_subtraction_allowed") is False,
        "C4 endpoint phase-window contract differs",
    )
    return value


def _validate_implementation_contract(
    path: Path, result: Mapping[str, object], *, phase_manifest: Path,
) -> dict[str, object]:
    value = _load_object(path, name="C4 implementation contract")
    _require(
        value.get("schema") == implementation.SCHEMA
        and value.get("fingerprint_sha256")
        == implementation.contract_fingerprint(value)
        == result["implementation_fingerprint_sha256"],
        "C4 implementation fingerprint differs",
    )
    manifest = value.get("phase_manifest")
    _require(isinstance(manifest, dict) and set(manifest) == {"path", "sha256"},
             "implementation phase-manifest binding differs")
    bound_manifest = Path(str(manifest["path"]))
    if not bound_manifest.is_absolute():
        bound_manifest = REPO_ROOT / bound_manifest
    _require(
        bound_manifest.resolve() == phase_manifest
        and manifest["sha256"] == _sha256(phase_manifest)
        and len(value.get("files", [])) == result["implementation_file_count"]
        and value.get("git_heads") == result["implementation_git_heads"]
        and value.get("environment_versions")
        == result["implementation_environment_versions"]
        and value.get("performance_claim_allowed") is False,
        "C4 implementation provenance differs from the node result",
    )
    return value


def _expected_schedule(
    manifest: Mapping[str, object], *, arm: ForegroundArm, replicate: int,
):
    rates = manifest.get("background_rates_per_s")
    _require(isinstance(rates, Mapping), "C4 background rates are missing")
    selection = LoadSelection(
        decoder_reference_rate_per_s=float(rates["decoder_hot"]),
        remote_reference_rate_per_s=float(rates["cold_remote_hot"]),
        decoder_fraction=1.0,
        remote_fraction=1.0,
        kv_remote_rate_per_s=float(rates["kv_remote_hot"]),
    )
    return build_schedule(
        states=manifest_builder.PHASES,
        selection=selection,
        foreground_arm=arm,
        foreground_rate_per_s=float(manifest["foreground_rate_per_s"]),
        trial_id=f"c4-r{replicate}-{arm.value}",
        shape=TrafficShape.STABLE,
        phase_duration_ms=float(manifest["phase_duration_ms"]),
        foreground_geometries=VALIDATION_FOREGROUND_GEOMETRIES,
        passive_endpoint_feedback=True,
    )


def _expected_request_index(
    manifest: Mapping[str, object], *, sequence: int,
    arm: ForegroundArm, replicate: int,
) -> tuple[dict[str, dict[str, object]], str]:
    schedule = _expected_schedule(manifest, arm=arm, replicate=replicate)
    result: dict[str, dict[str, object]] = {}
    for request in schedule:
        geometry = request.geometry
        geometry_index = (
            VALIDATION_FOREGROUND_GEOMETRIES.index(geometry)
            if request.tenant is Tenant.FOREGROUND else -1
        )
        terminal_item = client._terminal_item(
            tenant=request.tenant,
            ordinal=request.ordinal,
            geometry_index=geometry_index,
            cache_state=geometry.cache_state,
        )
        request_id = client._request_id(
            sequence=sequence,
            arm=request.arm,
            replicate=replicate,
            phase=request.phase,
            tenant=request.tenant,
            ordinal=request.ordinal,
            state=geometry.cache_state,
            terminal_item=terminal_item,
        )
        result[request_id] = {
            **request.semantic_dict(),
            "arm": request.arm.value,
            "terminal_item": terminal_item,
            "pair_key": (
                f"r{replicate}:{request.phase.value}:"
                f"foreground:{request.ordinal:06d}"
                if request.tenant is Tenant.FOREGROUND else None
            ),
        }
    return result, semantic_schedule_sha256(schedule)


def _validate_endpoint_payload(raw: object) -> dict[str, object]:
    _require(isinstance(raw, dict) and set(raw) == _PROBE_KEYS,
             "endpoint probe payload inventory differs")
    _require(raw.get("schema") == endpoint_probe.SCHEMA,
             "endpoint probe schema differs")
    endpoint = raw.get("endpoint")
    _require(isinstance(endpoint, dict) and set(endpoint) == _ENDPOINT_KEYS,
             "endpoint snapshot inventory differs")
    try:
        role = PDEndpointRole(endpoint["role"])
        identity = PDEndpointIdentity(
            endpoint_id=endpoint["endpoint_id"],
            role=role,
            pair_index=endpoint["pair_index"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("endpoint identity is invalid") from exc
    metrics = endpoint.get("metrics")
    _require(isinstance(metrics, dict)
             and set(metrics) == set(endpoint_metric_names(role)),
             "endpoint metric inventory differs")
    parsed_metrics = []
    for name in sorted(metrics):
        metric = metrics[name]
        _require(isinstance(metric, dict)
                 and set(metric) == {"support", "value"},
                 "endpoint metric binding differs")
        try:
            parsed_metrics.append(EndpointMetric(
                name=name,
                support=CounterSupport(metric["support"]),
                value=metric["value"],
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"endpoint metric is invalid: {name}") from exc
    snapshot = PDEndpointSnapshot(
        identity=identity,
        sequence=endpoint["sequence"],
        endpoint_monotonic_ns=endpoint["endpoint_monotonic_ns"],
        source=endpoint["source"],
        metrics=tuple(parsed_metrics),
        schema=endpoint["schema"],
    )
    _require(snapshot.as_dict() == endpoint, "endpoint snapshot is not canonical")

    cumulative = raw.get("vllm_cumulative")
    endpoint_probe.validate_vllm_endpoint_cumulative(cumulative)
    cassini = raw.get("cassini")
    validate_cassini_endpoint_sample(cassini)
    _require(
        cassini["endpoint_id"] == identity.endpoint_id
        and cassini["role"] == role.value
        and cassini["pair_index"] == identity.pair_index,
        "endpoint and Cassini identities differ",
    )
    fetch = raw.get("vllm_metrics_fetch")
    _require(isinstance(fetch, dict) and set(fetch) == _FETCH_KEYS,
             "vLLM metrics-fetch provenance differs")
    _require(
        type(fetch["attempts_configured"]) is int
        and 1 <= fetch["attempts_configured"] <= 3
        and type(fetch["attempts_used"]) is int
        and 1 <= fetch["attempts_used"] <= fetch["attempts_configured"]
        and isinstance(fetch["timeout_s_per_attempt"], (int, float))
        and not isinstance(fetch["timeout_s_per_attempt"], bool)
        and 0.0 < float(fetch["timeout_s_per_attempt"]) <= 10.0
        and fetch["retry_backoff_s"] == 0.05
        and isinstance(fetch["transient_errors"], list)
        and len(fetch["transient_errors"]) == fetch["attempts_used"] - 1
        and type(fetch["elapsed_ns"]) is int
        and fetch["elapsed_ns"] >= 0,
        "vLLM metrics-fetch values are invalid",
    )
    return raw


def _validate_snapshot_wrapper(raw: object) -> dict[str, object]:
    _require(isinstance(raw, dict) and set(raw) == _SNAPSHOT_WRAPPER_KEYS,
             "endpoint snapshot wrapper inventory differs")
    started = raw.get("client_fetch_started_monotonic_ns")
    received = raw.get("client_received_monotonic_ns")
    _require(
        type(raw.get("source_url")) is str
        and str(raw["source_url"]).startswith("http://")
        and type(started) is int
        and type(received) is int
        and 0 <= started <= received,
        "endpoint snapshot client timing differs",
    )
    _validate_endpoint_payload(raw.get("probe"))
    return raw


def _sample_index(sample: object) -> dict[str, dict[str, object]]:
    _require(isinstance(sample, dict), "endpoint phase sample is malformed")
    snapshots = sample.get("snapshots")
    _require(isinstance(snapshots, list) and len(snapshots) == 4,
             "endpoint phase sample count differs")
    result: dict[str, dict[str, object]] = {}
    for raw in snapshots:
        row = _validate_snapshot_wrapper(raw)
        endpoint_id = row["probe"]["endpoint"]["endpoint_id"]
        _require(endpoint_id not in result, "duplicate endpoint phase sample")
        result[endpoint_id] = row
    expected = {item[0] for item in _EXPECTED_ENDPOINTS}
    _require(set(result) == expected, "endpoint phase identity set differs")
    return result


def _cumulative_delta(
    start: Mapping[str, object], end: Mapping[str, object], *, name: str,
) -> dict[str, object]:
    endpoint_probe.validate_vllm_endpoint_cumulative(start)
    endpoint_probe.validate_vllm_endpoint_cumulative(end)
    _require(
        start["model_name"] == end["model_name"]
        and start["engine_indices"] == end["engine_indices"],
        f"{name} vLLM cumulative identity changed",
    )
    start_values = start["values"]
    end_values = end["values"]
    delta: dict[str, int | float] = {}
    for metric in sorted(endpoint_probe.VLLM_CUMULATIVE_METRICS):
        initial = start_values[metric]
        final = end_values[metric]
        _require(final >= initial, f"{name} cumulative vLLM metric regressed")
        delta[metric] = final - initial
    derived: dict[str, float | None] = {}
    for prefix in _MEAN_PREFIXES:
        count = delta[prefix + "_count"]
        total = delta[prefix + "_sum"]
        _require(count > 0 or math.isclose(float(total), 0.0, abs_tol=1e-12),
                 f"{name} vLLM sum changed without observations")
        derived["mean_" + prefix.removeprefix("vllm:")] = (
            float(total) / int(count) if count else None
        )
    return {
        "model_name": end["model_name"],
        "engine_indices": end["engine_indices"],
        "delta": delta,
        "derived": derived,
    }


def _cassini_half(raw: Mapping[str, object], *, phase_duration_ms: float) -> dict[str, object]:
    validate_cassini_endpoint_sample(raw)
    _require(raw["valid"] is True, "measured Cassini half-window is invalid")
    window_ms = float(raw["window_ms"])
    _require(
        0.25 * phase_duration_ms <= window_ms <= 0.75 * phase_duration_ms,
        "Cassini half-window is not phase aligned",
    )
    return {
        "sequence": raw["sequence"],
        "sampled_ns": raw["sampled_ns"],
        "window_ms": raw["window_ms"],
        "read_ms": raw["read_ms"],
        "cache_age_ms": raw["cache_age_ms"],
        "support": raw["support"],
        "signals": raw["signals"],
    }


def _capture_received_offset_ns(
    sample: Mapping[str, object], *, run_start_ns: int,
) -> int:
    rows = _sample_index(sample)
    value = max(row["client_received_monotonic_ns"] for row in rows.values())
    _require(value >= run_start_ns, "phase snapshot predates the workload start")
    return value - run_start_ns


def _validate_capture_timing(
    evidence: Mapping[str, object], request_index: Mapping[str, object],
    *, phase_duration_ms: float,
) -> None:
    marker = evidence["measurement_start_marker"]
    run_start_ns = int(marker["run_start_ns"])
    phase_ns = int(phase_duration_ms * 1_000_000)
    boundaries = evidence["phase_boundaries"]
    midpoints = evidence["phase_midpoints"]
    for phase_index in range(len(manifest_builder.PHASES)):
        midpoint_offset = _capture_received_offset_ns(
            midpoints[phase_index], run_start_ns=run_start_ns)
        midpoint_target = int((phase_index + 0.5) * phase_ns)
        _require(
            midpoint_target <= midpoint_offset < midpoint_target + phase_ns // 4,
            "C4 midpoint snapshot missed its frozen phase window",
        )
        boundary_offset = _capture_received_offset_ns(
            boundaries[phase_index + 1], run_start_ns=run_start_ns)
        boundary_target = (phase_index + 1) * phase_ns
        _require(
            boundary_target <= boundary_offset < boundary_target + phase_ns // 4,
            "C4 boundary snapshot missed its frozen phase window",
        )
        if phase_index + 1 < len(manifest_builder.PHASES):
            next_phase = manifest_builder.PHASES[phase_index + 1].value
            next_arrivals = [
                int(float(metadata["arrival_offset_ms"]) * 1_000_000)
                for metadata in request_index.values()
                if metadata["phase"] == next_phase
            ]
            _require(bool(next_arrivals), "next C4 phase has no requests")
            _require(
                boundary_offset < min(next_arrivals),
                "C4 boundary snapshot overlapped the next phase",
            )


def _endpoint_phase_rows(
    evidence: Mapping[str, object], *, block_key: str, sequence: int,
    arm: ForegroundArm, replicate: int, phase_duration_ms: float,
) -> list[dict[str, object]]:
    boundaries = evidence["phase_boundaries"]
    midpoints = evidence["phase_midpoints"]
    result = []
    for phase_index, phase in enumerate(manifest_builder.PHASES):
        start_index = _sample_index(boundaries[phase_index])
        midpoint_index = _sample_index(midpoints[phase_index])
        end_index = _sample_index(boundaries[phase_index + 1])
        for endpoint_id, role, pair_index in _EXPECTED_ENDPOINTS:
            start = start_index[endpoint_id]
            midpoint = midpoint_index[endpoint_id]
            end = end_index[endpoint_id]
            probes = [row["probe"] for row in (start, midpoint, end)]
            identities = [probe["endpoint"] for probe in probes]
            _require(all(
                item["endpoint_id"] == endpoint_id
                and item["role"] == role.value
                and item["pair_index"] == pair_index
                for item in identities
            ), "endpoint identity changed within a phase")
            # Validate both halves as well as the boundary-to-boundary window.
            _cumulative_delta(
                probes[0]["vllm_cumulative"],
                probes[1]["vllm_cumulative"],
                name=f"{block_key}/{phase.value}/{endpoint_id}/first-half",
            )
            _cumulative_delta(
                probes[1]["vllm_cumulative"],
                probes[2]["vllm_cumulative"],
                name=f"{block_key}/{phase.value}/{endpoint_id}/second-half",
            )
            full_delta = _cumulative_delta(
                probes[0]["vllm_cumulative"],
                probes[2]["vllm_cumulative"],
                name=f"{block_key}/{phase.value}/{endpoint_id}",
            )
            result.append({
                "block_key": block_key,
                "block_sequence": sequence,
                "foreground_arm": arm.value,
                "replicate": replicate,
                "phase": phase.value,
                "phase_index": phase_index,
                "endpoint_id": endpoint_id,
                "role": role.value,
                "pair_index": pair_index,
                "snapshot_sequences": {
                    "boundary_start": identities[0]["sequence"],
                    "midpoint": identities[1]["sequence"],
                    "boundary_end": identities[2]["sequence"],
                },
                "endpoint_local_monotonic_ns": {
                    "boundary_start": identities[0]["endpoint_monotonic_ns"],
                    "midpoint": identities[1]["endpoint_monotonic_ns"],
                    "boundary_end": identities[2]["endpoint_monotonic_ns"],
                },
                "load_gauges": {
                    "boundary_start": identities[0]["metrics"],
                    "midpoint": identities[1]["metrics"],
                    "boundary_end": identities[2]["metrics"],
                },
                "vllm_boundary_to_boundary": full_delta,
                "cassini_first_half": _cassini_half(
                    probes[1]["cassini"],
                    phase_duration_ms=phase_duration_ms),
                "cassini_second_half": _cassini_half(
                    probes[2]["cassini"],
                    phase_duration_ms=phase_duration_ms),
            })
    _require(len(result) == 24, "C4 block endpoint-phase row count differs")
    return result


def _percentile(values: list[float], fraction: float) -> float:
    _require(bool(values), "percentile input is empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p99": None, "max": None}
    return {
        "median": statistics.median(values),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _request_phase_tenant_rows(
    *, block_key: str, sequence: int, arm: ForegroundArm, replicate: int,
    request_index: Mapping[str, Mapping[str, object]],
    requests: Mapping[str, Mapping[str, object]],
    decisions: Mapping[str, Mapping[str, object]], phase_duration_ms: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for request_id, metadata in request_index.items():
        grouped[(str(metadata["phase"]), str(metadata["tenant"]))].append(request_id)
    result = []
    for phase_index, phase in enumerate(manifest_builder.PHASES):
        phase_end_ns = int((phase_index + 1) * phase_duration_ms * 1_000_000)
        for tenant in Tenant:
            identifiers = grouped[(phase.value, tenant.value)]
            metrics = [client._request_service_metrics(requests[item])
                       for item in identifiers]
            route_counts = Counter(decisions[item]["route"] for item in identifiers)
            state_counts = Counter(
                str(request_index[item]["cache_state"]) for item in identifiers)
            result.append({
                "block_key": block_key,
                "block_sequence": sequence,
                "foreground_arm": arm.value,
                "replicate": replicate,
                "phase": phase.value,
                "phase_index": phase_index,
                "tenant": tenant.value,
                "request_count": len(identifiers),
                "valid_request_count": sum(
                    requests[item].get("valid") is True for item in identifiers),
                "prompt_tokens_total": sum(
                    int(request_index[item]["prompt_tokens"])
                    for item in identifiers),
                "output_tokens_total": sum(
                    int(request_index[item]["output_tokens"])
                    for item in identifiers),
                "cache_state_counts": {
                    state.value: state_counts[state.value] for state in CacheState
                },
                "route_counts": {
                    client._LOCAL_ROUTE: route_counts[client._LOCAL_ROUTE],
                    client._REMOTE_ROUTE: route_counts[client._REMOTE_ROUTE],
                },
                "ttft_ms": _metric_summary(
                    [row["ttft_ms"] for row in metrics]),
                "e2e_ms": _metric_summary(
                    [row["e2e_ms"] for row in metrics]),
                "tpot_ms": _metric_summary(
                    [row["tpot_ms"] for row in metrics]),
                "completed_after_phase_boundary_count": sum(
                    int(requests[item]["stream_end_offset_ns"]) > phase_end_ns
                    for item in identifiers),
            })
    _require(len(result) == 24, "C4 block request phase/tenant row count differs")
    return result


def _validate_block(
    path: Path, *, parent_contract: Mapping[str, object], block_key: str,
    sequence: int, arm: ForegroundArm, replicate: int,
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    raw = _load_object(path, name=f"C4 block {block_key}")
    _require(raw.get("schema") == STREAM_SCHEMA,
             f"C4 block stream schema differs: {block_key}")
    validation = raw.get("validation")
    _require(
        isinstance(validation, dict)
        and validation.get("all_streams_valid") is True
        and validation.get("router_decisions_exact") is True,
        f"C4 block stream validation failed: {block_key}",
    )
    contract = raw.get("c4_fixed_phase_contract")
    _require(isinstance(contract, dict) and set(contract) == _BLOCK_CONTRACT_KEYS,
             f"C4 block contract inventory differs: {block_key}")
    _require(contract == parent_contract,
             f"C4 parent/block contract differs: {block_key}")
    _require(
        contract.get("schema") == client.BLOCK_SCHEMA
        and contract.get("sequence") == sequence
        and contract.get("foreground_arm") == arm.value
        and contract.get("replicate") == replicate
        and contract.get("all_requests_valid") is True
        and contract.get("decision_cache_states_exact") is True
        and contract.get("completion_cache_evidence_exact") is True
        and contract.get("workload_start_marker_exact") is True
        and contract.get("phase_aligned_endpoint_evidence") is True
        and contract.get("preparation_outside_measurement") is True
        and contract.get("actual_inference_background_only") is True
        and contract.get("cross_endpoint_clock_subtraction_allowed") is False,
        f"C4 block validity contract failed: {block_key}",
    )
    request_index = contract.get("request_index")
    _require(isinstance(request_index, dict),
             f"C4 request index is missing: {block_key}")
    expected, expected_schedule_sha = _expected_request_index(
        manifest, sequence=sequence, arm=arm, replicate=replicate)
    _require(
        contract.get("semantic_schedule_sha256") == expected_schedule_sha
        and set(request_index) == set(expected),
        f"C4 semantic schedule differs: {block_key}",
    )
    for request_id, metadata in request_index.items():
        _require(isinstance(metadata, dict)
                 and set(metadata) == _REQUEST_METADATA_KEYS,
                 f"C4 request metadata inventory differs: {request_id}")
        expected_metadata = expected[request_id]
        for name, value in expected_metadata.items():
            _require(metadata.get(name) == value,
                     f"C4 request semantic field differs: {request_id}/{name}")
        _canonical_sha(metadata.get("prompt_token_sha256"),
                       name=f"{request_id}.prompt_token_sha256")

    requests_raw = raw.get("requests")
    decisions_raw = raw.get("router_decisions")
    _require(isinstance(requests_raw, list) and isinstance(decisions_raw, list),
             f"C4 block rows are missing: {block_key}")
    requests = {str(row.get("request_id")): row for row in requests_raw}
    decisions = {str(row.get("request_id")): row for row in decisions_raw}
    _require(
        len(requests) == len(requests_raw)
        and len(decisions) == len(decisions_raw)
        and set(requests) == set(decisions) == set(request_index),
        f"C4 block request/decision IDs differ: {block_key}",
    )
    for request_id, metadata in request_index.items():
        request = requests[request_id]
        decision = decisions[request_id]
        _require(
            isinstance(request, dict)
            and request.get("valid") is True
            and request.get("requested_max_tokens") == metadata["output_tokens"],
            f"C4 measured request is invalid: {request_id}",
        )
        _canonical_sha(request.get("output_text_sha256"),
                       name=f"{request_id}.output_text_sha256")
        client._request_service_metrics(request)
        _require(
            isinstance(decision, dict)
            and decision.get("phase") == "complete"
            and decision.get("error") is None,
            f"C4 measured decision is incomplete: {request_id}",
        )
        client._validate_measured_decision(decision, metadata)

    evidence = raw.get("endpoint_evidence")
    client._validate_c4_endpoint_evidence(evidence)
    _validate_capture_timing(
        evidence, request_index,
        phase_duration_ms=float(manifest["phase_duration_ms"]),
    )
    endpoint_rows = _endpoint_phase_rows(
        evidence,
        block_key=block_key,
        sequence=sequence,
        arm=arm,
        replicate=replicate,
        phase_duration_ms=float(manifest["phase_duration_ms"]),
    )
    request_rows = _request_phase_tenant_rows(
        block_key=block_key,
        sequence=sequence,
        arm=arm,
        replicate=replicate,
        request_index=request_index,
        requests=requests,
        decisions=decisions,
        phase_duration_ms=float(manifest["phase_duration_ms"]),
    )
    return raw, endpoint_rows, request_rows


def _validate_gate(gate: object) -> dict[str, object]:
    _require(isinstance(gate, dict) and set(gate) == _GATE_KEYS,
             "C4 gate inventory differs")
    _require(
        gate.get("all_blocks_valid") is True
        and gate.get("paired_semantic_schedules_exact") is True
        and gate.get("paired_output_digests_exact") is True
        and gate.get("phase_aligned_endpoint_evidence") is True
        and gate.get("phase_geometry_cells_complete") is True
        and gate.get("authorizes_endpoint_profile_fit") is True
        and gate.get("performance_claim_allowed") is False,
        "C4 gate does not authorize a calibration-only profile fit",
    )
    service_rows = gate.get("service_rows")
    phase_rows = gate.get("phase_service_rows")
    route_rows = gate.get("phase_route_summaries")
    _require(isinstance(service_rows, list)
             and all(isinstance(row, dict) and set(row) == _SERVICE_KEYS
                     for row in service_rows),
             "C4 service-row inventory differs")
    _require(isinstance(phase_rows, list)
             and all(isinstance(row, dict) and set(row) == _PHASE_SERVICE_KEYS
                     for row in phase_rows),
             "C4 phase service-row inventory differs")
    _require(isinstance(route_rows, list)
             and all(isinstance(row, dict) and set(row) == _PHASE_ROUTE_KEYS
                     for row in route_rows),
             "C4 phase route-summary inventory differs")
    geometry_keys = {
        (geometry.prompt_tokens, geometry.output_tokens, geometry.cache_state.value)
        for geometry in VALIDATION_FOREGROUND_GEOMETRIES
    }
    _require(
        len(service_rows) == len(geometry_keys)
        and {
            (row["prompt_tokens"], row["output_tokens"], row["cache_state"])
            for row in service_rows
        } == geometry_keys,
        "C4 service geometry inventory differs",
    )
    expected_phase_cells = {
        (phase.value, *geometry)
        for phase in manifest_builder.PHASES for geometry in geometry_keys
    }
    observed_phase_cells = {
        (row["phase"], row["prompt_tokens"], row["output_tokens"],
         row["cache_state"])
        for row in phase_rows
    }
    _require(
        len(phase_rows) == len(expected_phase_cells)
        and observed_phase_cells == expected_phase_cells
        and all(type(row["paired_samples"]) is int
                and row["paired_samples"] >= 4 for row in phase_rows),
        "C4 phase service cell inventory is incomplete",
    )
    _require(
        len(route_rows) == len(manifest_builder.PHASES)
        and [row["phase"] for row in route_rows]
        == [phase.value for phase in manifest_builder.PHASES],
        "C4 phase route-summary inventory differs",
    )
    return gate


def _foreground_paired_samples(
    block_paths: Mapping[str, Path],
    contracts: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    indexed: dict[int, dict[str, tuple[str, dict[str, object], dict[str, object]]]] = (
        defaultdict(dict))
    for key, arm, replicate in _EXPECTED_BLOCKS:
        raw = _load_object(block_paths[key], name=f"C4 paired block {key}")
        requests = {
            str(row["request_id"]): row for row in raw["requests"]
        }
        contract = contracts[key]
        foreground = {}
        for request_id, metadata in contract["request_index"].items():
            if metadata["tenant"] != Tenant.FOREGROUND.value:
                continue
            pair_key = metadata["pair_key"]
            _require(type(pair_key) is str and pair_key not in foreground,
                     "C4 foreground pair key is invalid or duplicated")
            foreground[pair_key] = {
                "request_id": request_id,
                "metadata": metadata,
                "request": requests[request_id],
            }
        indexed[replicate][arm.value] = (key, foreground, raw)

    result = []
    for replicate in (0, 1):
        _require(set(indexed[replicate]) == {"local", "remote"},
                 "C4 paired sample replicate lacks an arm")
        local_key, local, _ = indexed[replicate]["local"]
        remote_key, remote, _ = indexed[replicate]["remote"]
        _require(set(local) == set(remote),
                 "C4 paired sample semantic keys differ")
        for pair_key in sorted(local):
            local_value = local[pair_key]
            remote_value = remote[pair_key]
            local_metadata = local_value["metadata"]
            remote_metadata = remote_value["metadata"]
            _require(
                all(local_metadata[name] == remote_metadata[name] for name in (
                    "phase", "tenant", "arrival_offset_ms", "prompt_tokens",
                    "output_tokens", "cache_state", "ordinal", "pair_key",
                )),
                "C4 foreground paired semantics differ",
            )
            local_request = local_value["request"]
            remote_request = remote_value["request"]
            _require(
                local_request["output_text_sha256"]
                == remote_request["output_text_sha256"],
                "C4 foreground paired output digest differs",
            )
            local_metrics = client._request_service_metrics(local_request)
            remote_metrics = client._request_service_metrics(remote_request)
            result.append({
                "pair_key": pair_key,
                "replicate": replicate,
                "phase": local_metadata["phase"],
                "arrival_offset_ms": local_metadata["arrival_offset_ms"],
                "prompt_tokens": local_metadata["prompt_tokens"],
                "output_tokens": local_metadata["output_tokens"],
                "cache_state": local_metadata["cache_state"],
                "ordinal": local_metadata["ordinal"],
                "local_block_key": local_key,
                "remote_block_key": remote_key,
                "local_request_id": local_value["request_id"],
                "remote_request_id": remote_value["request_id"],
                "output_text_sha256": local_request["output_text_sha256"],
                "local": local_metrics,
                "remote": remote_metrics,
                "remote_minus_local": {
                    name: remote_metrics[name] - local_metrics[name]
                    for name in ("ttft_ms", "e2e_ms", "tpot_ms")
                },
            })
    result.sort(key=lambda row: (
        [phase.value for phase in manifest_builder.PHASES].index(row["phase"]),
        row["prompt_tokens"], row["output_tokens"], row["cache_state"],
        row["replicate"], row["ordinal"],
    ))
    return result


def _analysis_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def analyze(
    result_path: Path, *, expected_result_sha256: str,
) -> dict[str, object]:
    result_path = result_path.resolve()
    expected_result_sha256 = _canonical_sha(
        expected_result_sha256, name="node result SHA-256")
    _require(_sha256(result_path) == expected_result_sha256,
             "C4 node result digest differs")
    node = _load_object(result_path, name="C4 node result")
    _require(set(node) == _NODE_KEYS and node.get("schema") == NODE_SCHEMA,
             "C4 node result inventory differs")

    result_root = result_path.parent
    client_raw_path = _bound_absolute_path(
        node["raw"], node["raw_sha256"], name="C4 client raw",
        within=result_root)
    manifest_path = _bound_absolute_path(
        node["phase_manifest"], node["phase_manifest_sha256"],
        name="C4 phase manifest")
    manifest = _validate_manifest(manifest_path, node)
    implementation_path = _bound_absolute_path(
        node["implementation_contract"], node["implementation_contract_sha256"],
        name="C4 implementation contract")
    contract = _validate_implementation_contract(
        implementation_path, node, phase_manifest=manifest_path)
    elastic_profile = _bound_absolute_path(
        node["elastic_profile"], node["elastic_profile_sha256"],
        name="C4 Elastic profile")
    source_workload = _bound_absolute_path(
        node["source_workload"], node["source_workload_sha256"],
        name="C4 source workload")
    _require(
        _manifest_entry_path(manifest.get("elastic_profile"),
                             name="Elastic profile") == elastic_profile
        and _manifest_entry_path(manifest.get("source_workload"),
                                 name="source workload") == source_workload,
        "C4 node/manifest workload binding differs",
    )
    _require(
        node.get("fixed_runtime_environment")
        == manifest.get("fixed_runtime_environment")
        and isinstance(node.get("transport_environment"), dict)
        and type(node.get("slurm_job_id")) is str
        and bool(str(node["slurm_job_id"]).strip())
        and isinstance(node.get("startup_readiness_timeout_s"), (int, float))
        and 600.0 <= float(node["startup_readiness_timeout_s"]) <= 3600.0
        and node.get("block_count") == 4
        and node.get("phase_service_row_count") == 36
        and node.get("phase_route_summary_count") == 6
        and node.get("cache_state_protocol_completion_backed") is True
        and node.get("decoder_cache_source_breakdown_exact") is True
        and node.get("phase_aligned_endpoint_evidence") is True
        and node.get("decoder_residency_basis")
        == "exact_local_preparation_hit_on_original_P_token_prompt"
        and node.get("characterization_gate_pass") is True
        and node.get("controller_tuning_allowed") is True
        and node.get("performance_claim_allowed") is False
        and node.get("physical_switch_bottleneck_claim_allowed") is False
        and node.get("unchanged_pd_data_plane") is True
        and node.get("transport") == "LMCacheConnectorV1:UCX",
        "C4 node characterization invariants differ",
    )

    parent = _load_object(client_raw_path, name="C4 client raw")
    _require(set(parent) == _CLIENT_KEYS and parent.get("schema") == client.SCHEMA,
             "C4 client raw inventory differs")
    _require(
        Path(str(parent["manifest"])).resolve() == manifest_path
        and parent["manifest_sha256"] == node["phase_manifest_sha256"]
        and parent["manifest_fingerprint_sha256"]
        == node["phase_manifest_fingerprint_sha256"]
        and parent.get("performance_claim_allowed") is False
        and parent.get("controller_tuning_allowed") is True,
        "C4 client/node manifest or claim binding differs",
    )
    _bound_absolute_path(
        parent["cache_plan"], parent["cache_plan_sha256"],
        name="C4 cache preparation plan", within=client_raw_path.parent)
    _bound_absolute_path(
        parent["cache_runtime_evidence"],
        parent["cache_runtime_evidence_sha256"],
        name="C4 cache runtime evidence", within=client_raw_path.parent)

    expected_keys = [item[0] for item in _EXPECTED_BLOCKS]
    expected_order = [
        {"arm": arm.value, "replicate": replicate}
        for _key, arm, replicate in _EXPECTED_BLOCKS
    ]
    artifacts = parent.get("artifacts")
    contracts = parent.get("contracts")
    _require(
        isinstance(artifacts, dict)
        and list(artifacts) == expected_keys
        and isinstance(contracts, dict)
        and list(contracts) == expected_keys
        and parent.get("block_order") == expected_order,
        "C4 four-block ABBA inventory differs",
    )

    block_paths: dict[str, Path] = {}
    endpoint_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    paired_gate_blocks = []
    block_bindings = []
    for sequence, (key, arm, replicate) in enumerate(_EXPECTED_BLOCKS):
        entry = artifacts[key]
        _require(isinstance(entry, dict) and set(entry) == {"path", "sha256"},
                 f"C4 block binding inventory differs: {key}")
        path = _bound_absolute_path(
            entry["path"], entry["sha256"], name=f"C4 block {key}",
            within=client_raw_path.parent)
        block_paths[key] = path
        raw, block_endpoint_rows, block_request_rows = _validate_block(
            path,
            parent_contract=contracts[key],
            block_key=key,
            sequence=sequence,
            arm=arm,
            replicate=replicate,
            manifest=manifest,
        )
        del raw
        endpoint_rows.extend(block_endpoint_rows)
        request_rows.extend(block_request_rows)
        paired_gate_blocks.append({
            "sequence": sequence,
            "arm": arm,
            "replicate": replicate,
            "schedule_sha256": contracts[key]["semantic_schedule_sha256"],
            "raw_path": str(path),
        })
        block_bindings.append({
            "key": key,
            "path": str(path),
            "sha256": entry["sha256"],
            "foreground_arm": arm.value,
            "replicate": replicate,
        })

    gate = _validate_gate(parent.get("gate"))
    recomputed_gate = client._paired_gate(paired_gate_blocks)
    _require(gate == recomputed_gate, "C4 gate differs from recomputed evidence")
    _require(node.get("paired_output_count") == gate["paired_output_count"],
             "C4 node/gate paired output count differs")
    foreground_pairs = _foreground_paired_samples(block_paths, contracts)
    _require(len(foreground_pairs) == gate["paired_output_count"],
             "C4 foreground paired-sample count differs")
    _require(len(endpoint_rows) == 96,
             "C4 endpoint-phase inventory must contain 96 rows")
    _require(len(request_rows) == 96,
             "C4 request phase/tenant inventory must contain 96 rows")

    output: dict[str, object] = {
        "schema": SCHEMA,
        "source_node_result": {
            "path": str(result_path),
            "sha256": expected_result_sha256,
        },
        "source_client_raw": {
            "path": str(client_raw_path),
            "sha256": node["raw_sha256"],
        },
        "phase_manifest": {
            "path": str(manifest_path),
            "sha256": node["phase_manifest_sha256"],
            "fingerprint_sha256": node["phase_manifest_fingerprint_sha256"],
        },
        "implementation_contract": {
            "path": str(implementation_path),
            "sha256": node["implementation_contract_sha256"],
            "fingerprint_sha256": contract["fingerprint_sha256"],
            "file_count": len(contract["files"]),
        },
        "elastic_profile": {
            "path": str(elastic_profile),
            "sha256": node["elastic_profile_sha256"],
        },
        "source_workload": {
            "path": str(source_workload),
            "sha256": node["source_workload_sha256"],
        },
        "slurm_job_id": node["slurm_job_id"],
        "block_artifacts": block_bindings,
        "endpoint_phase_rows": endpoint_rows,
        "request_phase_tenant_rows": request_rows,
        "foreground_paired_samples": foreground_pairs,
        "fixed_gate": {
            "paired_output_count": gate["paired_output_count"],
            "service_rows": gate["service_rows"],
            "phase_service_rows": gate["phase_service_rows"],
            "phase_route_summaries": gate["phase_route_summaries"],
        },
        "invariants": {
            "blocks": 4,
            "phases_per_block": 6,
            "endpoints_per_phase": 4,
            "endpoint_phase_rows": 96,
            "request_phase_tenant_rows": 96,
            "foreground_paired_samples": len(foreground_pairs),
            "vllm_window": "same_endpoint_boundary_to_boundary_cumulative_delta",
            "cassini_windows": "same_endpoint_midpoint_and_end_half_deltas",
            "phase_boundary_completed_before_next_phase_first_arrival": True,
            "all_streams_and_cache_source_decisions_valid": True,
            "all_provenance_hashes_and_fingerprints_valid": True,
            "gate_recomputed_exact": True,
            "cross_endpoint_clock_subtraction": False,
            "synthetic_network_background": False,
            "unchanged_pd_data_plane": True,
        },
        "authorizes_profile_fit": True,
        "profile_fit_scope": "calibration_only",
        "authorizes_controller_parameter_search": False,
        "authorizes_live_validation": False,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "interpretation_boundary": {
            "endpoint_counters_identify_endpoint_local_symptoms_not_a_physical_switch": True,
            "vllm_phase_deltas_are_boundary_window_activity_not_request_attribution": True,
            "zero_pause_or_ecn_proves_uncongested_fabric": False,
            "final_speedup_claim_requires_frozen_independent_validation": True,
        },
    }
    output["fingerprint_sha256"] = _analysis_fingerprint(output)
    return output


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="node-0 C4 result.json")
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    _require(not args.output.exists(), "refusing to overwrite C4 analysis")
    result = analyze(
        args.input, expected_result_sha256=args.expected_input_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "fingerprint_sha256": result["fingerprint_sha256"],
        "authorizes_profile_fit": result["authorizes_profile_fit"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
