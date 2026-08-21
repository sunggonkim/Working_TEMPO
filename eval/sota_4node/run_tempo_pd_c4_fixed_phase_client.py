#!/usr/bin/env python3
"""Run the frozen C4 fixed local/remote phase trace with real cache states.

This client never starts servers.  It runs inside the existing four-node
actual-vLLM lifecycle and performs all cache preparation before endpoint
``before`` snapshots.  The measured order is local/remote then remote/local.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Mapping
import urllib.request

from eval.sota_4node import build_tempo_pd_c4_phase_manifest as manifest_builder
from eval.sota_4node import run_tempo_pd_contention_fixed_client as fixed
from eval.sota_4node import (
    run_tempo_pd_elastic_stream_metrics_cache_protocol as protocol_client,
)
from tempo.pd_cache_state_protocol import (
    CachePreparationPlan,
    CacheProtocolItem,
    build_cache_preparation_plan,
)
from tempo.pd_contention_workload import (
    CacheState,
    ContentionState,
    ForegroundArm,
    LoadSelection,
    Tenant,
    TrafficShape,
    VALIDATION_FOREGROUND_GEOMETRIES,
    build_schedule,
    semantic_schedule_sha256,
)
from tempo.pd_decoder_cache_evidence import (
    EVIDENCE_SOURCE as DECODER_CACHE_EVIDENCE_SOURCE,
    full_prefix_hit_tokens,
)


SCHEMA = "tempo-pd-c4-fixed-phase-client-v1"
BLOCK_SCHEMA = "tempo-pd-c4-fixed-phase-block-v1"
ENDPOINT_EVIDENCE_SCHEMA = "tempo-pd-c4-endpoint-evidence-v2"
ENDPOINT_SAMPLING_POLICY = (
    "workload_start_boundary_midpoint_and_end_boundary")
RUNTIME_EVIDENCE_SCHEMA = protocol_client.RUNTIME_EVIDENCE_SCHEMA
MANIFEST_ENV = "TEMPO_PD_C4_PHASE_MANIFEST"
SOURCE_MODULE = "eval.sota_4node.run_tempo_pd_elastic_stream_metrics"
PROTOCOL_MODULE = (
    "eval.sota_4node.run_tempo_pd_elastic_stream_metrics_cache_protocol")
FOREGROUND_POOL_SIZE = 4
KV_REMOTE_POOL_SIZE = 32
BLOCK_ORDER = (
    (ForegroundArm.LOCAL, 0),
    (ForegroundArm.REMOTE, 0),
    (ForegroundArm.REMOTE, 1),
    (ForegroundArm.LOCAL, 1),
)
_STATE_MARKER = {
    CacheState.MISS: "cache-miss-measured",
    CacheState.P_ONLY: "cache-p-only-measured",
    CacheState.D_ONLY: "cache-d-only-measured",
    CacheState.BOTH: "cache-both-measured",
}
_DECISION_STATE = {
    CacheState.MISS: "miss",
    CacheState.P_ONLY: "prefill_only",
    CacheState.D_ONLY: "decode_only",
    CacheState.BOTH: "prefill_and_decode",
}
_LOCAL_ROUTE = "decoder_local_chunked_prefill"
_REMOTE_ROUTE = "official_lmcache_remote_prefill"
_FRONTEND_SCHEMA = "tempo-elastic-pd-frontend-canonical-replicated-affinity-3"
_SHARED_CAPTURE_STAGE = {
    "before_process_start": "before",
    "measurement_start": "before",
    "phase_midpoint": "midpoint",
    "phase_boundary": "after",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    _require(resolved.is_file(), f"C4 artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
    }


def _reset_decoder_prefix_cache(base_url: str) -> dict[str, object]:
    _require(
        isinstance(base_url, str)
        and base_url.startswith(("http://", "https://")),
        "decoder reset URL must be explicit HTTP(S)",
    )
    request = urllib.request.Request(
        base_url.rstrip("/") + "/tempo/reset_decoder_prefix_cache",
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:
        value = json.loads(response.read())
    _require(
        isinstance(value, dict)
        and value.get("schema") == _FRONTEND_SCHEMA
        and value.get("success") is True
        and value.get("pair_decoder_resets") == 2
        and value.get("external_cache_reset") is False,
        "decoder APC reset evidence differs",
    )
    return value


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_manifest(path: Path) -> dict[str, object]:
    _require(path.is_absolute(), "C4 manifest path must be absolute")
    _require(path.is_file(), "C4 manifest is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("schema") == manifest_builder.SCHEMA,
        "C4 manifest schema differs",
    )
    _require(
        value.get("fingerprint_sha256")
        == manifest_builder.manifest_fingerprint(value),
        "C4 manifest fingerprint differs",
    )
    _require(value.get("performance_claim_allowed") is False,
             "C4 characterization manifest permits a performance claim")
    _require(value.get("controller_tuning_allowed") is False,
             "C4 manifest permits pre-characterization tuning")
    protocol = value.get("cache_state_protocol")
    route_contracts = (
        protocol.get("measured_decoder_route_contracts")
        if isinstance(protocol, dict) else None)
    local_contract = (
        route_contracts.get(_LOCAL_ROUTE)
        if isinstance(route_contracts, dict) else None)
    remote_contract = (
        route_contracts.get(_REMOTE_ROUTE)
        if isinstance(route_contracts, dict) else None)
    _require(
        isinstance(protocol, dict)
        and protocol.get("schema")
        == manifest_builder.CACHE_STATE_PROTOCOL_SCHEMA
        and protocol.get("fixed_arm_pair_placement")
        == "terminal_item_modulo_two_pairs"
        and protocol.get("decoder_usage_breakdown_required") is True
        and protocol.get(
            "stock_cached_tokens_without_source_breakdown_allowed") is False
        and protocol.get(
            "request_id_labels_without_completion_evidence_allowed") is False
        and isinstance(local_contract, dict)
        and local_contract.get("usage_prompt_tokens") == "P"
        and local_contract.get("external_cached_tokens") == 0
        and isinstance(remote_contract, dict)
        and remote_contract.get("usage_prompt_tokens") == "P+1"
        and remote_contract.get("decoder_residency_basis")
        == "exact_local_preparation_hit_on_original_P_token_prompt"
        and remote_contract.get("local_cached_tokens_by_state", {}).get(
            "d_only") == "floor((P-1)/16)*16"
        and remote_contract.get("local_cached_tokens_by_state", {}).get(
            "both") == "floor((P-1)/16)*16"
        and remote_contract.get("total_cached_tokens") == "P"
        and remote_contract.get("external_cached_tokens")
        == "P-local_cached_tokens",
        "C4 manifest lacks the completion-backed cache protocol",
    )
    endpoint_contract = value.get("endpoint_evidence_contract")
    _require(
        isinstance(endpoint_contract, dict)
        and endpoint_contract.get("schema")
        == manifest_builder.ENDPOINT_EVIDENCE_CONTRACT_SCHEMA
        and endpoint_contract.get("measurement_start_marker_required") is True
        and endpoint_contract.get(
            "publisher_pid_matches_measured_child") is True
        and endpoint_contract.get("measurement_clock")
        == "same_frontend_host_child_time_perf_counter_ns"
        and endpoint_contract.get("sampling_policy")
        == ENDPOINT_SAMPLING_POLICY
        and endpoint_contract.get("phase_boundary_samples")
        == len(manifest_builder.PHASES) + 1
        and endpoint_contract.get("phase_midpoint_samples")
        == len(manifest_builder.PHASES)
        and endpoint_contract.get("cassini_phase_windows")
        == "two_nonoverlapping_half_phase_endpoint_deltas"
        and endpoint_contract.get("vllm_phase_windows")
        == "boundary_to_boundary_cumulative_deltas"
        and endpoint_contract.get(
            "cross_host_clock_subtraction_allowed") is False,
        "C4 manifest lacks phase-aligned endpoint evidence",
    )
    _require(value.get("fixed_arm_order") == [
        "local", "remote", "remote", "local"],
        "C4 fixed arm order differs",
    )
    _require(value.get("replicates") == 2, "C4 replicate count differs")
    _require(
        value.get("fixed_runtime_environment")
        == dict(sorted(manifest_builder.C4_FIXED_RUNTIME_ENVIRONMENT.items())),
        "C4 fixed runtime environment differs",
    )
    return value


def _prompt_token_sha256(token_ids: list[int]) -> str:
    _require(bool(token_ids) and all(type(value) is int for value in token_ids),
             "prompt token IDs are invalid")
    return hashlib.sha256(json.dumps(
        token_ids, separators=(",", ":")).encode()).hexdigest()


class _PromptFactory:
    """Allocate bounded token-preserving prompts, sharing only declared keys."""

    def __init__(self, tokenizer, templates: Mapping[int, tuple[int, ...]]):
        self.tokenizer = tokenizer
        self.templates = templates
        self._next_marker = 100_000
        self._values: dict[tuple[object, ...], tuple[str, str]] = {}

    def prompt(
        self, key: tuple[object, ...], prompt_tokens: int,
    ) -> tuple[str, str]:
        prior = self._values.get(key)
        if prior is not None:
            return prior
        _require(prompt_tokens in self.templates,
                 "C4 prompt template geometry is missing")
        marker = self._next_marker
        self._next_marker += 1
        _require(marker < (1 << 18), "C4 prompt marker space exhausted")
        prompt = fixed._unique_prompt(
            self.tokenizer, self.templates[prompt_tokens], marker)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        _require(len(token_ids) == prompt_tokens,
                 "C4 prompt geometry changed")
        value = (prompt, _prompt_token_sha256(token_ids))
        self._values[key] = value
        return value


def _terminal_item(
    *, tenant: Tenant, ordinal: int, geometry_index: int,
    cache_state: CacheState,
) -> int:
    if tenant is Tenant.KV_REMOTE_HOT:
        return ordinal % KV_REMOTE_POOL_SIZE
    if tenant is Tenant.FOREGROUND and cache_state in {
        CacheState.P_ONLY, CacheState.BOTH,
    }:
        return geometry_index * FOREGROUND_POOL_SIZE + (
            ordinal // len(VALIDATION_FOREGROUND_GEOMETRIES)
        ) % FOREGROUND_POOL_SIZE
    return ordinal


def _prompt_key(
    *, sequence: int, replicate: int, tenant: Tenant, ordinal: int,
    geometry_index: int, cache_state: CacheState, terminal_item: int,
) -> tuple[object, ...]:
    if tenant is Tenant.KV_REMOTE_HOT:
        return ("kv_remote_pool", terminal_item)
    if tenant is Tenant.FOREGROUND:
        if cache_state in {CacheState.P_ONLY, CacheState.BOTH}:
            # Shared across the paired local/remote blocks, but cache_salt keeps
            # their physical arm namespaces disjoint.
            return (
                "foreground_pool", replicate, geometry_index, terminal_item)
        return ("foreground_unique", replicate, geometry_index, ordinal)
    # Cold background prompts must not survive into the paired block.
    return ("background_unique", sequence, tenant.value, ordinal)


def _request_id(
    *, sequence: int, arm: ForegroundArm, replicate: int,
    phase: ContentionState,
    tenant: Tenant, ordinal: int, state: CacheState, terminal_item: int,
) -> str:
    endpoint_marker = (
        "endpoint-observed-" if tenant is not Tenant.FOREGROUND else "")
    return (
        f"epd-{arm.value}-c4-b{sequence}-{_STATE_MARKER[state]}-r{replicate}-"
        f"{phase.value}-{tenant.value}-{endpoint_marker}occ-{ordinal:06d}-"
        f"item-{terminal_item:06d}"
    )


def _materialize_block(
    *, sequence: int, arm: ForegroundArm, replicate: int,
    manifest: Mapping[str, object], factory: _PromptFactory,
) -> dict[str, object]:
    rates = manifest["background_rates_per_s"]
    _require(isinstance(rates, Mapping), "C4 background rates are missing")
    selection = LoadSelection(
        decoder_reference_rate_per_s=float(rates["decoder_hot"]),
        remote_reference_rate_per_s=float(rates["cold_remote_hot"]),
        decoder_fraction=1.0,
        remote_fraction=1.0,
        kv_remote_rate_per_s=float(rates["kv_remote_hot"]),
    )
    schedule = build_schedule(
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
    rows: list[dict[str, object]] = []
    items: list[CacheProtocolItem] = []
    request_index: dict[str, dict[str, object]] = {}
    for request in schedule:
        geometry = request.geometry
        geometry_index = (
            VALIDATION_FOREGROUND_GEOMETRIES.index(geometry)
            if request.tenant is Tenant.FOREGROUND else -1
        )
        terminal_item = _terminal_item(
            tenant=request.tenant,
            ordinal=request.ordinal,
            geometry_index=geometry_index,
            cache_state=geometry.cache_state,
        )
        key = _prompt_key(
            sequence=sequence,
            replicate=replicate,
            tenant=request.tenant,
            ordinal=request.ordinal,
            geometry_index=geometry_index,
            cache_state=geometry.cache_state,
            terminal_item=terminal_item,
        )
        prompt, prompt_key = factory.prompt(key, geometry.prompt_tokens)
        request_id = _request_id(
            sequence=sequence,
            arm=request.arm,
            replicate=replicate,
            phase=request.phase,
            tenant=request.tenant,
            ordinal=request.ordinal,
            state=geometry.cache_state,
            terminal_item=terminal_item,
        )
        row = {
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": geometry.output_tokens,
            "arrival_offset_ms": round(request.arrival_offset_ms, 6),
        }
        rows.append(row)
        items.append(CacheProtocolItem(
            request_id=request_id,
            prompt=prompt,
            prompt_token_sha256=prompt_key,
            prompt_tokens=geometry.prompt_tokens,
            output_tokens=geometry.output_tokens,
            cache_state=geometry.cache_state,
            terminal_item=terminal_item,
        ))
        request_index[request_id] = {
            **request.semantic_dict(),
            "arm": request.arm.value,
            "prompt_token_sha256": prompt_key,
            "terminal_item": terminal_item,
            "pair_key": (
                f"r{replicate}:{request.phase.value}:"
                f"foreground:{request.ordinal:06d}"
                if request.tenant is Tenant.FOREGROUND else None
            ),
        }
    rows.sort(key=lambda row: (
        float(row["arrival_offset_ms"]), str(row["request_id"])))
    return {
        "sequence": sequence,
        "arm": arm,
        "replicate": replicate,
        "schedule_sha256": semantic_schedule_sha256(schedule),
        "rows": rows,
        "items": items,
        "request_index": request_index,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ), encoding="utf-8")


def _load_templates(
    path: Path, tokenizer,
) -> dict[int, tuple[int, ...]]:
    _require(path.is_file(), "explicit C4 source workload is missing")
    templates: dict[int, tuple[int, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt")
        _require(isinstance(prompt, str) and prompt,
                 "C4 source prompt is invalid")
        token_ids = tuple(tokenizer.encode(
            prompt, add_special_tokens=False))
        templates.setdefault(len(token_ids), token_ids)
    required = {
        geometry.prompt_tokens for geometry in VALIDATION_FOREGROUND_GEOMETRIES
    } | {4094}
    missing = sorted(required - set(templates))
    _require(not missing, f"C4 source workload lacks templates: {missing}")
    return {length: templates[length] for length in sorted(required)}


def _stream_command(
    args: argparse.Namespace, *, module: str, workload: Path,
    output: Path, run_id: str, max_workers: int,
) -> list[str]:
    command = [
        sys.executable, "-m", module,
        "--base-url", args.base_url,
        "--model", str(args.model),
        "--served-model-name", args.served_model_name,
        "--workload", str(workload),
        "--output", str(output),
        "--mode", "tempo_auto",
        "--run-id", run_id,
        "--default-max-tokens", str(args.default_max_tokens),
        "--max-workers", str(max_workers),
        "--timeout-s", str(args.timeout_s),
        "--seed", str(args.seed),
    ]
    if args.api_key_env:
        command.extend(("--api-key-env", args.api_key_env))
    return command


def _run(command: list[str], *, env: Mapping[str, str] | None = None) -> None:
    completed = subprocess.run(command, env=env, check=False, timeout=1200.0)
    _require(completed.returncode == 0,
             f"cache-protocol child returned {completed.returncode}")


def _capture_c4_endpoint_evidence(
    urls: list[str], *, stage: str, require_valid_delta: bool,
) -> dict[str, object]:
    """Adapt C4 phase labels to the older three-stage capture helper."""

    _require(stage in _SHARED_CAPTURE_STAGE,
             "C4 endpoint evidence stage is invalid")
    shared_stage = _SHARED_CAPTURE_STAGE[stage]
    sample = fixed._capture_endpoint_evidence(
        urls, stage=shared_stage, require_valid_delta=require_valid_delta)
    _require(sample.get("stage") == shared_stage,
             "shared endpoint capture stage differs")
    sample = dict(sample)
    sample["stage"] = stage
    return sample


def _terminate_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=10.0)


def _run_with_endpoint_evidence(
    command: list[str], *, args: argparse.Namespace,
    env: Mapping[str, str], start_marker: Path,
    first_arrival_offset_ms: float,
) -> dict[str, object]:
    _require(start_marker.is_absolute(),
             "C4 measurement start marker must be absolute")
    _require(start_marker.parent.is_dir() and not start_marker.exists(),
             "C4 measurement start marker path is invalid")
    _require(
        math.isfinite(first_arrival_offset_ms)
        and first_arrival_offset_ms > 0.0,
        "C4 first arrival offset must be finite and positive",
    )
    before_process_start = _capture_c4_endpoint_evidence(
        args.endpoint_evidence_url,
        stage="before_process_start",
        require_valid_delta=False,
    )
    child_env = dict(env)
    child_env[protocol_client.START_MARKER_ENV] = str(start_marker)
    child = subprocess.Popen(command, env=child_env)
    try:
        marker_deadline = time.monotonic() + min(
            float(args.timeout_s), 120.0)
        while not start_marker.is_file():
            return_code = child.poll()
            _require(return_code is None,
                     "C4 child exited before publishing its workload start")
            _require(time.monotonic() < marker_deadline,
                     "C4 child did not publish its workload start in time")
            time.sleep(0.01)
        marker = json.loads(start_marker.read_text(encoding="utf-8"))
        _require(
            isinstance(marker, dict)
            and set(marker) == {
                "schema", "clock", "run_start_ns", "publisher_pid"}
            and marker.get("schema") == protocol_client.START_MARKER_SCHEMA
            and marker.get("clock") == "client time.perf_counter_ns"
            and type(marker.get("run_start_ns")) is int
            and int(marker["run_start_ns"]) > 0
            and type(marker.get("publisher_pid")) is int
            and int(marker["publisher_pid"]) > 0
            and int(marker["publisher_pid"]) == child.pid,
            "C4 workload-start marker is invalid",
        )
        started_ns = int(marker["run_start_ns"])
        observed_ns = time.perf_counter_ns()
        _require(started_ns <= observed_ns,
                 "C4 workload-start marker is from the future")

        measurement_start = _capture_c4_endpoint_evidence(
            args.endpoint_evidence_url,
            stage="measurement_start",
            require_valid_delta=True,
        )
        measurement_start.update({
            "boundary_index": 0,
            "completed_phase": None,
            "begins_phase": manifest_builder.PHASES[0].value,
        })
        received_ns = max(
            int(row["client_received_monotonic_ns"])
            for row in measurement_start["snapshots"]
        )
        first_arrival_offset_ns = int(first_arrival_offset_ms * 1_000_000)
        start_capture_completed_offset_ns = received_ns - started_ns
        _require(
            0 <= start_capture_completed_offset_ns < first_arrival_offset_ns,
            "C4 measurement-start snapshot overlapped the first request",
        )
    except BaseException:
        _terminate_child(child)
        raise

    phase_boundaries = [measurement_start]
    phase_midpoints = []
    try:
        for phase_index, phase in enumerate(manifest_builder.PHASES):
            midpoint_target_ns = started_ns + int(
                (phase_index + 0.5) * args.phase_duration_ms * 1_000_000)
            remaining_s = (
                midpoint_target_ns - time.perf_counter_ns()) / 1_000_000_000
            if remaining_s > 0.0:
                time.sleep(remaining_s)
            _require(child.poll() is None,
                     "C4 child exited before a phase midpoint sample")
            sample = _capture_c4_endpoint_evidence(
                args.endpoint_evidence_url,
                stage="phase_midpoint",
                require_valid_delta=True,
            )
            sample["phase"] = phase.value
            sample["phase_index"] = phase_index
            phase_midpoints.append(sample)

            boundary_target_ns = started_ns + int(
                (phase_index + 1.0)
                * args.phase_duration_ms * 1_000_000)
            remaining_s = (
                boundary_target_ns - time.perf_counter_ns()) / 1_000_000_000
            if remaining_s > 0.0:
                time.sleep(remaining_s)
            if phase_index + 1 < len(manifest_builder.PHASES):
                _require(child.poll() is None,
                         "C4 child exited before a phase boundary sample")
            boundary = _capture_c4_endpoint_evidence(
                args.endpoint_evidence_url,
                stage="phase_boundary",
                require_valid_delta=True,
            )
            boundary.update({
                "boundary_index": phase_index + 1,
                "completed_phase": phase.value,
                "begins_phase": (
                    manifest_builder.PHASES[phase_index + 1].value
                    if phase_index + 1 < len(manifest_builder.PHASES)
                    else None
                ),
            })
            phase_boundaries.append(boundary)
        return_code = child.wait(timeout=1200.0)
        _require(return_code == 0, f"C4 child returned {return_code}")
    except BaseException:
        _terminate_child(child)
        raise
    evidence = {
        "schema": ENDPOINT_EVIDENCE_SCHEMA,
        "sampling_policy": ENDPOINT_SAMPLING_POLICY,
        "cross_endpoint_clock_subtraction_allowed": False,
        "measurement_clock_alignment": (
            "same_frontend_host_child_time_perf_counter_ns_marker"),
        "before_process_start": before_process_start,
        "measurement_start_marker": {
            **marker,
            "path": str(start_marker),
            "sha256": _sha256(start_marker),
            "parent_observed_child_pid": child.pid,
            "parent_observed_offset_ns": observed_ns - started_ns,
        },
        "first_arrival_offset_ns": first_arrival_offset_ns,
        "measurement_start_capture_completed_offset_ns": (
            start_capture_completed_offset_ns),
        "phase_boundaries": phase_boundaries,
        "phase_midpoints": phase_midpoints,
    }
    _validate_c4_endpoint_evidence(evidence)
    return evidence


def _validate_c4_endpoint_evidence(raw: object) -> None:
    _require(isinstance(raw, dict), "C4 endpoint evidence is not an object")
    _require(raw.get("schema") == ENDPOINT_EVIDENCE_SCHEMA,
             "C4 endpoint evidence schema differs")
    _require(raw.get("sampling_policy") == ENDPOINT_SAMPLING_POLICY,
             "C4 endpoint sampling policy differs")
    _require(raw.get("cross_endpoint_clock_subtraction_allowed") is False,
             "C4 endpoint evidence permits cross-host clock subtraction")
    _require(
        raw.get("measurement_clock_alignment")
        == "same_frontend_host_child_time_perf_counter_ns_marker",
        "C4 endpoint clock is not workload-start aligned",
    )
    marker = raw.get("measurement_start_marker")
    _require(isinstance(marker, dict),
             "C4 workload-start marker evidence is missing")
    marker_path = Path(str(marker.get("path", "")))
    _require(
        marker.get("schema") == protocol_client.START_MARKER_SCHEMA
        and marker.get("clock") == "client time.perf_counter_ns"
        and type(marker.get("run_start_ns")) is int
        and type(marker.get("publisher_pid")) is int
        and type(marker.get("parent_observed_child_pid")) is int
        and marker.get("publisher_pid")
        == marker.get("parent_observed_child_pid")
        and type(marker.get("parent_observed_offset_ns")) is int
        and int(marker["parent_observed_offset_ns"]) >= 0
        and marker_path.is_absolute()
        and marker_path.is_file()
        and marker.get("sha256") == _sha256(marker_path),
        "C4 workload-start marker evidence differs",
    )
    first_arrival = raw.get("first_arrival_offset_ns")
    start_capture = raw.get("measurement_start_capture_completed_offset_ns")
    _require(
        type(first_arrival) is int
        and first_arrival > 0
        and type(start_capture) is int
        and 0 <= start_capture < first_arrival,
        "C4 start snapshot overlaps measured traffic",
    )
    boundaries = raw.get("phase_boundaries")
    midpoints = raw.get("phase_midpoints")
    _require(
        isinstance(boundaries, list)
        and len(boundaries) == len(manifest_builder.PHASES) + 1,
        "C4 phase boundary evidence count differs",
    )
    _require(isinstance(midpoints, list)
             and len(midpoints) == len(manifest_builder.PHASES),
             "C4 phase midpoint evidence count differs")
    expected_ids = {
        "pair0-prefill", "pair0-decoder",
        "pair1-prefill", "pair1-decoder",
    }

    def validate_sample(
        sample: object, *, stage: str, require_valid_delta: bool,
    ) -> dict[str, object]:
        _require(
            isinstance(sample, dict)
            and sample.get("schema") == fixed.ENDPOINT_EVIDENCE_SCHEMA
            and sample.get("stage") == stage,
            "C4 endpoint sample schema or stage differs",
        )
        snapshots = sample.get("snapshots")
        _require(isinstance(snapshots, list) and len(snapshots) == 4,
                 "C4 endpoint sample count differs")
        identities = {
            row["probe"]["endpoint"]["endpoint_id"] for row in snapshots
        }
        _require(identities == expected_ids,
                 "C4 endpoint identity set differs")
        if require_valid_delta:
            _require(all(
                row["probe"]["cassini"].get("valid") is True
                for row in snapshots
            ), "C4 measured Cassini delta is invalid")
        return sample

    before = validate_sample(
        raw.get("before_process_start"),
        stage="before_process_start", require_valid_delta=False)
    first_boundary = validate_sample(
        boundaries[0], stage="measurement_start", require_valid_delta=True)
    _require(
        first_boundary.get("boundary_index") == 0
        and first_boundary.get("completed_phase") is None
        and first_boundary.get("begins_phase")
        == manifest_builder.PHASES[0].value,
        "C4 measurement-start boundary differs",
    )
    for index, (sample, phase) in enumerate(zip(
        midpoints, manifest_builder.PHASES, strict=True,
    )):
        sample = validate_sample(
            sample, stage="phase_midpoint", require_valid_delta=True)
        _require(
            sample.get("phase_index") == index
            and sample.get("phase") == phase.value,
            "C4 phase midpoint endpoint evidence differs",
        )
        boundary = validate_sample(
            boundaries[index + 1],
            stage="phase_boundary", require_valid_delta=True)
        _require(
            boundary.get("boundary_index") == index + 1
            and boundary.get("completed_phase") == phase.value
            and boundary.get("begins_phase") == (
                manifest_builder.PHASES[index + 1].value
                if index + 1 < len(manifest_builder.PHASES) else None
            ),
            "C4 phase boundary endpoint evidence differs",
        )

    histories: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    chronological = [before, first_boundary]
    for index in range(len(manifest_builder.PHASES)):
        chronological.extend((midpoints[index], boundaries[index + 1]))
    for sample in chronological:
        for row in sample["snapshots"]:
            endpoint = row["probe"]["endpoint"]
            cassini = row["probe"]["cassini"]
            histories[endpoint["endpoint_id"]].append((
                int(endpoint["sequence"]), int(cassini["sequence"])))
    _require(set(histories) == expected_ids,
             "C4 endpoint history identity set differs")
    for history in histories.values():
        endpoint_sequences = [item[0] for item in history]
        cassini_sequences = [item[1] for item in history]
        _require(
            endpoint_sequences == sorted(set(endpoint_sequences))
            and cassini_sequences == sorted(set(cassini_sequences)),
            "C4 endpoint or Cassini sequence did not increase",
        )


def _artifact_rows(
    raw_path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]],
           dict[str, dict[str, object]]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(requests, list) and isinstance(decisions, list),
             "cache-protocol artifact rows are missing")
    request_index = {str(row.get("request_id")): row for row in requests}
    decision_index = {str(row.get("request_id")): row for row in decisions}
    _require(len(request_index) == len(requests),
             "cache-protocol requests contain duplicate IDs")
    _require(len(decision_index) == len(decisions),
             "cache-protocol decisions contain duplicate IDs")
    return raw, request_index, decision_index


def validate_source_preparation(
    raw_path: Path, plan: CachePreparationPlan,
) -> dict[str, object]:
    _raw, requests, decisions = _artifact_rows(raw_path)
    expected = {
        str(row["request_id"]) for row in plan.source_probe_rows}
    _require(set(requests) == expected and set(decisions) == expected,
             "source preparation IDs differ from the plan")
    source_rows = {
        str(row["request_id"]): row for row in plan.source_probe_rows}
    for request_id in sorted(expected):
        request = requests[request_id]
        decision = decisions[request_id]
        plan_items = _plan_row_items(plan, source_rows[request_id])
        prompt_tokens = _plan_row_prompt_tokens(plan, source_rows[request_id])
        terminal_items = {item.terminal_item for item in plan_items}
        _require(len(terminal_items) == 1,
                 "source preparation terminal item is ambiguous")
        expected_pair = next(iter(terminal_items)) % 2
        seed = request.get("p_only_cache_seed")
        expected_seed_id = request_id.replace(
            "-warm-", "-warm-seed-o2-", 1)
        _require(
            request.get("valid") is True
            and isinstance(seed, dict)
            and seed.get("valid") is True
            and seed.get("request_id") == expected_seed_id
            and seed.get("route") == _REMOTE_ROUTE,
            "source seed completion evidence is invalid",
        )
        _require(
            decision.get("route") == _REMOTE_ROUTE
            and decision.get("completion_cache_residency") == "prefill_only"
            and decision.get("lmcache_source_full_hit_observed") is True
            and decision.get("decoder_prefix_cached_tokens") == 0
            and decision.get("decoder_total_cached_tokens") == prompt_tokens
            and decision.get("decoder_external_cached_tokens") == prompt_tokens
            and decision.get("decoder_prefix_usage_prompt_tokens")
            == prompt_tokens + 1
            and decision.get("decoder_prefix_cache_evidence_source")
            == DECODER_CACHE_EVIDENCE_SOURCE
            and decision.get("frontend_pair_policy") == "item_modulo_v1"
            and decision.get("frontend_pair_index") == expected_pair,
            "source probe did not establish a full P-side hit",
        )
    return {
        "raw": str(raw_path.resolve()),
        "sha256": _sha256(raw_path),
        "requests": len(expected),
        "all_seed_misses_and_probe_full_hits_exact": True,
    }


def _plan_row_items(
    plan: CachePreparationPlan, row: Mapping[str, object],
) -> tuple[CacheProtocolItem, ...]:
    request_id = str(row["request_id"])
    prompt = str(row["prompt"])
    matches = tuple(
        item for item in plan.items
        if request_id.startswith(f"epd-{item.arm}-")
        and item.prompt == prompt
    )
    _require(bool(matches), "cache preparation row lacks a measured namespace")
    return matches


def _plan_row_prompt_tokens(
    plan: CachePreparationPlan, row: Mapping[str, object],
) -> int:
    values = {item.prompt_tokens for item in _plan_row_items(plan, row)}
    _require(len(values) == 1,
             "cache preparation prompt geometry is ambiguous")
    return next(iter(values))


def _decoder_expected(plan: CachePreparationPlan) -> dict[str, CacheState]:
    result: dict[str, CacheState] = {}
    for row in plan.decoder_prepare_rows:
        request_id = str(row["request_id"])
        matches = {item.cache_state for item in _plan_row_items(plan, row)}
        _require(len(matches) == 1,
                 "decoder preparation state lookup is ambiguous")
        result[request_id] = next(iter(matches))
    return result


def validate_decoder_preparation(
    raw_path: Path, plan: CachePreparationPlan,
) -> dict[str, object]:
    _raw, requests, decisions = _artifact_rows(raw_path)
    expected_ids = [
        str(row["request_id"]) for row in plan.decoder_prepare_rows]
    _require(set(requests) == set(expected_ids)
             and set(decisions) == set(expected_ids),
             "decoder preparation IDs differ from the plan")
    expected_state = _decoder_expected(plan)
    plan_rows = {
        str(row["request_id"]): row for row in plan.decoder_prepare_rows}
    for index in range(0, len(expected_ids), 2):
        seed_id, probe_id = expected_ids[index:index + 2]
        seed = decisions[seed_id]
        probe = decisions[probe_id]
        state = expected_state[probe_id]
        prompt_tokens = _plan_row_prompt_tokens(plan, plan_rows[probe_id])
        expected_hit = full_prefix_hit_tokens(prompt_tokens)
        _require(requests[seed_id].get("valid") is True
                 and requests[probe_id].get("valid") is True,
                 "decoder seed/probe stream is invalid")
        _require(
            seed.get("route") == _LOCAL_ROUTE
            and seed.get("decoder_prefix_cached_tokens") == 0
            and seed.get("decoder_total_cached_tokens") == 0
            and seed.get("decoder_external_cached_tokens") == 0
            and seed.get("decoder_prefix_usage_prompt_tokens") == prompt_tokens
            and seed.get("decoder_prefix_cache_evidence_source")
            == DECODER_CACHE_EVIDENCE_SOURCE
            and seed.get("decoder_prefix_read_skipped") is True,
            "decoder seed is not an exact local APC miss",
        )
        _require(
            probe.get("route") == _LOCAL_ROUTE
            and probe.get("decoder_prefix_full_hit_observed") is True
            and probe.get("decoder_prefix_cached_tokens") == expected_hit
            and probe.get("decoder_total_cached_tokens") == expected_hit
            and probe.get("decoder_external_cached_tokens") == 0
            and probe.get("decoder_prefix_usage_prompt_tokens") == prompt_tokens
            and probe.get("decoder_prefix_cache_evidence_source")
            == DECODER_CACHE_EVIDENCE_SOURCE
            and probe.get("decoder_prefix_read_skipped") is False
            and probe.get("completion_cache_residency")
            == _DECISION_STATE[state],
            "decoder probe did not establish the requested cache state",
        )
        _require(
            type(seed.get("frontend_pair_index")) is int
            and seed.get("frontend_pair_index")
            == probe.get("frontend_pair_index")
            and seed.get("frontend_pair_policy") == "item_modulo_v1"
            and probe.get("frontend_pair_policy") == "item_modulo_v1"
            and seed.get("frontend_pair_decode_affinity_required") is False
            and probe.get("frontend_pair_decode_affinity_required") is False
            and seed.get("frontend_pair_decode_affinity_hit") is False
            and probe.get("frontend_pair_decode_affinity_hit") is False,
            "decoder seed/probe did not execute on one proven decoder pair",
        )
    return {
        "raw": str(raw_path.resolve()),
        "sha256": _sha256(raw_path),
        "requests": len(expected_ids),
        "all_seed_misses_and_probe_full_hits_exact": True,
        "same_decoder_pair_enforced_by_terminal_item_modulo": True,
    }


def _runtime_evidence(
    *, plan_path: Path, plan: CachePreparationPlan,
    source: dict[str, object], reset: dict[str, object],
    decoder: dict[str, object],
) -> dict[str, object]:
    _require(
        reset.get("success") is True
        and reset.get("pair_decoder_resets") == 2
        and reset.get("external_cache_reset") is False,
        "decoder APC reset evidence is invalid",
    )
    return {
        "schema": RUNTIME_EVIDENCE_SCHEMA,
        "plan": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "plan_fingerprint_sha256": plan.fingerprint_sha256,
        "source_preparation": source,
        "quiescent_decoder_apc_reset": reset,
        "decoder_preparation": decoder,
        "preparation_completed_before_measurement": True,
        "measurement_includes_preparation_requests": False,
        "ready_for_measurement": True,
    }


def _validate_measured_decision(
    decision: Mapping[str, object], metadata: Mapping[str, object],
) -> None:
    state = CacheState(str(metadata["cache_state"]))
    expected_route = (
        _LOCAL_ROUTE if metadata["arm"] == "local" else _REMOTE_ROUTE)
    _require(decision.get("route") == expected_route,
             "C4 measured route differs from the fixed arm")
    _require(decision.get("request_cache_contract") == state.value,
             "C4 request cache contract differs")
    _require(decision.get("decision_cache_residency") == _DECISION_STATE[state],
             "C4 decision cache residency differs")
    prompt_tokens = int(metadata["prompt_tokens"])
    terminal_item = metadata.get("terminal_item")
    _require(type(terminal_item) is int and terminal_item >= 0,
             "C4 terminal item is invalid")
    _require(
        decision.get("frontend_pair_policy") == "item_modulo_v1"
        and decision.get("frontend_pair_index") == terminal_item % 2,
        "C4 fixed-arm frontend pair placement differs",
    )
    expected_skip = state in {CacheState.MISS, CacheState.P_ONLY}
    expected_usage_prompt = prompt_tokens + int(
        expected_route == _REMOTE_ROUTE)
    expected_local = (
        0 if expected_skip
        else full_prefix_hit_tokens(prompt_tokens)
    )
    expected_total = (
        prompt_tokens if expected_route == _REMOTE_ROUTE
        else expected_local
    )
    expected_external = expected_total - expected_local
    _require(
        decision.get("decoder_prefix_read_skipped") is expected_skip
        and decision.get("decoder_prefix_cached_tokens") == expected_local
        and decision.get("decoder_total_cached_tokens") == expected_total
        and decision.get("decoder_external_cached_tokens")
        == expected_external
        and decision.get("decoder_prefix_usage_prompt_tokens")
        == expected_usage_prompt
        and decision.get("decoder_prefix_expected_full_hit_tokens")
        == full_prefix_hit_tokens(prompt_tokens)
        and decision.get("decoder_prefix_full_hit_observed")
        is (not expected_skip)
        and decision.get("decoder_prefix_cache_evidence_source")
        == DECODER_CACHE_EVIDENCE_SOURCE,
        "C4 decoder cache-source evidence differs",
    )
    if expected_route == _LOCAL_ROUTE:
        _require(expected_external == 0,
                 "C4 local route unexpectedly expects external KV")
    else:
        expected_source = (
            prompt_tokens
            if state in {CacheState.P_ONLY, CacheState.BOTH} else 0)
        _require(decision.get("lmcache_source_cached_tokens") == expected_source,
                 "C4 remote source-cache evidence differs")


def validate_measured_block(
    raw_path: Path, block: Mapping[str, object],
    endpoint_evidence: dict[str, object],
) -> dict[str, object]:
    raw, requests, decisions = _artifact_rows(raw_path)
    request_index = block["request_index"]
    _require(isinstance(request_index, Mapping), "C4 request index is missing")
    expected = set(request_index)
    _require(set(requests) == expected and set(decisions) == expected,
             "C4 measured artifact IDs differ")
    _require(all(row.get("valid") is True for row in requests.values()),
             "C4 measured stream is invalid")
    for request_id, metadata in request_index.items():
        _require(isinstance(metadata, Mapping), "C4 metadata is malformed")
        _validate_measured_decision(decisions[request_id], metadata)
    _validate_c4_endpoint_evidence(endpoint_evidence)
    raw["c4_fixed_phase_contract"] = {
        "schema": BLOCK_SCHEMA,
        "sequence": block["sequence"],
        "foreground_arm": block["arm"].value,
        "replicate": block["replicate"],
        "semantic_schedule_sha256": block["schedule_sha256"],
        "request_index": request_index,
        "all_requests_valid": True,
        "decision_cache_states_exact": True,
        "completion_cache_evidence_exact": True,
        "workload_start_marker_exact": True,
        "phase_aligned_endpoint_evidence": True,
        "preparation_outside_measurement": True,
        "actual_inference_background_only": True,
        "cross_endpoint_clock_subtraction_allowed": False,
    }
    raw["endpoint_evidence"] = endpoint_evidence
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return raw["c4_fixed_phase_contract"]


def _request_service_metrics(row: Mapping[str, object]) -> dict[str, float]:
    dispatch = row.get("dispatch_offset_ns")
    arrivals = row.get("token_arrival_offsets_ns")
    stream_end = row.get("stream_end_offset_ns")
    expected_tokens = row.get("requested_max_tokens")
    _require(type(dispatch) is int and dispatch >= 0,
             "C4 foreground dispatch clock is invalid")
    _require(
        isinstance(arrivals, list)
        and len(arrivals) >= 2
        and type(expected_tokens) is int
        and len(arrivals) == expected_tokens
        and all(type(value) is int for value in arrivals)
        and arrivals == sorted(arrivals)
        and arrivals[0] > dispatch,
        "C4 foreground token clocks are invalid",
    )
    _require(type(stream_end) is int and stream_end >= arrivals[-1],
             "C4 foreground completion clock is invalid")
    intervals = [
        (arrivals[index] - arrivals[index - 1]) / 1_000_000.0
        for index in range(1, len(arrivals))
    ]
    return {
        "ttft_ms": (arrivals[0] - dispatch) / 1_000_000.0,
        "e2e_ms": (stream_end - dispatch) / 1_000_000.0,
        "tpot_ms": statistics.median(intervals),
    }


def _paired_gate(blocks: list[dict[str, object]]) -> dict[str, object]:
    _require(len(blocks) == 4, "C4 gate requires exactly four ABBA blocks")
    by_replicate: dict[int, dict[str, dict[str, object]]] = (
        collections.defaultdict(dict))
    for block in blocks:
        by_replicate[int(block["replicate"])][block["arm"].value] = block
    _require(set(by_replicate) == {0, 1},
             "C4 gate replicate set differs")
    output_pairs = 0
    service_rows: dict[
        tuple[int, int, str, str], list[dict[str, float]]
    ] = collections.defaultdict(list)
    phase_service_rows: dict[
        tuple[str, int, int, str, str], list[dict[str, float]]
    ] = collections.defaultdict(list)
    phase_pairs: dict[
        tuple[str, int, int, str], list[dict[str, float]]
    ] = collections.defaultdict(list)
    for replicate in (0, 1):
        pair = by_replicate[replicate]
        _require(set(pair) == {"local", "remote"},
                 "C4 paired block is missing an arm")
        _require(pair["local"]["schedule_sha256"]
                 == pair["remote"]["schedule_sha256"],
                 "C4 paired semantic schedules differ")
        loaded: dict[str, dict[str, dict[str, object]]] = {}
        for arm, block in pair.items():
            raw = json.loads(Path(block["raw_path"]).read_text(encoding="utf-8"))
            contract = raw["c4_fixed_phase_contract"]
            _require(
                contract.get("all_requests_valid") is True
                and contract.get("completion_cache_evidence_exact") is True
                and contract.get("workload_start_marker_exact") is True
                and contract.get("phase_aligned_endpoint_evidence") is True,
                "C4 block lacks completion or phase-aligned evidence",
            )
            request_index = contract["request_index"]
            requests = {row["request_id"]: row for row in raw["requests"]}
            foreground: dict[str, dict[str, object]] = {}
            for request_id, metadata in request_index.items():
                if metadata["tenant"] != Tenant.FOREGROUND.value:
                    continue
                row = requests[request_id]
                pair_key = str(metadata["pair_key"])
                _require(
                    row.get("valid") is True
                    and row.get("requested_max_tokens")
                    == metadata["output_tokens"]
                    and pair_key != "None",
                    "C4 foreground request or pair key is invalid",
                )
                metrics = _request_service_metrics(row)
                foreground[pair_key] = {
                    "metadata": metadata,
                    "request": row,
                    "metrics": metrics,
                }
                geometry = (
                    int(metadata["prompt_tokens"]),
                    int(metadata["output_tokens"]),
                    str(metadata["cache_state"]),
                )
                service_rows[(*geometry, arm)].append(metrics)
                phase_service_rows[(
                    str(metadata["phase"]), *geometry, arm,
                )].append(metrics)
            loaded[arm] = foreground
        _require(set(loaded["local"]) == set(loaded["remote"]),
                 "C4 paired foreground keys differ")
        for key in loaded["local"]:
            local = loaded["local"][key]
            remote = loaded["remote"][key]
            local_request = local["request"]
            remote_request = remote["request"]
            local_metadata = local["metadata"]
            remote_metadata = remote["metadata"]
            _require(
                local_request["output_text_sha256"]
                == remote_request["output_text_sha256"],
                "C4 paired output digest differs",
            )
            semantic_fields = (
                "phase", "tenant", "prompt_tokens",
                "output_tokens", "cache_state", "ordinal",
            )
            _require(
                all(local_metadata[field] == remote_metadata[field]
                    for field in semantic_fields),
                "C4 paired foreground semantics differ",
            )
            geometry = (
                str(local_metadata["phase"]),
                int(local_metadata["prompt_tokens"]),
                int(local_metadata["output_tokens"]),
                str(local_metadata["cache_state"]),
            )
            local_metrics = local["metrics"]
            remote_metrics = remote["metrics"]
            phase_pairs[geometry].append({
                name: remote_metrics[name] - local_metrics[name]
                for name in ("ttft_ms", "e2e_ms", "tpot_ms")
            })
            output_pairs += 1

    def median_metric(rows: list[dict[str, float]], name: str) -> float:
        return statistics.median(row[name] for row in rows)

    summaries = []
    geometry_keys = {
        (geometry.prompt_tokens, geometry.output_tokens,
         geometry.cache_state.value)
        for geometry in VALIDATION_FOREGROUND_GEOMETRIES
    }
    for prompt_tokens, output_tokens, state in sorted(geometry_keys):
        local = service_rows[(prompt_tokens, output_tokens, state, "local")]
        remote = service_rows[(prompt_tokens, output_tokens, state, "remote")]
        _require(len(local) >= 2 and len(remote) >= 2,
                 "C4 service row lacks two samples per route")
        summaries.append({
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "cache_state": state,
            **{
                f"local_{name}_median_ms": median_metric(local, f"{name}_ms")
                for name in ("ttft", "e2e", "tpot")
            },
            **{
                f"remote_{name}_median_ms": median_metric(remote, f"{name}_ms")
                for name in ("ttft", "e2e", "tpot")
            },
            "samples_local": len(local),
            "samples_remote": len(remote),
        })

    phase_summaries = []
    phase_route_summaries = []
    for phase in manifest_builder.PHASES:
        all_phase_pairs = []
        for geometry in VALIDATION_FOREGROUND_GEOMETRIES:
            key = (
                phase.value, geometry.prompt_tokens, geometry.output_tokens,
                geometry.cache_state.value,
            )
            local = phase_service_rows[(*key, "local")]
            remote = phase_service_rows[(*key, "remote")]
            paired = phase_pairs[key]
            _require(
                len(local) >= 4
                and len(remote) >= 4
                and len(paired) == len(local) == len(remote),
                "C4 phase/geometry cell lacks two-replicate route samples",
            )
            all_phase_pairs.extend(paired)
            phase_summaries.append({
                "phase": phase.value,
                "prompt_tokens": geometry.prompt_tokens,
                "output_tokens": geometry.output_tokens,
                "cache_state": geometry.cache_state.value,
                **{
                    f"local_{name}_median_ms": median_metric(
                        local, f"{name}_ms")
                    for name in ("ttft", "e2e", "tpot")
                },
                **{
                    f"remote_{name}_median_ms": median_metric(
                        remote, f"{name}_ms")
                    for name in ("ttft", "e2e", "tpot")
                },
                **{
                    f"remote_minus_local_{name}_median_ms": statistics.median(
                        row[f"{name}_ms"] for row in paired)
                    for name in ("ttft", "e2e", "tpot")
                },
                "remote_e2e_faster_fraction": (
                    sum(row["e2e_ms"] < 0.0 for row in paired) / len(paired)),
                "paired_samples": len(paired),
            })
        phase_route_summaries.append({
            "phase": phase.value,
            **{
                f"remote_minus_local_{name}_median_ms": statistics.median(
                    row[f"{name}_ms"] for row in all_phase_pairs)
                for name in ("ttft", "e2e", "tpot")
            },
            "remote_e2e_faster_fraction": (
                sum(row["e2e_ms"] < 0.0 for row in all_phase_pairs)
                / len(all_phase_pairs)),
            "paired_samples": len(all_phase_pairs),
        })
    return {
        "all_blocks_valid": True,
        "paired_semantic_schedules_exact": True,
        "paired_output_digests_exact": True,
        "phase_aligned_endpoint_evidence": True,
        "phase_geometry_cells_complete": True,
        "paired_output_count": output_pairs,
        "service_rows": summaries,
        "phase_service_rows": phase_summaries,
        "phase_route_summaries": phase_route_summaries,
        "authorizes_endpoint_profile_fit": True,
        "performance_claim_allowed": False,
    }


def _measured(
    args: argparse.Namespace, tokenizer, templates,
    manifest_path: Path, manifest: dict[str, object],
) -> int:
    _require(os.environ.get("TEMPO_VLLM_DECODER_PREFIX_CACHING") == "1",
             "C4 requires decoder prefix caching")
    _require(os.environ.get(
        "TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY") == "1",
        "C4 requires replicated producer warm affinity")
    _require(len(args.endpoint_evidence_url) == 4,
             "C4 requires four endpoint evidence probes")
    _require(float(args.phase_duration_ms)
             == float(manifest["phase_duration_ms"]),
             "C4 phase duration differs from the manifest")
    root = args.output.parent / "c4_fixed_phase"
    workload_root = root / "workloads"
    root.mkdir()
    workload_root.mkdir()

    factory = _PromptFactory(tokenizer, templates)
    blocks = [
        _materialize_block(
            sequence=sequence, arm=arm, replicate=replicate,
            manifest=manifest, factory=factory)
        for sequence, (arm, replicate) in enumerate(BLOCK_ORDER)
    ]
    plan = build_cache_preparation_plan(
        item for block in blocks for item in block["items"])
    plan_path = root / "cache_preparation_plan.json"
    plan_path.write_text(
        json.dumps(plan.manifest_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_workload = workload_root / "source_prepare.jsonl"
    source_raw = root / "source_prepare.raw.json"
    _write_rows(source_workload, list(plan.source_probe_rows))
    _run(_stream_command(
        args, module=SOURCE_MODULE, workload=source_workload,
        output=source_raw, run_id=f"{args.run_id}-source-prepare",
        max_workers=1))
    source_evidence = validate_source_preparation(source_raw, plan)

    reset = _reset_decoder_prefix_cache(args.base_url)

    decoder_workload = workload_root / "decoder_prepare.jsonl"
    decoder_raw = root / "decoder_prepare.raw.json"
    _write_rows(decoder_workload, list(plan.decoder_prepare_rows))
    decoder_env = dict(os.environ)
    decoder_env[protocol_client.PHASE_ENV] = "decoder_prepare"
    decoder_env[protocol_client.PLAN_ENV] = str(plan_path.resolve())
    decoder_env.pop(protocol_client.EVIDENCE_ENV, None)
    _run(_stream_command(
        args, module=PROTOCOL_MODULE, workload=decoder_workload,
        output=decoder_raw, run_id=f"{args.run_id}-decoder-prepare",
        max_workers=1), env=decoder_env)
    decoder_evidence = validate_decoder_preparation(decoder_raw, plan)

    runtime_evidence = _runtime_evidence(
        plan_path=plan_path, plan=plan,
        source=source_evidence, reset=reset, decoder=decoder_evidence)
    evidence_path = root / "cache_runtime_evidence.json"
    evidence_path.write_text(
        json.dumps(runtime_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    measured_env = dict(os.environ)
    measured_env[protocol_client.PHASE_ENV] = "measured"
    measured_env[protocol_client.PLAN_ENV] = str(plan_path.resolve())
    measured_env[protocol_client.EVIDENCE_ENV] = str(evidence_path.resolve())
    artifacts: dict[str, dict[str, str]] = {}
    contracts: dict[str, dict[str, object]] = {}
    evidence_args = argparse.Namespace(**vars(args))
    evidence_args.phase_duration_ms = float(manifest["phase_duration_ms"])
    public_blocks: list[dict[str, object]] = []
    for block in blocks:
        key = (
            f"{block['sequence']:02d}_{block['arm'].value}_"
            f"r{block['replicate']}")
        workload = workload_root / f"{key}.jsonl"
        raw_path = root / f"{key}.raw.json"
        _write_rows(workload, block["rows"])
        endpoint_evidence = _run_with_endpoint_evidence(
            _stream_command(
                args, module=PROTOCOL_MODULE, workload=workload,
                output=raw_path, run_id=f"{args.run_id}-{key}",
                max_workers=args.max_workers,
            ),
            args=evidence_args,
            env=measured_env,
            start_marker=(root / f"{key}.measurement-start.json").resolve(),
            first_arrival_offset_ms=min(
                float(row["arrival_offset_ms"])
                for row in block["rows"]),
        )
        contract = validate_measured_block(
            raw_path, block, endpoint_evidence)
        artifacts[key] = _artifact_binding(raw_path)
        contracts[key] = contract
        public_blocks.append({**block, "raw_path": str(raw_path.resolve())})
        if int(block["sequence"]) + 1 < len(blocks):
            time.sleep(args.cooldown_s)

    gate = _paired_gate(public_blocks)
    public = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_fingerprint_sha256": manifest["fingerprint_sha256"],
        "cache_plan": str(plan_path.resolve()),
        "cache_plan_sha256": _sha256(plan_path),
        "cache_runtime_evidence": str(evidence_path.resolve()),
        "cache_runtime_evidence_sha256": _sha256(evidence_path),
        "block_order": [
            {"arm": arm.value, "replicate": replicate}
            for arm, replicate in BLOCK_ORDER
        ],
        "artifacts": artifacts,
        "contracts": contracts,
        "gate": gate,
        "performance_claim_allowed": False,
        "controller_tuning_allowed": gate["authorizes_endpoint_profile_fit"],
    }
    args.output.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


def main() -> int:
    args = fixed._parse()
    _require(args.mode == "tempo_auto", "C4 client requires tempo_auto")
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    _require(args.model.is_absolute(), "model path must be absolute")
    raw_manifest = os.environ.get(MANIFEST_ENV)
    _require(isinstance(raw_manifest, str) and raw_manifest,
             f"{MANIFEST_ENV} is required")
    manifest_path = Path(raw_manifest)
    manifest = _load_manifest(manifest_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True)
    templates = _load_templates(args.workload, tokenizer)
    if args.run_id.endswith("-warmup"):
        return fixed._warmup(args, tokenizer, templates)
    return _measured(
        args, tokenizer, templates, manifest_path.resolve(), manifest)


if __name__ == "__main__":
    raise SystemExit(main())
