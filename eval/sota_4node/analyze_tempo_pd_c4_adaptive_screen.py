#!/usr/bin/env python3
"""Analyze the post-C4 adaptive screen without authorizing a paper claim."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as fixed_analysis
from eval.sota_4node import build_tempo_pd_c4_adaptive_run_contract as contract_builder
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as manifest_builder
from eval.sota_4node import run_tempo_pd_c4_adaptive_screen_client as client
from eval.sota_4node import vllm_lmcache_pd_c4_adaptive_screen_node as node_module
from tempo.pd_contention_workload import (
    CacheState,
    ForegroundArm,
    Tenant,
    VALIDATION_FOREGROUND_GEOMETRIES,
)


SCHEMA = "tempo-pd-c4-adaptive-screen-analysis-v2"
NODE_SCHEMA = node_module.SCHEMA
MIN_MEAN_GAIN_VS_STRONGEST_FIXED = 0.03
MIN_MEAN_GAIN_VS_PREDICTOR = 0.02
MAX_P99_REGRESSION = 0.05
MIN_PAIRED_WIN_FRACTION = 0.55
REPO_ROOT = Path(__file__).resolve().parents[2]

_NODE_KEYS = frozenset({
    "schema", "raw", "raw_sha256", "run_contract", "run_contract_sha256",
    "run_contract_fingerprint_sha256", "adaptive_implementation_contract",
    "adaptive_implementation_contract_sha256",
    "adaptive_implementation_fingerprint_sha256",
    "adaptive_implementation_file_count", "source_workload",
    "source_workload_sha256", "phase_manifest", "phase_manifest_sha256",
    "elastic_profile", "elastic_profile_sha256",
    "endpoint_service_profile", "endpoint_service_profile_sha256",
    "slurm_job_id", "startup_readiness_timeout_s", "block_count",
    "block_artifacts", "tempo_both_routes_exercised",
    "fixed_runtime_environment", "transport_environment",
    "correctness_gate_pass", "calibration_only", "performance_claim_allowed",
    "physical_switch_bottleneck_claim_allowed",
    "independent_validation_required", "unchanged_pd_data_plane", "transport",
})
_CLIENT_KEYS = frozenset({
    "schema", "run_id", "run_contract", "run_contract_sha256", "manifest",
    "manifest_sha256", "cache_plan", "cache_plan_sha256",
    "cache_runtime_evidence", "cache_runtime_evidence_sha256", "block_order",
    "artifacts", "contracts", "summaries", "paired_output_gate",
    "blocks_completed", "live_screen_correctness_pass",
    "live_screen_route_diversity_pass", "calibration_only",
    "performance_claim_allowed", "physical_switch_bottleneck_claim_allowed",
    "unchanged_pd_data_plane",
})
_BLOCK_CONTRACT_KEYS = frozenset({
    "schema", "sequence", "arm", "replicate", "semantic_schedule_sha256",
    "request_index", "controller_generations", "all_requests_valid",
    "decision_cache_states_exact", "completion_cache_evidence_exact",
    "phase_aligned_endpoint_evidence", "controller_reset_before_block_exact",
    "controller_quiescent_after_block", "one_way_route_commit_exact",
    "unchanged_pd_data_plane", "performance_claim_allowed",
})
_SEMANTIC_BLOCK_CONTRACT_EXTRA_KEYS = frozenset({
    "endpoint_routing_policy",
    "endpoint_service_profile_fingerprint_sha256",
    "semantic_credit_contract",
    "passive_external_endpoint_credit",
    "semantic_decisions_exact",
    "external_credit_lifecycle_exact",
    "external_route_pinned_requests",
    "passive_completions",
})
_REQUEST_METADATA_KEYS = fixed_analysis._REQUEST_METADATA_KEYS
_ARMS = tuple(arm.value for arm in client.ARMS)


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
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _bound_path(
    raw_path: object, expected_sha256: object, *, name: str,
    within: Path | None = None,
) -> Path:
    _canonical_sha(expected_sha256, name=f"{name} SHA-256")
    _require(type(raw_path) is str and raw_path, f"{name} path is missing")
    path = Path(raw_path)
    _require(path.is_absolute(), f"{name} path must be absolute")
    path = path.resolve()
    if within is not None:
        try:
            path.relative_to(within.resolve())
        except ValueError as exc:
            raise ValueError(f"{name} escapes its result root") from exc
    _require(path.is_file() and _sha256(path) == expected_sha256,
             f"{name} digest differs")
    return path


def _contract_entry(
    contract: Mapping[str, object], name: str,
) -> tuple[Path, Mapping[str, object]]:
    entry = contract.get(name)
    _require(isinstance(entry, Mapping), f"run contract lacks {name}")
    path = _bound_path(
        entry.get("path"), entry.get("sha256"), name=f"run-contract {name}")
    return path, entry


def _validate_run_contract(path: Path) -> dict[str, object]:
    value = _load_object(path, name="adaptive run contract")
    _require(
        value.get("schema") == contract_builder.SCHEMA
        and value.get("fingerprint_sha256")
        == contract_builder.contract_fingerprint(value)
        and value.get("offline_replay_authorized") is True
        and value.get("performance_claim_allowed") is False,
        "adaptive run contract is invalid",
    )
    arguments = {}
    for argument, entry_name in (
        ("analysis", "analysis"),
        ("manifest", "phase_manifest"),
        ("elastic", "elastic_profile"),
        ("endpoint", "endpoint_service_profile"),
        ("receipt", "profile_receipt"),
        ("replay", "offline_replay"),
        ("implementation", "adaptive_implementation_contract"),
    ):
        artifact, entry = _contract_entry(value, entry_name)
        arguments[f"{argument}_path"] = artifact
        arguments[f"{argument}_sha256"] = entry["sha256"]
    rebuilt = contract_builder.build_run_contract(
        **arguments, repo_root=REPO_ROOT)
    _require(rebuilt == value, "adaptive run contract does not reproduce")
    return value


def _nearest_rank(values: list[float], fraction: float) -> float:
    _require(bool(values), "metric summary is empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _arm_summary(
    rows: list[Mapping[str, object]], *, e2e_slo_ms: float,
    ttft_slo_ms: float, tpot_slo_ms: float,
) -> dict[str, object]:
    _require(bool(rows), "arm summary is empty")
    result: dict[str, object] = {"requests": len(rows)}
    for metric in ("ttft_ms", "e2e_ms", "tpot_ms"):
        values = [float(row[metric]) for row in rows]
        result[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p99": _nearest_rank(values, 0.99),
        }
    result["e2e_goodput_fraction"] = sum(
        float(row["e2e_ms"]) <= e2e_slo_ms for row in rows) / len(rows)
    result["all_slo_goodput_fraction"] = sum(
        float(row["e2e_ms"]) <= e2e_slo_ms
        and float(row["ttft_ms"]) <= ttft_slo_ms
        and float(row["tpot_ms"]) <= tpot_slo_ms
        for row in rows
    ) / len(rows)
    return result


def _gain(candidate: float, baseline: float) -> float:
    _require(baseline > 0.0, "performance baseline must be positive")
    return (baseline - candidate) / baseline


def _controller_profile_matches(
    state: Mapping[str, object], *, endpoint_fingerprint: str,
    passive_feedback: bool = False,
) -> bool:
    profile = state.get("endpoint_service_profile")
    return (
        state.get("endpoint_feedback_mode") == "adaptive"
        and state.get("endpoint_passive_feedback") is passive_feedback
        and state.get("endpoint_routing_policy") == (
            "semantic_epoch_v1" if passive_feedback else "instant_score_v1")
        and isinstance(profile, Mapping)
        and profile.get("fingerprint_sha256") == endpoint_fingerprint
        and profile.get("deployment_scope") == "calibration_only"
    )


def _validate_block(
    path: Path, *, parent_contract: Mapping[str, object], block_key: str,
    sequence: int, arm: ForegroundArm, replicate: int,
    manifest: Mapping[str, object], endpoint_fingerprint: str,
    semantic_contract: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]],
           list[dict[str, object]], list[dict[str, object]]]:
    raw = _load_object(path, name=f"adaptive block {block_key}")
    _require(raw.get("schema") == fixed_analysis.STREAM_SCHEMA,
             f"adaptive stream schema differs: {block_key}")
    validation = raw.get("validation")
    _require(
        isinstance(validation, dict)
        and validation.get("all_streams_valid") is True
        and validation.get("router_decisions_exact") is True,
        f"adaptive stream validation failed: {block_key}",
    )
    contract = raw.get("c4_adaptive_screen_contract")
    expected_contract_keys = _BLOCK_CONTRACT_KEYS
    expected_block_schema = client.BLOCK_SCHEMA
    if semantic_contract is not None:
        expected_contract_keys |= _SEMANTIC_BLOCK_CONTRACT_EXTRA_KEYS
        expected_block_schema = client.SEMANTIC_BLOCK_SCHEMA
    _require(
        isinstance(contract, dict)
        and set(contract) == expected_contract_keys
        and contract == parent_contract
        and contract.get("schema") == expected_block_schema
        and contract.get("sequence") == sequence
        and contract.get("arm") == arm.value
        and contract.get("replicate") == replicate
        and contract.get("all_requests_valid") is True
        and contract.get("decision_cache_states_exact") is True
        and contract.get("completion_cache_evidence_exact") is True
        and contract.get("phase_aligned_endpoint_evidence") is True
        and contract.get("controller_reset_before_block_exact") is True
        and contract.get("controller_quiescent_after_block") is True
        and contract.get("one_way_route_commit_exact") is True
        and contract.get("unchanged_pd_data_plane") is True
        and contract.get("performance_claim_allowed") is False,
        f"adaptive block contract differs: {block_key}",
    )
    if semantic_contract is not None:
        _require(
            contract.get("endpoint_routing_policy") == "semantic_epoch_v1"
            and contract.get("endpoint_service_profile_fingerprint_sha256")
            == endpoint_fingerprint
            and contract.get("semantic_credit_contract")
            == semantic_contract.get("semantic_credit_contract")
            and contract.get("passive_external_endpoint_credit") is True
            and contract.get("semantic_decisions_exact") is True
            and contract.get("external_credit_lifecycle_exact") is True,
            f"semantic block policy contract differs: {block_key}",
        )
    request_index = contract.get("request_index")
    _require(isinstance(request_index, dict),
             f"adaptive request index is missing: {block_key}")
    expected, schedule_sha = fixed_analysis._expected_request_index(
        manifest, sequence=sequence, arm=arm, replicate=replicate)
    _require(
        set(request_index) == set(expected)
        and contract.get("semantic_schedule_sha256") == schedule_sha,
        f"adaptive semantic schedule differs: {block_key}",
    )
    for request_id, metadata in request_index.items():
        _require(
            isinstance(metadata, dict)
            and set(metadata) == _REQUEST_METADATA_KEYS,
            f"adaptive request metadata differs: {request_id}",
        )
        for name, expected_value in expected[request_id].items():
            _require(metadata.get(name) == expected_value,
                     f"adaptive semantic field differs: {request_id}/{name}")
        _canonical_sha(
            metadata.get("prompt_token_sha256"),
            name=f"{request_id}.prompt_token_sha256",
        )

    requests_raw = raw.get("requests")
    decisions_raw = raw.get("router_decisions")
    _require(isinstance(requests_raw, list) and isinstance(decisions_raw, list),
             f"adaptive measured rows are missing: {block_key}")
    requests = {str(row.get("request_id")): row for row in requests_raw}
    decisions = {str(row.get("request_id")): row for row in decisions_raw}
    _require(
        len(requests) == len(requests_raw)
        and len(decisions) == len(decisions_raw)
        and set(requests) == set(decisions) == set(request_index),
        f"adaptive request/decision IDs differ: {block_key}",
    )
    foreground = {}
    for request_id, metadata in request_index.items():
        request = requests[request_id]
        decision = decisions[request_id]
        _require(
            isinstance(request, dict)
            and request.get("valid") is True
            and request.get("requested_max_tokens") == metadata["output_tokens"],
            f"adaptive measured request is invalid: {request_id}",
        )
        _canonical_sha(
            request.get("output_text_sha256"),
            name=f"{request_id}.output_text_sha256",
        )
        metrics = client.c4._request_service_metrics(request)
        _require(
            isinstance(decision, dict)
            and decision.get("phase") == "complete"
            and decision.get("error") is None,
            f"adaptive decision is incomplete: {request_id}",
        )
        route = client._validate_dynamic_decision(
            decision, metadata, block_arm=arm, request_id=request_id,
            semantic_contract=semantic_contract)
        if metadata["tenant"] == Tenant.FOREGROUND.value:
            pair_key = metadata["pair_key"]
            _require(type(pair_key) is str and pair_key not in foreground,
                     f"adaptive foreground pair key differs: {request_id}")
            foreground[pair_key] = {
                "request_id": request_id,
                "metadata": metadata,
                "output_text_sha256": request["output_text_sha256"],
                "route": route,
                **metrics,
            }

    evidence = raw.get("endpoint_evidence")
    client.c4._validate_c4_endpoint_evidence(evidence)
    fixed_analysis._validate_capture_timing(
        evidence,
        request_index,
        phase_duration_ms=float(manifest["phase_duration_ms"]),
    )
    endpoint_rows = fixed_analysis._endpoint_phase_rows(
        evidence,
        block_key=block_key,
        sequence=sequence,
        arm=arm,
        replicate=replicate,
        phase_duration_ms=float(manifest["phase_duration_ms"]),
    )
    request_rows = fixed_analysis._request_phase_tenant_rows(
        block_key=block_key,
        sequence=sequence,
        arm=arm,
        replicate=replicate,
        request_index=request_index,
        requests=requests,
        decisions=decisions,
        phase_duration_ms=float(manifest["phase_duration_ms"]),
    )

    reset = raw.get("endpoint_controller_reset")
    before = raw.get("endpoint_controller_before")
    after = raw.get("endpoint_controller_after")
    _require(
        isinstance(reset, list) and len(reset) == 2
        and isinstance(before, list) and len(before) == 2
        and isinstance(after, list) and len(after) == 2
        and client._controllers_quiescent(before)
        and client._controllers_quiescent(after),
        f"adaptive controller boundary differs: {block_key}",
    )
    generations = contract["controller_generations"]
    _require(
        isinstance(generations, list) and len(generations) == 2
        and generations == [row.get("controller_generation") for row in reset]
        == [row.get("controller_generation") for row in before]
        == [row.get("controller_generation") for row in after]
        and all(
            row.get("success") is True
            and row.get("profile_fingerprint_sha256") == endpoint_fingerprint
            and row.get("controller", {}).get("inflight") == 0
            and not any(row.get("controller", {}).get("resources", {}).values())
            for row in reset
        )
        and all(_controller_profile_matches(
            row, endpoint_fingerprint=endpoint_fingerprint,
            passive_feedback=semantic_contract is not None)
            for row in (*before, *after)),
        f"adaptive controller generation/profile differs: {block_key}",
    )
    completed = sum(int(row["controller"]["completed"]) for row in after)
    _require(
        completed == (len(foreground) if arm is ForegroundArm.TEMPO else 0),
        f"adaptive controller first-response count differs: {block_key}",
    )
    passive_completed = sum(
        int(row["controller"].get("passive_completed", 0)) for row in after)
    external_requests = sum(
        metadata["tenant"] != Tenant.FOREGROUND.value
        for metadata in request_index.values())
    _require(
        passive_completed
        == (external_requests if semantic_contract is not None else 0),
        f"adaptive controller passive count differs: {block_key}",
    )
    if semantic_contract is not None:
        _require(
            contract.get("external_route_pinned_requests") == external_requests
            and contract.get("passive_completions") == passive_completed,
            f"semantic external-credit count differs: {block_key}",
        )
    return raw, foreground, endpoint_rows, request_rows


def _paired_samples(
    blocks: Mapping[tuple[int, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    result = []
    semantic_fields = (
        "phase", "tenant", "arrival_offset_ms", "prompt_tokens",
        "output_tokens", "cache_state", "ordinal", "pair_key",
        "prompt_token_sha256", "terminal_item",
    )
    for replicate in (0, 1):
        selected = {
            arm: blocks[(replicate, arm)] for arm in _ARMS
        }
        key_sets = [set(value) for value in selected.values()]
        _require(all(keys == key_sets[0] for keys in key_sets[1:]),
                 "adaptive paired foreground key sets differ")
        for pair_key in sorted(key_sets[0]):
            by_arm = {arm: selected[arm][pair_key] for arm in _ARMS}
            reference = by_arm[ForegroundArm.LOCAL.value]["metadata"]
            _require(
                all(
                    all(value["metadata"][name] == reference[name]
                        for name in semantic_fields)
                    for value in by_arm.values()
                )
                and len({
                    value["output_text_sha256"] for value in by_arm.values()
                }) == 1,
                f"adaptive paired semantics/output differ: {pair_key}",
            )
            result.append({
                "pair_key": pair_key,
                "replicate": replicate,
                **{name: reference[name] for name in (
                    "phase", "arrival_offset_ms", "prompt_tokens",
                    "output_tokens", "cache_state", "ordinal",
                )},
                "output_text_sha256": by_arm[
                    ForegroundArm.LOCAL.value]["output_text_sha256"],
                "arms": {
                    arm: {
                        name: value[name] for name in (
                            "request_id", "route", "ttft_ms", "e2e_ms",
                            "tpot_ms",
                        )
                    } for arm, value in by_arm.items()
                },
            })
    result.sort(key=lambda row: (
        row["replicate"], row["arrival_offset_ms"], row["ordinal"],
        row["pair_key"],
    ))
    return result


def _screen_metrics(
    samples: list[Mapping[str, object]], manifest: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    measurement = manifest["measurement"]
    e2e_slo = float(measurement["e2e_slo_ms"])
    ttft_slo = float(measurement["ttft_slo_ms"])
    tpot_slo = float(measurement["tpot_slo_ms"])
    arm_rows = {
        arm: [sample["arms"][arm] for sample in samples] for arm in _ARMS
    }
    summaries = {
        arm: _arm_summary(
            rows,
            e2e_slo_ms=e2e_slo,
            ttft_slo_ms=ttft_slo,
            tpot_slo_ms=tpot_slo,
        ) for arm, rows in arm_rows.items()
    }
    strongest_fixed = min(
        (ForegroundArm.LOCAL.value, ForegroundArm.REMOTE.value),
        key=lambda arm: (
            summaries[arm]["e2e_ms"]["mean"],
            0 if arm == ForegroundArm.LOCAL.value else 1,
        ),
    )
    tempo = summaries[ForegroundArm.TEMPO.value]
    predictor = summaries[ForegroundArm.PREDICTOR.value]
    fixed = summaries[strongest_fixed]
    mean_gain_fixed = _gain(
        float(tempo["e2e_ms"]["mean"]),
        float(fixed["e2e_ms"]["mean"]),
    )
    mean_gain_predictor = _gain(
        float(tempo["e2e_ms"]["mean"]),
        float(predictor["e2e_ms"]["mean"]),
    )
    p99_regression = -_gain(
        float(tempo["e2e_ms"]["p99"]),
        float(fixed["e2e_ms"]["p99"]),
    )
    paired_win_fraction = sum(
        float(sample["arms"][ForegroundArm.TEMPO.value]["e2e_ms"])
        < float(sample["arms"][strongest_fixed]["e2e_ms"])
        for sample in samples
    ) / len(samples)
    tempo_routes = Counter(
        sample["arms"][ForegroundArm.TEMPO.value]["route"]
        for sample in samples
    )
    gates = {
        "all_paired_requests_complete": bool(samples),
        "both_tempo_routes_exercised": all(
            tempo_routes[route] > 0
            for route in (client.c4._LOCAL_ROUTE, client.c4._REMOTE_ROUTE)
        ),
        "mean_gain_vs_strongest_fixed_at_least_3pct": (
            mean_gain_fixed >= MIN_MEAN_GAIN_VS_STRONGEST_FIXED),
        "mean_gain_vs_predictor_at_least_2pct": (
            mean_gain_predictor >= MIN_MEAN_GAIN_VS_PREDICTOR),
        "e2e_goodput_not_below_strongest_fixed": (
            tempo["e2e_goodput_fraction"]
            >= fixed["e2e_goodput_fraction"]),
        "all_slo_goodput_not_below_strongest_fixed": (
            tempo["all_slo_goodput_fraction"]
            >= fixed["all_slo_goodput_fraction"]),
        "p99_e2e_regression_at_most_5pct": (
            p99_regression <= MAX_P99_REGRESSION),
        "paired_win_fraction_at_least_55pct": (
            paired_win_fraction >= MIN_PAIRED_WIN_FRACTION),
    }
    group_rows = []
    for phase in manifest["phase_order"]:
        for geometry in VALIDATION_FOREGROUND_GEOMETRIES:
            selected = [
                sample for sample in samples
                if sample["phase"] == phase
                and sample["prompt_tokens"] == geometry.prompt_tokens
                and sample["output_tokens"] == geometry.output_tokens
                and sample["cache_state"] == geometry.cache_state.value
            ]
            _require(bool(selected), "adaptive phase/geometry cell is empty")
            arm_means = {
                arm: statistics.fmean(
                    float(sample["arms"][arm]["e2e_ms"])
                    for sample in selected)
                for arm in _ARMS
            }
            fixed_name = min(
                (ForegroundArm.LOCAL.value, ForegroundArm.REMOTE.value),
                key=lambda arm: arm_means[arm],
            )
            group_rows.append({
                "phase": phase,
                "prompt_tokens": geometry.prompt_tokens,
                "output_tokens": geometry.output_tokens,
                "cache_state": geometry.cache_state.value,
                "paired_requests": len(selected),
                "mean_e2e_ms": arm_means,
                "tempo_gain_vs_cell_strongest_fixed": _gain(
                    arm_means[ForegroundArm.TEMPO.value],
                    arm_means[fixed_name],
                ),
                "tempo_gain_vs_predictor": _gain(
                    arm_means[ForegroundArm.TEMPO.value],
                    arm_means[ForegroundArm.PREDICTOR.value],
                ),
            })
    _require(len(group_rows) == 36,
             "adaptive phase/geometry row inventory differs")
    metrics = {
        "arm_summaries": summaries,
        "strongest_fixed_name_calibration_only": strongest_fixed,
        "mean_gain_vs_strongest_fixed": mean_gain_fixed,
        "mean_gain_vs_predictor": mean_gain_predictor,
        "p99_e2e_regression_vs_strongest_fixed": p99_regression,
        "paired_win_fraction_vs_strongest_fixed": paired_win_fraction,
        "tempo_route_counts": {
            client.c4._LOCAL_ROUTE: tempo_routes[client.c4._LOCAL_ROUTE],
            client.c4._REMOTE_ROUTE: tempo_routes[client.c4._REMOTE_ROUTE],
        },
        "screen_gates": gates,
        "authorizes_independent_validation": all(gates.values()),
        "calibration_only": True,
        "performance_claim_allowed": False,
    }
    return metrics, group_rows


def analysis_fingerprint(value: Mapping[str, object]) -> str:
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
        expected_result_sha256, name="adaptive node result SHA-256")
    _require(result_path.is_file() and _sha256(result_path) == expected_result_sha256,
             "adaptive node result digest differs")
    node = _load_object(result_path, name="adaptive node result")
    _require(set(node) == _NODE_KEYS and node.get("schema") == NODE_SCHEMA,
             "adaptive node result inventory differs")
    result_root = result_path.parent
    raw_path = _bound_path(
        node["raw"], node["raw_sha256"], name="adaptive client raw",
        within=result_root)
    run_contract_path = _bound_path(
        node["run_contract"], node["run_contract_sha256"],
        name="adaptive run contract")
    run_contract = _validate_run_contract(run_contract_path)
    _require(
        node["run_contract_fingerprint_sha256"]
        == run_contract["fingerprint_sha256"],
        "adaptive node/run-contract fingerprint differs",
    )
    manifest_path, manifest_entry = _contract_entry(
        run_contract, "phase_manifest")
    manifest = _load_object(manifest_path, name="adaptive manifest")
    endpoint_path, endpoint_entry = _contract_entry(
        run_contract, "endpoint_service_profile")
    del endpoint_path
    source_result_path, source_result_entry = _contract_entry(
        run_contract, "source_node_result")
    source_result = _load_object(source_result_path, name="source C4 result")
    _require(
        node.get("slurm_job_id") == source_result.get("slurm_job_id")
        and type(node.get("slurm_job_id")) is str
        and bool(node["slurm_job_id"].strip()),
        "C4 and adaptive screen did not reuse one persistent allocation",
    )
    _require(
        node.get("source_workload") == run_contract["source_workload"]["path"]
        and node.get("source_workload_sha256")
        == run_contract["source_workload"]["sha256"]
        and node.get("phase_manifest") == str(manifest_path)
        and node.get("phase_manifest_sha256") == manifest_entry["sha256"]
        and node.get("elastic_profile")
        == run_contract["elastic_profile"]["path"]
        and node.get("elastic_profile_sha256")
        == run_contract["elastic_profile"]["sha256"]
        and node.get("endpoint_service_profile")
        == run_contract["endpoint_service_profile"]["path"]
        and node.get("endpoint_service_profile_sha256")
        == run_contract["endpoint_service_profile"]["sha256"]
        and node.get("fixed_runtime_environment")
        == run_contract["fixed_runtime_environment"]
        and node.get("block_count") == 8
        and node.get("correctness_gate_pass") is True
        and node.get("calibration_only") is True
        and node.get("performance_claim_allowed") is False
        and node.get("physical_switch_bottleneck_claim_allowed") is False
        and node.get("independent_validation_required") is True
        and node.get("unchanged_pd_data_plane") is True
        and node.get("transport") == "LMCacheConnectorV1:UCX"
        and isinstance(node.get("transport_environment"), dict)
        and 600.0 <= float(node["startup_readiness_timeout_s"]) <= 3600.0,
        "adaptive node lineage or invariant differs",
    )
    implementation_path, implementation_entry = _contract_entry(
        run_contract, "adaptive_implementation_contract")
    _require(
        node.get("adaptive_implementation_contract")
        == str(implementation_path)
        and node.get("adaptive_implementation_contract_sha256")
        == implementation_entry["sha256"]
        and node.get("adaptive_implementation_fingerprint_sha256")
        == implementation_entry["fingerprint_sha256"],
        "adaptive node implementation binding differs",
    )

    parent = _load_object(raw_path, name="adaptive client raw")
    _require(set(parent) == _CLIENT_KEYS and parent.get("schema") == client.SCHEMA,
             "adaptive client raw inventory differs")
    _require(
        Path(str(parent["run_contract"])).resolve() == run_contract_path
        and parent["run_contract_sha256"] == node["run_contract_sha256"]
        and Path(str(parent["manifest"])).resolve() == manifest_path
        and parent["manifest_sha256"] == manifest_entry["sha256"]
        and parent.get("blocks_completed") == 8
        and parent.get("live_screen_correctness_pass") is True
        and parent.get("calibration_only") is True
        and parent.get("performance_claim_allowed") is False,
        "adaptive client lineage or claim differs",
    )
    _bound_path(
        parent["cache_plan"], parent["cache_plan_sha256"],
        name="adaptive cache plan", within=raw_path.parent)
    runtime_path = _bound_path(
        parent["cache_runtime_evidence"],
        parent["cache_runtime_evidence_sha256"],
        name="adaptive cache runtime evidence", within=raw_path.parent)
    runtime = _load_object(runtime_path, name="adaptive cache runtime evidence")
    _require(
        runtime.get("schema") == client.c4.RUNTIME_EVIDENCE_SCHEMA
        and runtime.get("preparation_completed_before_measurement") is True
        and runtime.get("measurement_includes_preparation_requests") is False
        and runtime.get("ready_for_measurement") is True,
        "adaptive cache runtime evidence differs",
    )
    validated_node_artifacts = node_module._validate_client_artifacts(
        parent, client_raw_path=raw_path)
    _require(validated_node_artifacts == node["block_artifacts"],
             "adaptive node/client child bindings differ")

    expected_keys = [item[0] for item in node_module._EXPECTED_BLOCKS]
    artifacts = parent["artifacts"]
    contracts = parent["contracts"]
    _require(
        list(artifacts) == expected_keys
        and list(contracts) == expected_keys
        and parent["block_order"] == [
            {"arm": arm, "replicate": replicate}
            for _key, arm, replicate in node_module._EXPECTED_BLOCKS
        ],
        "adaptive parent block inventory differs",
    )
    blocks = {}
    endpoint_rows = []
    request_rows = []
    generation_history = [[], []]
    block_bindings = []
    for sequence, (key, arm_value, replicate) in enumerate(
        node_module._EXPECTED_BLOCKS
    ):
        arm = ForegroundArm(arm_value)
        entry = artifacts[key]
        path = _bound_path(
            entry["path"], entry["sha256"], name=f"adaptive block {key}",
            within=raw_path.parent)
        raw, foreground, block_endpoint_rows, block_request_rows = (
            _validate_block(
                path,
                parent_contract=contracts[key],
                block_key=key,
                sequence=sequence,
                arm=arm,
                replicate=replicate,
                manifest=manifest,
                endpoint_fingerprint=endpoint_entry["fingerprint_sha256"],
            ))
        blocks[(replicate, arm_value)] = foreground
        endpoint_rows.extend(block_endpoint_rows)
        request_rows.extend(block_request_rows)
        for controller_index, generation in enumerate(
            contracts[key]["controller_generations"]
        ):
            generation_history[controller_index].append(generation)
        block_bindings.append({
            "key": key,
            "path": str(path),
            "sha256": entry["sha256"],
            "arm": arm_value,
            "replicate": replicate,
        })
        del raw
    _require(
        generation_history == [list(range(1, 9)), list(range(1, 9))],
        "adaptive controller generations were not reset once per block",
    )
    _require(len(endpoint_rows) == 192,
             "adaptive endpoint-phase inventory must contain 192 rows")
    _require(len(request_rows) == 192,
             "adaptive request phase/tenant inventory must contain 192 rows")
    samples = _paired_samples(blocks)
    expected_foreground = sum(
        1 for metadata in contracts[expected_keys[0]]["request_index"].values()
        if metadata["tenant"] == Tenant.FOREGROUND.value
    ) * 2
    _require(len(samples) == expected_foreground,
             "adaptive paired foreground count differs")
    recomputed_paired = client._paired_gate(
        block_paths={key: Path(artifacts[key]["path"]) for key in expected_keys},
        contracts=contracts,
    )
    _require(parent["paired_output_gate"] == recomputed_paired,
             "adaptive paired gate differs from child evidence")
    metrics, group_rows = _screen_metrics(samples, manifest)
    _require(
        node.get("tempo_both_routes_exercised")
        == metrics["screen_gates"]["both_tempo_routes_exercised"]
        == parent.get("live_screen_route_diversity_pass"),
        "adaptive node/client route-diversity evidence differs",
    )
    output: dict[str, object] = {
        "schema": SCHEMA,
        "source_node_result": {
            "path": str(result_path), "sha256": expected_result_sha256},
        "source_c4_node_result": {
            "path": str(source_result_path),
            "sha256": source_result_entry["sha256"],
        },
        "persistent_allocation_job_id": node["slurm_job_id"],
        "run_contract": {
            "path": str(run_contract_path),
            "sha256": node["run_contract_sha256"],
            "fingerprint_sha256": run_contract["fingerprint_sha256"],
        },
        "block_artifacts": block_bindings,
        "controller_generation_history": generation_history,
        "endpoint_phase_rows": endpoint_rows,
        "request_phase_tenant_rows": request_rows,
        "foreground_paired_samples": samples,
        "phase_geometry_rows": group_rows,
        "screen_metrics": metrics,
        "authorizes_independent_validation": metrics[
            "authorizes_independent_validation"],
        "strongest_fixed_selection_authoritative": False,
        "calibration_only": True,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "independent_validation_required": True,
    }
    output["fingerprint_sha256"] = analysis_fingerprint(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite adaptive analysis")
    value = analyze(
        args.result, expected_result_sha256=args.expected_result_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "fingerprint_sha256": value["fingerprint_sha256"],
        "authorizes_independent_validation": value[
            "authorizes_independent_validation"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
