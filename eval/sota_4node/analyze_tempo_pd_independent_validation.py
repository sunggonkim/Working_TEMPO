#!/usr/bin/env python3
"""Issue the only authoritative verdict for frozen TEMPO validation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_adaptive_screen as adaptive_analysis
from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as fixed_analysis
from eval.sota_4node import analyze_tempo_pd_c4_semantic_epoch_screen as semantic_policy
from eval.sota_4node import analyze_tempo_pd_c4_semantic_load as semantic_load
from eval.sota_4node import build_tempo_pd_independent_validation_manifest as manifest_builder
from eval.sota_4node import build_tempo_pd_independent_validation_run_contract as contract_builder
from eval.sota_4node import run_tempo_pd_independent_validation_client as client
from eval.sota_4node import vllm_lmcache_pd_independent_validation_node as node_module
from tempo.pd_contention_workload import (
    ForegroundArm,
    LoadSelection,
    Tenant,
    TrafficShape,
    VALIDATION_FOREGROUND_GEOMETRIES,
    build_schedule,
    semantic_schedule_sha256,
)


SCHEMA = "tempo-pd-independent-validation-analysis-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_SCHEMA = node_module.SCHEMA
_ARMS = tuple(arm.value for arm in client.ARMS)
_REPLICATES = (2, 3, 4, 5)
_NODE_KEYS = frozenset({
    "schema", "raw", "raw_sha256", "run_contract",
    "run_contract_sha256", "run_contract_fingerprint_sha256",
    "independent_implementation_contract",
    "independent_implementation_contract_sha256",
    "independent_implementation_fingerprint_sha256",
    "independent_implementation_file_count", "source_workload",
    "source_workload_sha256", "independent_manifest",
    "independent_manifest_sha256", "promoted_elastic_profile",
    "promoted_elastic_profile_sha256",
    "promoted_endpoint_service_profile",
    "promoted_endpoint_service_profile_sha256", "slurm_job_id",
    "calibration_slurm_job_id", "separate_validation_allocation",
    "startup_readiness_timeout_s", "block_count", "block_artifacts",
    "tempo_both_routes_exercised", "fixed_runtime_environment",
    "transport_environment", "correctness_gate_pass",
    "held_out_burst_workload", "calibration_only",
    "post_validation_tuning_allowed", "performance_claim_allowed",
    "physical_switch_bottleneck_claim_allowed", "unchanged_pd_data_plane",
    "transport", "candidate",
})
_CLIENT_KEYS = frozenset({
    "schema", "run_id", "run_contract", "run_contract_sha256", "manifest",
    "manifest_sha256", "cache_plan", "cache_plan_sha256",
    "cache_runtime_evidence", "cache_runtime_evidence_sha256", "block_order",
    "artifacts", "contracts", "summaries", "paired_output_gate",
    "blocks_completed", "held_out_burst_workload",
    "independent_correctness_pass", "independent_route_diversity_pass",
    "calibration_only", "post_validation_tuning_allowed",
    "performance_claim_allowed", "physical_switch_bottleneck_claim_allowed",
    "unchanged_pd_data_plane", "candidate",
})
_BLOCK_CONTRACT_KEYS = frozenset({
    "schema", "sequence", "arm", "replicate", "semantic_schedule_sha256",
    "request_index", "controller_generations", "all_requests_valid",
    "decision_cache_states_exact", "completion_cache_evidence_exact",
    "phase_aligned_endpoint_evidence", "controller_reset_before_block_exact",
    "controller_quiescent_after_block", "one_way_route_commit_exact",
    "unchanged_pd_data_plane", "performance_claim_allowed",
    "held_out_burst_workload", "calibration_only",
})
_SEMANTIC_BLOCK_CONTRACT_KEYS = (
    _BLOCK_CONTRACT_KEYS
    | adaptive_analysis._SEMANTIC_BLOCK_CONTRACT_EXTRA_KEYS)
_REQUEST_METADATA_KEYS = fixed_analysis._REQUEST_METADATA_KEYS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object, *, name: str) -> str:
    return manifest_builder._canonical_sha(value, name=name)


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    return manifest_builder._load_object(path, name=name)


def _semantic_cell_summary(
    rows: list[dict[str, object]], *, max_num_seqs: int,
) -> dict[str, object]:
    """Represent an observed phase/pair cell without forcing pair usage."""
    if rows:
        value = semantic_load._summary(rows)
        _require(
            value["max_num_seqs"] == max_num_seqs,
            "independent semantic capacity differs within block",
        )
        return value
    empty_distribution = semantic_load._distribution(())
    return {
        "requests": 0,
        "max_num_seqs": max_num_seqs,
        "active_requests_before": dict(empty_distribution),
        "decode_tokens_before": dict(empty_distribution),
        "occupancy_ratio_before": dict(empty_distribution),
        "capacity_event_fraction": {
            "at_least_half": None,
            "at_least_three_quarters": None,
            "at_or_above_max_num_seqs": None,
        },
        "pair_counts": {},
        "tenant_counts": {},
    }


def _candidate_exercise(
    raw_blocks: list[tuple[Mapping[str, object], ForegroundArm]],
    *, semantic_contract: Mapping[str, object] | None,
) -> dict[str, object]:
    if semantic_contract is None:
        return {
            "policy": "instant_score_v1",
            "semantic_epoch_required": False,
            "all_pass": True,
        }

    tempo_routes: Counter[str] = Counter()
    tempo_reasons: Counter[str] = Counter()
    external_routes: Counter[str] = Counter()
    external_modes: Counter[str] = Counter()
    generations: list[int] = []
    tempo_requests = 0
    external_requests = 0
    for raw, arm in raw_blocks:
        block_contract = raw.get("independent_validation_contract")
        _require(isinstance(block_contract, Mapping),
                 "independent semantic block contract is missing")
        request_index = block_contract.get("request_index")
        decisions_raw = raw.get("router_decisions")
        _require(
            isinstance(request_index, Mapping)
            and isinstance(decisions_raw, list),
            "independent semantic exercise rows are missing",
        )
        decisions = {
            str(row.get("request_id")): row for row in decisions_raw
            if isinstance(row, Mapping)
        }
        _require(set(decisions) == set(request_index),
                 "independent semantic exercise IDs differ")
        for request_id, metadata in request_index.items():
            _require(isinstance(metadata, Mapping),
                     "independent semantic request metadata differs")
            decision = decisions[str(request_id)]
            _require(
                decision.get("frontend_semantic_load_schema")
                == semantic_policy.LOAD_SCHEMA
                and decision.get("frontend_semantic_load_source")
                == semantic_policy.LOAD_SOURCE,
                f"independent semantic ledger is missing: {request_id}",
            )
            tenant = metadata.get("tenant")
            if tenant == Tenant.FOREGROUND.value and arm is ForegroundArm.TEMPO:
                route, reason, generation = (
                    semantic_policy._validate_semantic_decision(
                        decision, contract=semantic_contract))
                tempo_routes[route] += 1
                tempo_reasons[reason] += 1
                generations.append(generation)
                tempo_requests += 1
            elif tenant != Tenant.FOREGROUND.value:
                route, mode = semantic_policy._validate_external_decision(
                    decision, metadata=metadata, request_id=str(request_id))
                external_routes[route] += 1
                external_modes[mode] += 1
                external_requests += 1
    gates = {
        "both_tempo_routes": set(tempo_routes) == {
            semantic_policy.LOCAL_ROUTE, semantic_policy.REMOTE_ROUTE},
        "epoch_generation_advanced": bool(generations)
        and max(generations) > 0,
        "remote_epoch_opened": (
            tempo_reasons["semantic_epoch_open_remote_high_water"] > 0),
        "remote_epoch_closed": any(
            tempo_reasons[name] > 0 for name in (
                "semantic_epoch_close_remote_unavailable",
                "semantic_epoch_close_decoder_low_water",
            )),
        "both_external_routes_observed": (
            external_routes[semantic_policy.LOCAL_ROUTE] > 0
            and external_routes[semantic_policy.REMOTE_ROUTE] > 0),
    }
    return {
        "policy": "semantic_epoch_v1",
        "semantic_epoch_required": True,
        "semantic_tempo_requests": tempo_requests,
        "external_route_pinned_requests": external_requests,
        "tempo_route_counts": dict(sorted(tempo_routes.items())),
        "tempo_reason_counts": dict(sorted(tempo_reasons.items())),
        "external_route_counts": dict(sorted(external_routes.items())),
        "external_service_proxy_mode_counts": dict(
            sorted(external_modes.items())),
        "maximum_epoch_generation": max(generations) if generations else 0,
        "gates": gates,
        "all_pass": all(gates.values()),
    }


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
        entry.get("path"), entry.get("sha256"),
        name=f"run-contract {name}")
    return path, entry


def _validate_run_contract(path: Path) -> dict[str, object]:
    value = _load_object(path, name="independent run contract")
    _require(
        value.get("schema") == contract_builder.SCHEMA
        and value.get("fingerprint_sha256")
        == contract_builder.contract_fingerprint(value)
        and value.get("independent_validation_authorized") is True
        and value.get("post_validation_tuning_allowed") is False
        and value.get("performance_claim_allowed") is False,
        "independent run contract is invalid",
    )
    arguments = {}
    for argument, entry_name in (
        ("manifest", "independent_manifest"),
        ("adaptive_analysis", "candidate_screen_analysis"),
        ("preregistration", "preregistration"),
        ("elastic", "promoted_elastic_profile"),
        ("endpoint", "promoted_endpoint_service_profile"),
        ("promotion_receipt", "profile_promotion_receipt"),
        ("implementation", "independent_implementation_contract"),
    ):
        artifact, entry = _contract_entry(value, entry_name)
        arguments[f"{argument}_path"] = artifact
        arguments[f"{argument}_sha256"] = entry["sha256"]
    rebuilt = contract_builder.build_run_contract(
        **arguments, repo_root=REPO_ROOT)
    _require(rebuilt == value, "independent run contract does not reproduce")
    return value


def _expected_request_index(
    manifest: Mapping[str, object], *, sequence: int,
    arm: ForegroundArm, replicate: int,
) -> tuple[dict[str, dict[str, object]], str]:
    rates = manifest.get("background_rates_per_s")
    _require(isinstance(rates, Mapping),
             "independent background rates are missing")
    selection = LoadSelection(
        decoder_reference_rate_per_s=float(rates["decoder_hot"]),
        remote_reference_rate_per_s=float(rates["cold_remote_hot"]),
        decoder_fraction=1.0,
        remote_fraction=1.0,
        kv_remote_rate_per_s=float(rates["kv_remote_hot"]),
    )
    schedule = build_schedule(
        states=client.c4.manifest_builder.PHASES,
        selection=selection,
        foreground_arm=arm,
        foreground_rate_per_s=float(manifest["foreground_rate_per_s"]),
        trial_id=f"independent-r{replicate}-{arm.value}",
        shape=TrafficShape.BURST,
        phase_duration_ms=float(manifest["phase_duration_ms"]),
        foreground_geometries=VALIDATION_FOREGROUND_GEOMETRIES,
        passive_endpoint_feedback=True,
    )
    result = {}
    for request in schedule:
        geometry = request.geometry
        geometry_index = (
            VALIDATION_FOREGROUND_GEOMETRIES.index(geometry)
            if request.tenant is Tenant.FOREGROUND else -1
        )
        terminal_item = client.c4._terminal_item(
            tenant=request.tenant,
            ordinal=request.ordinal,
            geometry_index=geometry_index,
            cache_state=geometry.cache_state,
        )
        request_id = client.c4._request_id(
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


def _controller_profile_matches(
    state: Mapping[str, object], *, endpoint_fingerprint: str,
    semantic_candidate: bool = False,
) -> bool:
    profile = state.get("endpoint_service_profile")
    return (
        state.get("endpoint_feedback_mode") == "adaptive"
        and state.get("endpoint_passive_feedback") is semantic_candidate
        and state.get("endpoint_routing_policy") == (
            "semantic_epoch_v1" if semantic_candidate else "instant_score_v1")
        and isinstance(profile, Mapping)
        and profile.get("fingerprint_sha256") == endpoint_fingerprint
        and profile.get("deployment_scope") == "frozen_validation"
    )


def _validate_block(
    path: Path, *, parent_contract: Mapping[str, object], block_key: str,
    sequence: int, arm: ForegroundArm, replicate: int,
    manifest: Mapping[str, object], endpoint_fingerprint: str,
    semantic_contract: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]],
           list[dict[str, object]], list[dict[str, object]], dict[str, object],
           dict[str, object]]:
    raw = _load_object(path, name=f"independent block {block_key}")
    _require(raw.get("schema") == fixed_analysis.STREAM_SCHEMA,
             f"independent stream schema differs: {block_key}")
    validation = raw.get("validation")
    _require(
        isinstance(validation, dict)
        and validation.get("all_streams_valid") is True
        and validation.get("router_decisions_exact") is True,
        f"independent stream validation failed: {block_key}",
    )
    contract = raw.get("independent_validation_contract")
    expected_contract_keys = (
        _SEMANTIC_BLOCK_CONTRACT_KEYS
        if semantic_contract is not None else _BLOCK_CONTRACT_KEYS)
    _require(
        isinstance(contract, dict)
        and set(contract) == expected_contract_keys
        and contract == parent_contract
        and contract.get("schema") == client.BLOCK_SCHEMA
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
        and contract.get("held_out_burst_workload") is True
        and contract.get("calibration_only") is False
        and contract.get("unchanged_pd_data_plane") is True
        and contract.get("performance_claim_allowed") is False,
        f"independent block contract differs: {block_key}",
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
            f"independent semantic policy differs: {block_key}",
        )
    request_index = contract.get("request_index")
    _require(isinstance(request_index, dict),
             f"independent request index is missing: {block_key}")
    expected, schedule_sha = _expected_request_index(
        manifest, sequence=sequence, arm=arm, replicate=replicate)
    _require(
        set(request_index) == set(expected)
        and contract.get("semantic_schedule_sha256") == schedule_sha,
        f"independent semantic schedule differs: {block_key}",
    )
    for request_id, metadata in request_index.items():
        _require(
            isinstance(metadata, dict)
            and set(metadata) == _REQUEST_METADATA_KEYS,
            f"independent request metadata differs: {request_id}",
        )
        for name, expected_value in expected[request_id].items():
            _require(metadata.get(name) == expected_value,
                     f"independent semantic field differs: {request_id}/{name}")
        _canonical_sha(
            metadata.get("prompt_token_sha256"),
            name=f"{request_id}.prompt_token_sha256")

    requests_raw = raw.get("requests")
    decisions_raw = raw.get("router_decisions")
    _require(isinstance(requests_raw, list) and isinstance(decisions_raw, list),
             f"independent measured rows are missing: {block_key}")
    requests = {str(row.get("request_id")): row for row in requests_raw}
    decisions = {str(row.get("request_id")): row for row in decisions_raw}
    _require(
        len(requests) == len(requests_raw)
        and len(decisions) == len(decisions_raw)
        and set(requests) == set(decisions) == set(request_index),
        f"independent request/decision IDs differ: {block_key}",
    )
    foreground = {}
    for request_id, metadata in request_index.items():
        request = requests[request_id]
        decision = decisions[request_id]
        _require(
            isinstance(request, dict)
            and request.get("valid") is True
            and request.get("error") is None
            and request.get("requested_max_tokens") == metadata["output_tokens"],
            f"independent measured request is invalid: {request_id}",
        )
        _canonical_sha(
            request.get("output_text_sha256"),
            name=f"{request_id}.output_text_sha256")
        metrics = client.c4._request_service_metrics(request)
        _require(
            isinstance(decision, dict)
            and decision.get("phase") == "complete"
            and decision.get("error") is None
            and "fallback" not in str(decision.get("reason", "")).lower(),
            f"independent decision is incomplete or fallback: {request_id}",
        )
        route = client.adaptive._validate_dynamic_decision(
            decision, metadata, block_arm=arm, request_id=request_id,
            semantic_contract=semantic_contract)
        if metadata["tenant"] == Tenant.FOREGROUND.value:
            pair_key = metadata["pair_key"]
            _require(type(pair_key) is str and pair_key not in foreground,
                     f"independent foreground pair key differs: {request_id}")
            dispatch = request.get("dispatch_offset_ns")
            stream_end = request.get("stream_end_offset_ns")
            _require(
                type(dispatch) is int and dispatch >= 0
                and type(stream_end) is int and stream_end > dispatch,
                f"independent foreground service clock differs: {request_id}",
            )
            foreground[pair_key] = {
                "request_id": request_id,
                "metadata": metadata,
                "output_text_sha256": request["output_text_sha256"],
                "route": route,
                "dispatch_offset_ns": dispatch,
                "stream_end_offset_ns": stream_end,
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
        and client.adaptive._controllers_quiescent(before)
        and client.adaptive._controllers_quiescent(after),
        f"independent controller boundary differs: {block_key}",
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
            semantic_candidate=semantic_contract is not None)
            for row in (*before, *after)),
        f"independent controller generation/profile differs: {block_key}",
    )
    completed = sum(int(row["controller"]["completed"]) for row in after)
    _require(
        completed == (len(foreground) if arm is ForegroundArm.TEMPO else 0),
        f"independent controller first-response count differs: {block_key}",
    )
    passive_completed = sum(
        int(row["controller"].get("passive_completed", 0)) for row in after)
    external_requests = sum(
        metadata["tenant"] != Tenant.FOREGROUND.value
        for metadata in request_index.values())
    _require(
        passive_completed
        == (external_requests if semantic_contract is not None else 0),
        f"independent passive endpoint count differs: {block_key}",
    )
    if semantic_contract is not None:
        _require(
            contract.get("external_route_pinned_requests") == external_requests
            and contract.get("passive_completions") == passive_completed,
            f"independent external credit count differs: {block_key}",
        )
    frontend_contract, semantic_rows = semantic_load._block_rows(raw)
    _require(
        frontend_contract.get("semantic_contract_name")
        == "independent_validation_contract"
        and frontend_contract.get("arm") == arm.value
        and frontend_contract.get("replicate") == replicate
        and frontend_contract.get("block_sequence_index") == sequence
        and len(semantic_rows) == len(request_index),
        f"independent frontend semantic-load lineage differs: {block_key}",
    )
    capacities = {int(row["max_num_seqs"]) for row in semantic_rows}
    _require(
        len(capacities) == 1,
        f"independent semantic capacity differs: {block_key}",
    )
    max_num_seqs = next(iter(capacities))
    phase_pair_all = {}
    phase_pair_foreground = {}
    for phase in manifest["phase_order"]:
        for pair in (0, 1):
            selected = [
                row for row in semantic_rows
                if row["phase"] == phase and row["pair_index"] == pair
            ]
            selected_foreground = [row for row in selected if row["foreground"]]
            key = f"{phase}:pair{pair}"
            phase_pair_all[key] = _semantic_cell_summary(
                selected, max_num_seqs=max_num_seqs)
            phase_pair_foreground[key] = _semantic_cell_summary(
                selected_foreground, max_num_seqs=max_num_seqs)
    semantic_summary = {
        "schema": semantic_load.SCHEMA,
        "load_schema": semantic_load.LOAD_SCHEMA,
        "load_source": semantic_load.LOAD_SOURCE,
        "policy_input_used": semantic_contract is not None,
        "block_key": block_key,
        "sequence": sequence,
        "arm": arm.value,
        "replicate": replicate,
        "requests": len(semantic_rows),
        "phase_pair_all_tenants": phase_pair_all,
        "phase_pair_foreground": phase_pair_foreground,
    }
    duration_s = (
        max(row["stream_end_offset_ns"] for row in foreground.values())
        - min(row["dispatch_offset_ns"] for row in foreground.values())
    ) / 1_000_000_000.0
    _require(duration_s > 0.0, f"independent block duration differs: {block_key}")
    block_goodput = {
        "block_key": block_key,
        "sequence": sequence,
        "arm": arm.value,
        "replicate": replicate,
        "foreground_requests": len(foreground),
        "foreground_output_tokens": sum(
            int(row["metadata"]["output_tokens"])
            for row in foreground.values()),
        "dispatch_to_stream_end_s": duration_s,
    }
    return (
        raw, foreground, endpoint_rows, request_rows, block_goodput,
        semantic_summary,
    )


def _paired_samples(
    blocks: Mapping[tuple[int, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    result = []
    semantic_fields = (
        "phase", "tenant", "arrival_offset_ms", "prompt_tokens",
        "output_tokens", "cache_state", "ordinal", "pair_key",
        "prompt_token_sha256", "terminal_item",
    )
    for replicate in _REPLICATES:
        selected = {arm: blocks[(replicate, arm)] for arm in _ARMS}
        key_sets = [set(value) for value in selected.values()]
        _require(all(keys == key_sets[0] for keys in key_sets[1:]),
                 "independent paired foreground key sets differ")
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
                f"independent paired semantics/output differ: {pair_key}",
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
                            "tpot_ms", "dispatch_offset_ns",
                            "stream_end_offset_ns",
                        )
                    } for arm, value in by_arm.items()
                },
            })
    result.sort(key=lambda row: (
        row["replicate"], row["arrival_offset_ms"], row["ordinal"],
        row["pair_key"],
    ))
    return result


def _nearest_rank(values: list[float], fraction: float) -> float:
    _require(bool(values), "metric summary is empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _gain(candidate: float, baseline: float) -> float:
    _require(baseline > 0.0, "performance baseline must be positive")
    return (baseline - candidate) / baseline


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
            "p99_nearest_rank": _nearest_rank(values, 0.99),
            "maximum": max(values),
        }
    result["all_slo_success_fraction"] = sum(
        float(row["e2e_ms"]) <= e2e_slo_ms
        and float(row["ttft_ms"]) <= ttft_slo_ms
        and float(row["tpot_ms"]) <= tpot_slo_ms
        for row in rows
    ) / len(rows)
    return result


def _performance_metrics(
    samples: list[Mapping[str, object]],
    block_goodput: list[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    measurement = manifest["measurement"]
    thresholds = manifest["success_gates"]
    arm_rows = {
        arm: [sample["arms"][arm] for sample in samples] for arm in _ARMS
    }
    summaries = {
        arm: _arm_summary(
            rows,
            e2e_slo_ms=float(measurement["e2e_slo_ms"]),
            ttft_slo_ms=float(measurement["ttft_slo_ms"]),
            tpot_slo_ms=float(measurement["tpot_slo_ms"]),
        ) for arm, rows in arm_rows.items()
    }
    strongest_fixed = min(
        (ForegroundArm.LOCAL.value, ForegroundArm.REMOTE.value),
        key=lambda arm: (
            summaries[arm]["e2e_ms"]["median"],
            0 if arm == ForegroundArm.LOCAL.value else 1,
        ),
    )
    tempo_name = ForegroundArm.TEMPO.value
    predictor_name = ForegroundArm.PREDICTOR.value
    tempo = summaries[tempo_name]
    fixed = summaries[strongest_fixed]
    predictor = summaries[predictor_name]
    median_gain_fixed = _gain(
        float(tempo["e2e_ms"]["median"]),
        float(fixed["e2e_ms"]["median"]))
    median_gain_predictor = _gain(
        float(tempo["e2e_ms"]["median"]),
        float(predictor["e2e_ms"]["median"]))

    goodput = {}
    for arm in _ARMS:
        blocks = [row for row in block_goodput if row["arm"] == arm]
        _require(len(blocks) == 4, f"independent goodput blocks differ: {arm}")
        duration = sum(float(row["dispatch_to_stream_end_s"]) for row in blocks)
        requests = sum(int(row["foreground_requests"]) for row in blocks)
        tokens = sum(int(row["foreground_output_tokens"]) for row in blocks)
        _require(duration > 0.0 and requests == len(arm_rows[arm]),
                 f"independent goodput accounting differs: {arm}")
        goodput[arm] = {
            "foreground_requests": requests,
            "foreground_output_tokens": tokens,
            "summed_dispatch_to_stream_end_s": duration,
            "request_goodput_per_s": requests / duration,
            "output_token_goodput_per_s": tokens / duration,
        }
    goodput_gain = (
        float(goodput[tempo_name]["request_goodput_per_s"])
        / float(goodput[strongest_fixed]["request_goodput_per_s"]) - 1.0
    )

    paired_deltas_global = [
        float(sample["arms"][tempo_name]["e2e_ms"])
        - float(sample["arms"][strongest_fixed]["e2e_ms"])
        for sample in samples
    ]
    paired_deltas_oracle = [
        float(sample["arms"][tempo_name]["e2e_ms"])
        - min(
            float(sample["arms"][ForegroundArm.LOCAL.value]["e2e_ms"]),
            float(sample["arms"][ForegroundArm.REMOTE.value]["e2e_ms"]),
        ) for sample in samples
    ]
    paired_win_fraction = sum(
        delta < 0.0 for delta in paired_deltas_global) / len(samples)
    pooled_e2e_p99_regression = (
        float(tempo["e2e_ms"]["p99_nearest_rank"])
        / float(fixed["e2e_ms"]["p99_nearest_rank"]) - 1.0
    )
    pooled_tpot_p99_regression = (
        float(tempo["tpot_ms"]["p99_nearest_rank"])
        / float(fixed["tpot_ms"]["p99_nearest_rank"]) - 1.0
    )

    group_rows = []
    all_group_gates = True
    for phase in manifest["phase_order"]:
        for geometry in VALIDATION_FOREGROUND_GEOMETRIES:
            selected = [
                sample for sample in samples
                if sample["phase"] == phase
                and sample["prompt_tokens"] == geometry.prompt_tokens
                and sample["output_tokens"] == geometry.output_tokens
                and sample["cache_state"] == geometry.cache_state.value
            ]
            _require(
                len(selected)
                == int(manifest["paired_foreground_samples_per_group"]),
                "independent phase/geometry group sample count differs",
            )
            group_summaries = {
                arm: _arm_summary(
                    [sample["arms"][arm] for sample in selected],
                    e2e_slo_ms=float(measurement["e2e_slo_ms"]),
                    ttft_slo_ms=float(measurement["ttft_slo_ms"]),
                    tpot_slo_ms=float(measurement["tpot_slo_ms"]),
                ) for arm in _ARMS
            }
            group_fixed = min(
                (ForegroundArm.LOCAL.value, ForegroundArm.REMOTE.value),
                key=lambda name: (
                    group_summaries[name]["e2e_ms"]["median"],
                    0 if name == ForegroundArm.LOCAL.value else 1,
                ),
            )
            wins = sum(
                float(sample["arms"][tempo_name]["e2e_ms"])
                < float(sample["arms"][group_fixed]["e2e_ms"])
                for sample in selected
            )
            win_fraction = wins / len(selected)
            e2e_regression = (
                float(group_summaries[tempo_name]["e2e_ms"][
                    "p99_nearest_rank"])
                / float(group_summaries[group_fixed]["e2e_ms"][
                    "p99_nearest_rank"]) - 1.0
            )
            tpot_regression = (
                float(group_summaries[tempo_name]["tpot_ms"][
                    "p99_nearest_rank"])
                / float(group_summaries[group_fixed]["tpot_ms"][
                    "p99_nearest_rank"]) - 1.0
            )
            group_gates = {
                "paired_win_fraction_at_least_60pct": win_fraction >= float(
                    thresholds[
                        "minimum_group_paired_win_fraction_vs_group_strongest_fixed"]),
                "e2e_p99_regression_at_most_5pct": e2e_regression <= float(
                    thresholds[
                        "maximum_group_e2e_p99_regression_vs_group_strongest_fixed"]),
                "tpot_p99_regression_at_most_5pct": tpot_regression <= float(
                    thresholds[
                        "maximum_group_tpot_p99_regression_vs_group_strongest_fixed"]),
            }
            all_group_gates = all_group_gates and all(group_gates.values())
            group_rows.append({
                "phase": phase,
                "prompt_tokens": geometry.prompt_tokens,
                "output_tokens": geometry.output_tokens,
                "cache_state": geometry.cache_state.value,
                "paired_requests": len(selected),
                "strongest_fixed_arm_for_group": group_fixed,
                "arm_summaries": group_summaries,
                "paired_win_fraction_vs_group_strongest_fixed": win_fraction,
                "e2e_p99_regression_vs_group_strongest_fixed": e2e_regression,
                "tpot_p99_regression_vs_group_strongest_fixed": tpot_regression,
                "gates": group_gates,
                "group_pass": all(group_gates.values()),
            })
    _require(len(group_rows) == 36,
             "independent workload group inventory differs")

    counterfactual_values = {"local": [], "remote": []}
    for sample in samples:
        route = sample["arms"][tempo_name]["route"]
        selected = float(sample["arms"][tempo_name]["e2e_ms"])
        if route == client.c4._LOCAL_ROUTE:
            other = float(sample["arms"][ForegroundArm.REMOTE.value]["e2e_ms"])
            counterfactual_values["local"].append((other - selected) / other)
        elif route == client.c4._REMOTE_ROUTE:
            other = float(sample["arms"][ForegroundArm.LOCAL.value]["e2e_ms"])
            counterfactual_values["remote"].append((other - selected) / other)
        else:
            raise ValueError("TEMPO independent route is not canonical")
    minimum_route_samples = int(
        measurement["minimum_route_counterfactual_samples_per_route"])
    route_counterfactual = {
        route: {
            "selected_requests": len(values),
            "median_relative_gain_vs_other_fixed_route": (
                statistics.median(values) if values else None),
        } for route, values in counterfactual_values.items()
    }
    local_counterfactual_pass = (
        len(counterfactual_values["local"]) >= minimum_route_samples
        and statistics.median(counterfactual_values["local"]) >= float(
            thresholds[
                "minimum_median_local_selection_gain_vs_remote_counterfactual"])
    )
    remote_counterfactual_pass = (
        len(counterfactual_values["remote"]) >= minimum_route_samples
        and statistics.median(counterfactual_values["remote"]) >= float(
            thresholds[
                "minimum_median_remote_selection_gain_vs_local_counterfactual"])
    )
    gates = {
        "all_paired_requests_complete_and_exact": bool(samples),
        "both_tempo_routes_exercised_with_minimum_samples": (
            len(counterfactual_values["local"]) >= minimum_route_samples
            and len(counterfactual_values["remote"]) >= minimum_route_samples),
        "pooled_median_e2e_gain_vs_strongest_fixed_at_least_10pct": (
            median_gain_fixed >= float(thresholds[
                "minimum_pooled_median_e2e_gain_vs_strongest_fixed"])),
        "pooled_median_e2e_gain_vs_predictor_at_least_5pct": (
            median_gain_predictor >= float(thresholds[
                "minimum_pooled_median_e2e_gain_vs_predictor"])),
        "request_goodput_gain_vs_strongest_fixed_at_least_5pct": (
            goodput_gain >= float(thresholds[
                "minimum_request_goodput_gain_vs_strongest_fixed"])),
        "overall_paired_win_fraction_at_least_75pct": (
            paired_win_fraction >= float(thresholds[
                "minimum_overall_paired_win_fraction_vs_strongest_fixed"])),
        "all_36_workload_groups_pass": all_group_gates,
        "pooled_e2e_p99_regression_at_most_5pct": (
            pooled_e2e_p99_regression <= float(thresholds[
                "maximum_pooled_e2e_p99_regression_vs_strongest_fixed"])),
        "pooled_tpot_p99_regression_at_most_5pct": (
            pooled_tpot_p99_regression <= float(thresholds[
                "maximum_pooled_tpot_p99_regression_vs_strongest_fixed"])),
        "worst_paired_e2e_regression_vs_strongest_fixed_at_most_100ms": (
            max(paired_deltas_global) <= float(thresholds[
                "maximum_worst_paired_e2e_regression_ms_vs_strongest_fixed"])),
        "worst_paired_e2e_regression_vs_request_oracle_fixed_at_most_100ms": (
            max(paired_deltas_oracle) <= float(thresholds[
                "maximum_worst_paired_e2e_regression_ms_vs_per_request_best_fixed"])),
        "local_selection_counterfactual_gain_at_least_5pct":
            local_counterfactual_pass,
        "remote_selection_counterfactual_gain_at_least_5pct":
            remote_counterfactual_pass,
    }
    metrics = {
        "arm_summaries": summaries,
        "strongest_fixed_arm_authoritative": strongest_fixed,
        "pooled_median_e2e_gain_vs_strongest_fixed": median_gain_fixed,
        "pooled_median_e2e_gain_vs_predictor": median_gain_predictor,
        "request_goodput": goodput,
        "request_goodput_gain_vs_strongest_fixed": goodput_gain,
        "paired_win_fraction_vs_strongest_fixed": paired_win_fraction,
        "pooled_e2e_p99_regression_vs_strongest_fixed":
            pooled_e2e_p99_regression,
        "pooled_tpot_p99_regression_vs_strongest_fixed":
            pooled_tpot_p99_regression,
        "worst_paired_e2e_regression_ms_vs_strongest_fixed":
            max(paired_deltas_global),
        "worst_paired_e2e_regression_ms_vs_request_oracle_fixed":
            max(paired_deltas_oracle),
        "route_counterfactual": route_counterfactual,
        "final_gates": gates,
        "all_performance_gates_pass": all(gates.values()),
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
        expected_result_sha256, name="independent node result SHA-256")
    _require(result_path.is_file() and _sha256(result_path) == expected_result_sha256,
             "independent node result digest differs")
    node = _load_object(result_path, name="independent node result")
    _require(set(node) == _NODE_KEYS and node.get("schema") == NODE_SCHEMA,
             "independent node result inventory differs")
    result_root = result_path.parent
    raw_path = _bound_path(
        node["raw"], node["raw_sha256"], name="independent client raw",
        within=result_root)
    run_contract_path = _bound_path(
        node["run_contract"], node["run_contract_sha256"],
        name="independent run contract")
    run_contract = _validate_run_contract(run_contract_path)
    candidate = run_contract.get("candidate")
    _require(isinstance(candidate, Mapping),
             "independent run contract candidate is missing")
    semantic_contract = (
        run_contract
        if candidate.get("kind") == "candidate_b_semantic_epoch_v1"
        else None)
    _require(
        node["run_contract_fingerprint_sha256"]
        == run_contract["fingerprint_sha256"],
        "independent node/run-contract fingerprint differs",
    )
    manifest_path, manifest_entry = _contract_entry(
        run_contract, "independent_manifest")
    manifest = _load_object(manifest_path, name="independent manifest")
    endpoint_path, endpoint_entry = _contract_entry(
        run_contract, "promoted_endpoint_service_profile")
    del endpoint_path
    _require(
        type(node.get("slurm_job_id")) is str
        and bool(node["slurm_job_id"].strip())
        and node.get("calibration_slurm_job_id")
        == run_contract.get("calibration_slurm_job_id")
        and node["slurm_job_id"] != node["calibration_slurm_job_id"]
        and node.get("separate_validation_allocation") is True,
        "independent validation did not use a separate allocation",
    )
    _require(
        node.get("source_workload") == run_contract["source_workload"]["path"]
        and node.get("candidate") == candidate
        and node.get("source_workload_sha256")
        == run_contract["source_workload"]["sha256"]
        and node.get("independent_manifest") == str(manifest_path)
        and node.get("independent_manifest_sha256") == manifest_entry["sha256"]
        and node.get("promoted_elastic_profile")
        == run_contract["promoted_elastic_profile"]["path"]
        and node.get("promoted_elastic_profile_sha256")
        == run_contract["promoted_elastic_profile"]["sha256"]
        and node.get("promoted_endpoint_service_profile")
        == run_contract["promoted_endpoint_service_profile"]["path"]
        and node.get("promoted_endpoint_service_profile_sha256")
        == run_contract["promoted_endpoint_service_profile"]["sha256"]
        and node.get("fixed_runtime_environment")
        == run_contract["fixed_runtime_environment"]
        and node.get("block_count") == 16
        and node.get("correctness_gate_pass") is True
        and node.get("held_out_burst_workload") is True
        and node.get("calibration_only") is False
        and node.get("post_validation_tuning_allowed") is False
        and node.get("performance_claim_allowed") is False
        and node.get("physical_switch_bottleneck_claim_allowed") is False
        and node.get("unchanged_pd_data_plane") is True
        and node.get("transport") == "LMCacheConnectorV1:UCX"
        and isinstance(node.get("transport_environment"), dict)
        and 600.0 <= float(node["startup_readiness_timeout_s"]) <= 3600.0,
        "independent node lineage or invariant differs",
    )
    implementation_path, implementation_entry = _contract_entry(
        run_contract, "independent_implementation_contract")
    _require(
        node.get("independent_implementation_contract")
        == str(implementation_path)
        and node.get("independent_implementation_contract_sha256")
        == implementation_entry["sha256"]
        and node.get("independent_implementation_fingerprint_sha256")
        == implementation_entry["fingerprint_sha256"],
        "independent node implementation binding differs",
    )

    parent = _load_object(raw_path, name="independent client raw")
    _require(set(parent) == _CLIENT_KEYS and parent.get("schema") == client.SCHEMA,
             "independent client raw inventory differs")
    _require(
        Path(str(parent["run_contract"])).resolve() == run_contract_path
        and parent.get("candidate") == candidate
        and parent["run_contract_sha256"] == node["run_contract_sha256"]
        and Path(str(parent["manifest"])).resolve() == manifest_path
        and parent["manifest_sha256"] == manifest_entry["sha256"]
        and parent.get("blocks_completed") == 16
        and parent.get("independent_correctness_pass") is True
        and parent.get("held_out_burst_workload") is True
        and parent.get("calibration_only") is False
        and parent.get("post_validation_tuning_allowed") is False
        and parent.get("performance_claim_allowed") is False,
        "independent client lineage or claim differs",
    )
    _bound_path(
        parent["cache_plan"], parent["cache_plan_sha256"],
        name="independent cache plan", within=raw_path.parent)
    runtime_path = _bound_path(
        parent["cache_runtime_evidence"],
        parent["cache_runtime_evidence_sha256"],
        name="independent cache runtime evidence", within=raw_path.parent)
    runtime = _load_object(runtime_path, name="independent cache runtime evidence")
    _require(
        runtime.get("schema") == client.c4.RUNTIME_EVIDENCE_SCHEMA
        and runtime.get("preparation_completed_before_measurement") is True
        and runtime.get("measurement_includes_preparation_requests") is False
        and runtime.get("ready_for_measurement") is True,
        "independent cache runtime evidence differs",
    )
    validated_node_artifacts = node_module._validate_client_artifacts(
        parent, client_raw_path=raw_path)
    _require(validated_node_artifacts == node["block_artifacts"],
             "independent node/client child bindings differ")

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
        "independent parent block inventory differs",
    )
    blocks = {}
    endpoint_rows = []
    request_rows = []
    generation_history = [[], []]
    block_bindings = []
    block_goodput = []
    semantic_load_blocks = []
    raw_blocks: list[tuple[Mapping[str, object], ForegroundArm]] = []
    for sequence, (key, arm_value, replicate) in enumerate(
        node_module._EXPECTED_BLOCKS
    ):
        arm = ForegroundArm(arm_value)
        entry = artifacts[key]
        path = _bound_path(
            entry["path"], entry["sha256"], name=f"independent block {key}",
            within=raw_path.parent)
        (raw, foreground, block_endpoint_rows, block_request_rows, goodput,
         semantic_summary) = (
            _validate_block(
                path,
                parent_contract=contracts[key],
                block_key=key,
                sequence=sequence,
                arm=arm,
                replicate=replicate,
                manifest=manifest,
                endpoint_fingerprint=endpoint_entry["fingerprint_sha256"],
                semantic_contract=semantic_contract,
            )
        )
        blocks[(replicate, arm_value)] = foreground
        endpoint_rows.extend(block_endpoint_rows)
        request_rows.extend(block_request_rows)
        block_goodput.append(goodput)
        semantic_load_blocks.append(semantic_summary)
        raw_blocks.append((raw, arm))
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
    _require(
        generation_history == [list(range(1, 17)), list(range(1, 17))],
        "independent controllers were not reset once per block",
    )
    _require(len(endpoint_rows) == 384,
             "independent endpoint-phase inventory must contain 384 rows")
    _require(len(request_rows) == 384,
             "independent phase/tenant inventory must contain 384 rows")
    _require(
        len(semantic_load_blocks) == 16
        and all(
            len(row["phase_pair_all_tenants"]) == 12
            and len(row["phase_pair_foreground"]) == 12
            and row["policy_input_used"] is (semantic_contract is not None)
            for row in semantic_load_blocks
        ),
        "independent frontend semantic-load block inventory differs",
    )
    candidate_exercise = _candidate_exercise(
        raw_blocks, semantic_contract=semantic_contract)
    samples = _paired_samples(blocks)
    expected_foreground = sum(
        1 for metadata in contracts[expected_keys[0]]["request_index"].values()
        if metadata["tenant"] == Tenant.FOREGROUND.value
    ) * len(_REPLICATES)
    _require(len(samples) == expected_foreground == 576,
             "independent paired foreground count differs")
    recomputed_paired = client._paired_gate(
        block_paths={key: Path(artifacts[key]["path"]) for key in expected_keys},
        contracts=contracts,
    )
    _require(parent["paired_output_gate"] == recomputed_paired,
             "independent paired gate differs from child evidence")
    metrics, group_rows = _performance_metrics(
        samples, block_goodput, manifest)
    correctness_gates = {
        "node_and_client_correctness": True,
        "all_16_child_hashes_and_contracts_exact": True,
        "all_576_four_arm_pairs_exact": True,
        "all_cache_preparation_outside_measurement": True,
        "all_endpoint_phase_windows_exact": True,
        "controller_reset_and_credit_lifecycle_exact": True,
        "all_frontend_pair_semantic_load_evidence_exact": True,
        "candidate_policy_and_external_credit_exact":
            candidate_exercise["all_pass"] is True,
        "separate_held_out_allocation": True,
        "unchanged_pd_data_plane": True,
    }
    final_gates = {
        "all_correctness_gates_pass": all(correctness_gates.values()),
        **metrics["final_gates"],
    }
    final_scheme_authorized = all(final_gates.values())
    _require(
        node.get("tempo_both_routes_exercised")
        == parent.get("independent_route_diversity_pass")
        == recomputed_paired["tempo_both_routes_exercised"],
        "independent route-diversity evidence differs",
    )
    output: dict[str, object] = {
        "schema": SCHEMA,
        "candidate": dict(candidate),
        "source_node_result": {
            "path": str(result_path), "sha256": expected_result_sha256},
        "run_contract": {
            "path": str(run_contract_path),
            "sha256": node["run_contract_sha256"],
            "fingerprint_sha256": run_contract["fingerprint_sha256"],
        },
        "validation_slurm_job_id": node["slurm_job_id"],
        "calibration_slurm_job_id": node["calibration_slurm_job_id"],
        "separate_held_out_allocation": True,
        "block_artifacts": block_bindings,
        "controller_generation_history": generation_history,
        "endpoint_phase_rows": endpoint_rows,
        "request_phase_tenant_rows": request_rows,
        "block_goodput_windows": block_goodput,
        "frontend_semantic_load_blocks": semantic_load_blocks,
        "candidate_exercise": candidate_exercise,
        "foreground_paired_samples": samples,
        "workload_group_rows": group_rows,
        "correctness_gates": correctness_gates,
        "performance_metrics": metrics,
        "final_gates": final_gates,
        "strongest_fixed_selection_authoritative": True,
        "final_scheme_authorized": final_scheme_authorized,
        "negative_conclusion_authorized": False,
        "negative_conclusion_note": (
            "a failed one-shot validation freezes this candidate but does not "
            "by itself prove every original stopping-rule branch"
        ),
        "verdict": (
            "finalize_frozen_tempo_endpoint_feedback"
            if final_scheme_authorized
            else "independent_validation_failed_no_post_hoc_tuning"
        ),
        "calibration_only": False,
        "post_validation_tuning_allowed": False,
        "performance_claim_allowed": final_scheme_authorized,
        "physical_switch_bottleneck_claim_allowed": False,
        "claim_boundary": run_contract["claim_boundary"],
    }
    output["fingerprint_sha256"] = analysis_fingerprint(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(),
             "refusing to overwrite independent analysis")
    value = analyze(
        args.result, expected_result_sha256=args.expected_result_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "fingerprint_sha256": value["fingerprint_sha256"],
        "final_scheme_authorized": value["final_scheme_authorized"],
        "verdict": value["verdict"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
