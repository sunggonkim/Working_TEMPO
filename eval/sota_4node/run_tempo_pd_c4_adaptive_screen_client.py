#!/usr/bin/env python3
"""Run the post-C4 four-arm adaptive screen with the exact C4 cache mix."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Mapping
from urllib.request import Request, urlopen

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as analysis_module
from eval.sota_4node import analyze_tempo_pd_c4_semantic_epoch_screen as semantic_validator
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as manifest_module
from eval.sota_4node import build_tempo_pd_c4_calibrated_profiles as profiles_module
from eval.sota_4node import replay_tempo_pd_c4_calibrated_controller as replay_module
from eval.sota_4node import run_tempo_pd_c4_fixed_phase_client as c4
from eval.sota_4node import build_tempo_pd_c4_adaptive_run_contract as run_contract_module
from eval.sota_4node import build_tempo_pd_c4_semantic_integration_run_contract as semantic_contract_module
from eval.sota_4node import build_tempo_pd_semantic_epoch_endpoint_profile as semantic_profile_builder
from tempo.pd_cache_state_protocol import build_cache_preparation_plan
from tempo.pd_contention_workload import (
    CacheState,
    ForegroundArm,
    Tenant,
)
from tempo.pd_endpoint_profile import (
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)
from tempo.pd_elastic_profile import load_elastic_profile


SCHEMA = "tempo-pd-c4-adaptive-screen-client-v2"
BLOCK_SCHEMA = "tempo-pd-c4-adaptive-screen-block-v2"
SEMANTIC_SCHEMA = "tempo-pd-c4-semantic-integration-screen-client-v1"
SEMANTIC_BLOCK_SCHEMA = "tempo-pd-c4-semantic-integration-screen-block-v1"
RUN_CONTRACT_SCHEMA = run_contract_module.SCHEMA
RUN_CONTRACT_ENV = "TEMPO_PD_C4_ADAPTIVE_RUN_CONTRACT"
RUN_CONTRACT_SHA_ENV = "TEMPO_PD_C4_ADAPTIVE_RUN_CONTRACT_SHA256"
SEMANTIC_RUN_CONTRACT_ENV = semantic_contract_module.RUN_CONTRACT_ENV
SEMANTIC_RUN_CONTRACT_SHA_ENV = semantic_contract_module.RUN_CONTRACT_SHA_ENV
WORKLOAD_SHA_ENV = "TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256"
CONTROLLER_URLS_ENV = "TEMPO_PD_ENDPOINT_CONTROLLER_URLS"
PROTOCOL_MODULE = c4.PROTOCOL_MODULE
SOURCE_MODULE = c4.SOURCE_MODULE
ARMS = (
    ForegroundArm.LOCAL,
    ForegroundArm.REMOTE,
    ForegroundArm.PREDICTOR,
    ForegroundArm.TEMPO,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--default-max-tokens", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--api-key-env")
    parser.add_argument("--phase-duration-ms", type=float, required=True)
    parser.add_argument("--cooldown-s", type=float, required=True)
    parser.add_argument("--endpoint-evidence-url", action="append", default=[])
    parser.add_argument("--endpoint-controller-url", action="append", default=[])
    return parser.parse_args()


def _resolve_entry(
    contract: Mapping[str, object], name: str,
) -> tuple[Path, Mapping[str, object]]:
    entry = contract.get(name)
    _require(isinstance(entry, Mapping), f"adaptive run contract lacks {name}")
    raw_path = entry.get("path")
    _require(type(raw_path) is str and raw_path,
             f"adaptive {name} path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path = path.resolve()
    _require(path.is_file(), f"adaptive {name} is missing")
    _require(_sha256(path) == entry.get("sha256"),
             f"adaptive {name} digest differs")
    return path, entry


def _runtime_contract_binding() -> tuple[str, str, bool]:
    adaptive_values = (
        os.environ.get(RUN_CONTRACT_ENV),
        os.environ.get(RUN_CONTRACT_SHA_ENV),
    )
    semantic_values = (
        os.environ.get(SEMANTIC_RUN_CONTRACT_ENV),
        os.environ.get(SEMANTIC_RUN_CONTRACT_SHA_ENV),
    )
    adaptive_present = any(adaptive_values)
    semantic_present = any(semantic_values)
    _require(adaptive_present is not semantic_present,
             "exactly one frozen C4 screen contract is required")
    values = semantic_values if semantic_present else adaptive_values
    _require(all(values), "C4 screen contract path/SHA pair is incomplete")
    return str(values[0]), str(values[1]), semantic_present


def _load_contract():
    raw_path, expected_sha, semantic = _runtime_contract_binding()
    path = Path(str(raw_path)).resolve()
    _require(path.is_file() and _sha256(path) == expected_sha,
             "C4 screen run contract digest differs")
    contract = json.loads(path.read_text(encoding="utf-8"))
    builder = semantic_contract_module if semantic else run_contract_module
    expected_schema = (
        semantic_contract_module.SCHEMA if semantic else RUN_CONTRACT_SCHEMA)
    fixed_environment = (
        semantic_contract_module.SEMANTIC_FIXED_RUNTIME_ENVIRONMENT
        if semantic else run_contract_module.ADAPTIVE_FIXED_RUNTIME_ENVIRONMENT)
    _require(
        contract.get("schema") == expected_schema
        and contract.get("fingerprint_sha256")
        == builder.contract_fingerprint(contract)
        and contract.get("performance_claim_allowed") is False
        and contract.get("physical_switch_bottleneck_claim_allowed") is False
        and contract.get("transport") == "LMCacheConnectorV1:UCX"
        and contract.get("unchanged_pd_data_plane") is True
        and contract.get("offline_replay_authorized") is True
        and contract.get("controller_parameter_search_allowed") is False,
        "C4 screen run contract claim or transport differs",
    )
    if semantic:
        _require(
            contract.get("semantic_policy_authorized") is True
            and contract.get("same_allocation_calibration_required") is True
            and contract.get("endpoint_routing_policy") == "semantic_epoch_v1"
            and contract.get("passive_external_credit") is True
            and contract.get("semantic_credit_contract")
            == semantic_profile_builder.SEMANTIC_ROUTING_POLICY,
            "semantic integration policy authorization differs",
        )
    manifest_path, manifest_entry = _resolve_entry(contract, "phase_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema") == manifest_module.SCHEMA
        and manifest.get("fingerprint_sha256")
        == manifest_module.manifest_fingerprint(manifest)
        == manifest_entry.get("fingerprint_sha256")
        and manifest.get("performance_claim_allowed") is False
        and os.environ.get(WORKLOAD_SHA_ENV) == _sha256(manifest_path),
        "adaptive phase manifest binding differs",
    )
    source_path, source_entry = _resolve_entry(contract, "source_workload")
    manifest_source = manifest.get("source_workload")
    _require(isinstance(manifest_source, Mapping),
             "adaptive manifest source workload is missing")
    manifest_source_path = Path(str(manifest_source.get("path", "")))
    if not manifest_source_path.is_absolute():
        manifest_source_path = Path(__file__).resolve().parents[2] / manifest_source_path
    _require(
        manifest_source_path.resolve() == source_path
        and manifest_source.get("sha256") == source_entry.get("sha256"),
        "adaptive source workload binding differs",
    )
    analysis_path, analysis_entry = _resolve_entry(contract, "analysis")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    _require(
        analysis.get("schema") == analysis_module.SCHEMA
        and analysis.get("fingerprint_sha256")
        == analysis_module._analysis_fingerprint(analysis)
        == analysis_entry.get("fingerprint_sha256"),
        "adaptive C4 analysis binding differs",
    )
    elastic_path, elastic_entry = _resolve_entry(contract, "elastic_profile")
    endpoint_path, endpoint_entry = _resolve_entry(
        contract, "endpoint_service_profile")
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    elastic = load_elastic_profile(elastic_path)
    loaded_endpoint = load_endpoint_service_profile(endpoint_path)
    _require(
        elastic.fingerprint_sha256 == elastic_entry.get("fingerprint_sha256")
        and endpoint.get("fingerprint_sha256")
        == endpoint_service_profile_fingerprint(endpoint)
        == endpoint_entry.get("fingerprint_sha256")
        == loaded_endpoint.fingerprint_sha256
        and endpoint.get("elastic_profile_fingerprint_sha256")
        == elastic_entry.get("fingerprint_sha256")
        and endpoint.get("workload_manifest_sha256") == _sha256(manifest_path),
        "adaptive endpoint/Elastic profile binding differs",
    )
    if semantic:
        source_endpoint_path, source_endpoint_entry = _resolve_entry(
            contract, "source_endpoint_service_profile")
        reproduced_endpoint = semantic_profile_builder.build_profile(
            source_endpoint_path,
            expected_base_sha256=str(source_endpoint_entry["sha256"]),
            profile_id=str(endpoint.get("profile_id", "")),
        )
        _require(
            endpoint == reproduced_endpoint
            and loaded_endpoint.routing_policy is not None
            and loaded_endpoint.routing_policy.as_dict()
            == contract["semantic_credit_contract"]
            and endpoint_entry.get("derived_from_sha256")
            == source_endpoint_entry.get("sha256"),
            "semantic endpoint profile derivation or policy differs",
        )
    receipt_path, receipt_entry = _resolve_entry(contract, "profile_receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(
        receipt.get("schema") == profiles_module.SCHEMA
        and receipt.get("fingerprint_sha256")
        == profiles_module._receipt_fingerprint(receipt)
        == receipt_entry.get("fingerprint_sha256"),
        "adaptive profile receipt binding differs",
    )
    replay_path, replay_entry = _resolve_entry(contract, "offline_replay")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    _require(
        replay.get("schema") == replay_module.SCHEMA
        and replay.get("fingerprint_sha256")
        == replay_module.replay_fingerprint(replay)
        == replay_entry.get("fingerprint_sha256")
        and replay.get("live_adaptive_screen_authorized") is True
        and replay.get("performance_claim_allowed") is False,
        "offline replay did not authorize the adaptive screen",
    )
    _resolve_entry(contract, "fixed_c4_implementation_contract")
    _resolve_entry(contract, "adaptive_implementation_contract")
    if semantic:
        _resolve_entry(contract, "semantic_integration_implementation_contract")
        _resolve_entry(contract, "semantic_exploratory_analysis")
        _resolve_entry(contract, "semantic_exploratory_run_contract")
    _require(
        contract.get("fixed_runtime_environment")
        == dict(sorted(fixed_environment.items())),
        "adaptive fixed runtime environment differs",
    )
    for name, expected in (
        ("TEMPO_ELASTIC_PD_PROFILE", str(elastic_path)),
        ("TEMPO_PD_ENDPOINT_SERVICE_PROFILE", str(endpoint_path)),
        (WORKLOAD_SHA_ENV, _sha256(manifest_path)),
    ):
        actual = os.environ.get(name)
        if name.endswith("PROFILE"):
            actual = str(Path(actual).resolve()) if actual else actual
        _require(actual == expected, f"adaptive runtime {name} differs")
    for name, expected in fixed_environment.items():
        _require(os.environ.get(name) == expected,
                 f"adaptive runtime {name} differs")
    screen = {
        "semantic": semantic,
        "client_schema": SEMANTIC_SCHEMA if semantic else SCHEMA,
        "block_schema": SEMANTIC_BLOCK_SCHEMA if semantic else BLOCK_SCHEMA,
        "stage_name": (
            "tempo_pd_c4_semantic_integration_screen"
            if semantic else "tempo_pd_c4_adaptive_screen"),
        "fixed_runtime_environment": fixed_environment,
    }
    return path, contract, manifest_path, manifest, source_path, screen


def _controller_get(url: str) -> dict[str, object]:
    with urlopen(url.rstrip("/") + "/tempo/endpoint_controller", timeout=10.0) as response:
        _require(response.status == 200, "endpoint controller GET failed")
        value = json.loads(response.read().decode("utf-8"))
    _require(isinstance(value, dict) and value.get("controller") is not None,
             "endpoint controller state is unavailable")
    return value


def _controller_reset(url: str) -> dict[str, object]:
    request = Request(
        url.rstrip("/") + "/tempo/reset_endpoint_controller",
        data=b"",
        method="POST",
    )
    with urlopen(request, timeout=10.0) as response:
        _require(response.status == 200, "endpoint controller reset failed")
        value = json.loads(response.read().decode("utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("success") is True
        and type(value.get("controller_generation")) is int
        and value["controller_generation"] >= 1
        and value.get("controller", {}).get("inflight") == 0
        and all(not item for item in value["controller"]["resources"].values()),
        "endpoint controller reset evidence differs",
    )
    return value


def _controllers_quiescent(values: list[Mapping[str, object]]) -> bool:
    return all(
        value.get("queued_requests") == 0
        and value.get("controller", {}).get("inflight") == 0
        and all(not item for item in value["controller"]["resources"].values())
        for value in values
    )


def _block_order(
    manifest: Mapping[str, object],
) -> tuple[tuple[ForegroundArm, int], ...]:
    raw_orders = manifest.get("arm_order_by_replicate")
    expected = [
        list(values) for values in manifest_module.ARM_ORDER_BY_REPLICATE
    ]
    _require(raw_orders == expected,
             "adaptive screen arm order differs from preregistration")
    order = tuple(
        (ForegroundArm(value), replicate)
        for replicate, values in enumerate(raw_orders)
        for value in values
    )
    _require(
        len(order) == 8
        and {
            (arm, replicate) for arm, replicate in order
        } == {
            (arm, replicate)
            for arm in ARMS for replicate in range(2)
        },
        "adaptive screen block inventory differs",
    )
    return order


def _validate_dynamic_decision(
    decision: Mapping[str, object], metadata: Mapping[str, object],
    *, block_arm: ForegroundArm, request_id: str | None = None,
    semantic_contract: Mapping[str, object] | None = None,
) -> str:
    route = decision.get("route")
    _require(route in {c4._LOCAL_ROUTE, c4._REMOTE_ROUTE},
             "adaptive screen request lacks a canonical route")
    tenant = Tenant(str(metadata["tenant"]))
    scheduled_arm = ForegroundArm(str(metadata["arm"]))
    state = CacheState(str(metadata["cache_state"]))
    if tenant is Tenant.FOREGROUND:
        _require(scheduled_arm is block_arm,
                 "foreground metadata arm differs from its block")
        if block_arm is ForegroundArm.LOCAL:
            _require(route == c4._LOCAL_ROUTE,
                     "fixed-local screen request escaped its route")
        elif block_arm is ForegroundArm.REMOTE:
            _require(route == c4._REMOTE_ROUTE,
                     "fixed-remote screen request escaped its route")
        elif state in {CacheState.D_ONLY, CacheState.BOTH}:
            _require(route == c4._LOCAL_ROUTE,
                     "D_ONLY/BOTH adaptive request ignored decoder residency")
    else:
        _require(
            scheduled_arm in {ForegroundArm.LOCAL, ForegroundArm.REMOTE},
            "contention tenant is not route pinned",
        )
        expected_background_route = (
            c4._LOCAL_ROUTE
            if scheduled_arm is ForegroundArm.LOCAL else c4._REMOTE_ROUTE)
        _require(route == expected_background_route,
                 "contention tenant escaped its fixed route")
    route_metadata = dict(metadata)
    route_metadata["arm"] = (
        "local" if route == c4._LOCAL_ROUTE else "remote")
    c4._validate_measured_decision(decision, route_metadata)
    if tenant is Tenant.FOREGROUND:
        if block_arm is ForegroundArm.TEMPO:
            if semantic_contract is not None:
                semantic_route, _reason, _generation = (
                    semantic_validator._validate_semantic_decision(
                        decision, contract=semantic_contract)
                )
                _require(
                    semantic_route == route
                    and decision.get("arm") == "tempo"
                    and decision.get("endpoint_feedback_mode") == "adaptive"
                    and decision.get("endpoint_policy_applied") is True
                    and decision.get("endpoint_feedback_accepted") is True,
                    "semantic TEMPO endpoint-feedback provenance differs",
                )
            else:
                history = decision.get("endpoint_decision_history")
                _require(
                    decision.get("arm") == "tempo"
                    and decision.get("endpoint_feedback_mode") == "adaptive"
                    and decision.get("endpoint_policy_applied") is True
                    and decision.get("endpoint_decision_route") == route
                    and type(decision.get("endpoint_decision_attempts")) is int
                    and decision["endpoint_decision_attempts"] >= 1
                    and isinstance(history, list)
                    and len(history) == decision["endpoint_decision_attempts"]
                    and history[-1].get("route") == route
                    and decision.get("endpoint_request_local_allowed") is True
                    and decision.get("endpoint_request_remote_allowed")
                    is (state not in {CacheState.D_ONLY, CacheState.BOTH})
                    and decision.get("admission_credit_release_event")
                    == "first_response_chunk"
                    and type(decision.get("admission_credit_released_ns")) is int
                    and decision["admission_credit_released_ns"] > 0
                    and decision.get("endpoint_feedback_event")
                    == "first_response_chunk"
                    and decision.get("endpoint_feedback_accepted") is True,
                    "TEMPO endpoint-feedback provenance differs",
                )
        else:
            expected_reason = {
                (ForegroundArm.LOCAL, c4._LOCAL_ROUTE):
                    "fixed_always_local",
                (ForegroundArm.REMOTE, c4._REMOTE_ROUTE):
                    "fixed_official_lmcache_remote",
                (ForegroundArm.PREDICTOR, c4._LOCAL_ROUTE): (
                    "predictor_decoder_residency_local"
                    if state in {CacheState.D_ONLY, CacheState.BOTH}
                    else "predictor_local_safe"),
                (ForegroundArm.PREDICTOR, c4._REMOTE_ROUTE):
                    "predictor_remote_lower_bound",
            }.get((block_arm, route))
            _require(
                expected_reason is not None
                and decision.get("reason") == expected_reason,
                "non-TEMPO foreground policy provenance differs",
            )
    if semantic_contract is not None and tenant is not Tenant.FOREGROUND:
        _require(type(request_id) is str and bool(request_id),
                 "semantic external request ID is missing")
        external_route, _lookup = semantic_validator._validate_external_decision(
            decision, metadata=metadata, request_id=request_id)
        _require(
            external_route == route
            and decision.get("endpoint_policy_applied") is False
            and decision.get("endpoint_decision_attempts") == 0,
            "semantic external request used active endpoint policy state",
        )
    elif block_arm is not ForegroundArm.TEMPO or tenant is not Tenant.FOREGROUND:
        _require(
            decision.get("endpoint_policy_applied") is False
            and decision.get("endpoint_decision_attempts") == 0
            and decision.get("endpoint_feedback_event") is None
            and decision.get("admission_credit_release_event") is None
            and decision.get("admission_credit_released_ns") is None,
            "non-TEMPO request used endpoint-feedback state",
        )
    return str(route)


def _validate_block(
    raw_path: Path, block: Mapping[str, object], endpoint_evidence: object,
    *, controller_reset: list[Mapping[str, object]],
    controller_before: list[Mapping[str, object]],
    controller_after: list[Mapping[str, object]],
    semantic_contract: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    raw, requests, decisions = c4._artifact_rows(raw_path)
    request_index = block["request_index"]
    _require(set(requests) == set(decisions) == set(request_index),
             "adaptive screen block IDs differ")
    _require(all(row.get("valid") is True for row in requests.values()),
             "adaptive screen contains an invalid stream")
    c4._validate_c4_endpoint_evidence(endpoint_evidence)
    _require(_controllers_quiescent(controller_before)
             and _controllers_quiescent(controller_after),
             "adaptive screen controller is not quiescent at a boundary")
    reset_generations = [value["controller_generation"]
                         for value in controller_reset]
    before_generations = [value["controller_generation"]
                          for value in controller_before]
    after_generations = [value["controller_generation"]
                         for value in controller_after]
    _require(reset_generations == before_generations == after_generations,
             "adaptive screen controller generation changed within a block")

    arm = block["arm"]
    route_counts = Counter()
    foreground_metrics = []
    phase_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    tempo_foreground = 0
    external_requests = 0
    for request_id, metadata in request_index.items():
        decision = decisions[request_id]
        route = _validate_dynamic_decision(
            decision, metadata, block_arm=arm, request_id=request_id,
            semantic_contract=semantic_contract)
        route_counts[route] += 1
        if metadata["tenant"] == Tenant.FOREGROUND.value:
            metrics = c4._request_service_metrics(requests[request_id])
            foreground_metrics.append(metrics)
            phase_metrics[str(metadata["phase"])].append(metrics)
            if arm is ForegroundArm.TEMPO:
                tempo_foreground += 1
        else:
            external_requests += 1
    if arm is ForegroundArm.TEMPO:
        active_samples = sum(
            int(value["controller"]["completed"])
            for value in controller_after)
        _require(active_samples == tempo_foreground,
                 "TEMPO first-response completion count differs")
    else:
        _require(all(
            int(value["controller"]["completed"]) == 0
            for value in controller_after
        ), "non-TEMPO arm changed endpoint controller state")

    passive_completed = sum(
        int(value["controller"].get("passive_completed", 0))
        for value in controller_after)
    if semantic_contract is not None:
        _require(
            passive_completed == external_requests,
            "semantic passive completion count differs from external requests",
        )
    else:
        _require(passive_completed == 0,
                 "instant-score screen observed passive endpoint feedback")

    contract = {
        "schema": (
            SEMANTIC_BLOCK_SCHEMA
            if semantic_contract is not None else BLOCK_SCHEMA),
        "sequence": block["sequence"],
        "arm": arm.value,
        "replicate": block["replicate"],
        "semantic_schedule_sha256": block["schedule_sha256"],
        "request_index": request_index,
        "controller_generations": reset_generations,
        "all_requests_valid": True,
        "decision_cache_states_exact": True,
        "completion_cache_evidence_exact": True,
        "phase_aligned_endpoint_evidence": True,
        "controller_reset_before_block_exact": True,
        "controller_quiescent_after_block": True,
        "one_way_route_commit_exact": True,
        "unchanged_pd_data_plane": True,
        "performance_claim_allowed": False,
    }
    if semantic_contract is not None:
        endpoint = semantic_contract["endpoint_service_profile"]
        _require(isinstance(endpoint, Mapping),
                 "semantic endpoint profile binding is missing")
        contract.update({
            "endpoint_routing_policy": "semantic_epoch_v1",
            "endpoint_service_profile_fingerprint_sha256": endpoint[
                "fingerprint_sha256"],
            "semantic_credit_contract": semantic_contract[
                "semantic_credit_contract"],
            "passive_external_endpoint_credit": True,
            "semantic_decisions_exact": True,
            "external_credit_lifecycle_exact": True,
            "external_route_pinned_requests": external_requests,
            "passive_completions": passive_completed,
        })
    raw["c4_adaptive_screen_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    raw["endpoint_controller_reset"] = controller_reset
    raw["endpoint_controller_before"] = controller_before
    raw["endpoint_controller_after"] = controller_after
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def summary(values: list[dict[str, float]]) -> dict[str, object]:
        return {
            "requests": len(values),
            **{
                f"{name}_median_ms": statistics.median(
                    row[f"{name}_ms"] for row in values)
                for name in ("ttft", "e2e", "tpot")
            },
            **{
                f"{name}_p99_ms": sorted(
                    row[f"{name}_ms"] for row in values
                )[max(0, math.ceil(0.99 * len(values)) - 1)]
                for name in ("ttft", "e2e", "tpot")
            },
        }

    return contract, {
        "sequence": block["sequence"],
        "arm": arm.value,
        "replicate": block["replicate"],
        "route_counts": dict(route_counts),
        "foreground": summary(foreground_metrics),
        "foreground_by_phase": {
            phase.value: summary(phase_metrics[phase.value])
            for phase in c4.manifest_builder.PHASES
        },
    }


def _paired_gate(
    *, block_paths: Mapping[str, Path],
    contracts: Mapping[str, Mapping[str, object]],
    block_schema: str = BLOCK_SCHEMA,
) -> dict[str, object]:
    _require(
        set(block_paths) == set(contracts) and len(contracts) == 8,
        "adaptive screen child artifact inventory differs",
    )
    samples: dict[tuple[int, str], dict[str, tuple[str, str]]] = defaultdict(dict)
    schedules: dict[int, set[str]] = defaultdict(set)
    tempo_routes = Counter()
    observed_blocks = set()
    for key, contract in contracts.items():
        _require(
            contract.get("schema") == block_schema
            and contract.get("all_requests_valid") is True
            and contract.get("completion_cache_evidence_exact") is True
            and contract.get("phase_aligned_endpoint_evidence") is True
            and contract.get("controller_reset_before_block_exact") is True
            and contract.get("controller_quiescent_after_block") is True,
            f"adaptive screen block contract differs: {key}",
        )
        replicate = int(contract["replicate"])
        arm = str(contract["arm"])
        observed_blocks.add((arm, replicate))
        schedules[replicate].add(str(contract["semantic_schedule_sha256"]))
        raw = json.loads(block_paths[key].read_text(encoding="utf-8"))
        requests = {row["request_id"]: row for row in raw["requests"]}
        decisions = {row["request_id"]: row for row in raw["router_decisions"]}
        for request_id, metadata in contract["request_index"].items():
            if metadata["tenant"] != Tenant.FOREGROUND.value:
                continue
            pair_key = str(metadata["pair_key"])
            samples[(replicate, pair_key)][arm] = (
                str(requests[request_id]["output_text_sha256"]),
                str(metadata["prompt_token_sha256"]),
            )
            if arm == ForegroundArm.TEMPO.value:
                tempo_routes[str(decisions[request_id]["route"])] += 1
    expected_arms = {arm.value for arm in ARMS}
    _require(
        observed_blocks == {
            (arm.value, replicate)
            for arm in ARMS for replicate in range(2)
        },
        "adaptive screen arm/replicate inventory differs",
    )
    _require(
        set(schedules) == {0, 1}
        and all(len(value) == 1 for value in schedules.values()),
             "adaptive screen semantic schedules differ within a replicate")
    _require(bool(samples), "adaptive screen has no paired foreground requests")
    _require(all(
        set(by_arm) == expected_arms and len(set(by_arm.values())) == 1
        for by_arm in samples.values()
    ), "adaptive screen paired prompt/output digests differ")
    return {
        "paired_foreground_requests": len(samples),
        "all_four_arms_present": True,
        "semantic_schedules_exact_within_replicate": True,
        "prompt_and_output_digests_exact": True,
        "tempo_route_counts": {
            c4._LOCAL_ROUTE: tempo_routes[c4._LOCAL_ROUTE],
            c4._REMOTE_ROUTE: tempo_routes[c4._REMOTE_ROUTE],
        },
        "tempo_both_routes_exercised": all(
            tempo_routes[route] > 0
            for route in (c4._LOCAL_ROUTE, c4._REMOTE_ROUTE)),
        "performance_claim_allowed": False,
    }


def _measured(
    args: argparse.Namespace, tokenizer, templates,
    contract_path: Path, manifest_path: Path, manifest: Mapping[str, object],
    contract: Mapping[str, object], screen: Mapping[str, object],
) -> int:
    _require(os.environ.get("TEMPO_VLLM_DECODER_PREFIX_CACHING") == "1",
             "adaptive screen requires decoder prefix caching")
    _require(os.environ.get("TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY") == "1",
             "adaptive screen requires replicated warm affinity")
    _require(len(args.endpoint_evidence_url) == 4,
             "adaptive screen requires four endpoint probes")
    _require(len(args.endpoint_controller_url) == 2,
             "adaptive screen requires two endpoint controllers")
    _require(args.phase_duration_ms == float(manifest["phase_duration_ms"])
             and args.request_rate == float(manifest["foreground_rate_per_s"])
             and args.cooldown_s == float(manifest["cooldown_s"]),
             "adaptive screen runtime workload differs from its manifest")
    order = _block_order(manifest)

    semantic_contract = contract if screen["semantic"] is True else None
    root = args.output.parent / str(screen["stage_name"])
    workload_root = root / "workloads"
    root.mkdir()
    workload_root.mkdir()
    factory = c4._PromptFactory(tokenizer, templates)
    blocks = [
        c4._materialize_block(
            sequence=sequence,
            arm=arm,
            replicate=replicate,
            manifest=manifest,
            factory=factory,
        )
        for sequence, (arm, replicate) in enumerate(order)
    ]
    plan = build_cache_preparation_plan(
        item for block in blocks for item in block["items"])
    plan_path = root / "cache_preparation_plan.json"
    plan_path.write_text(
        json.dumps(plan.manifest_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    source_workload = workload_root / "source_prepare.jsonl"
    source_raw = root / "source_prepare.raw.json"
    c4._write_rows(source_workload, list(plan.source_probe_rows))
    c4._run(c4._stream_command(
        args, module=SOURCE_MODULE, workload=source_workload,
        output=source_raw, run_id=f"{args.run_id}-source-prepare",
        max_workers=1))
    source_evidence = c4.validate_source_preparation(source_raw, plan)
    reset = c4._reset_decoder_prefix_cache(args.base_url)

    decoder_workload = workload_root / "decoder_prepare.jsonl"
    decoder_raw = root / "decoder_prepare.raw.json"
    c4._write_rows(decoder_workload, list(plan.decoder_prepare_rows))
    decoder_env = dict(os.environ)
    decoder_env[c4.protocol_client.PHASE_ENV] = "decoder_prepare"
    decoder_env[c4.protocol_client.PLAN_ENV] = str(plan_path.resolve())
    decoder_env.pop(c4.protocol_client.EVIDENCE_ENV, None)
    c4._run(c4._stream_command(
        args, module=PROTOCOL_MODULE, workload=decoder_workload,
        output=decoder_raw, run_id=f"{args.run_id}-decoder-prepare",
        max_workers=1), env=decoder_env)
    decoder_evidence = c4.validate_decoder_preparation(decoder_raw, plan)
    runtime_evidence = c4._runtime_evidence(
        plan_path=plan_path, plan=plan,
        source=source_evidence, reset=reset, decoder=decoder_evidence)
    evidence_path = root / "cache_runtime_evidence.json"
    evidence_path.write_text(
        json.dumps(runtime_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    measured_env = dict(os.environ)
    measured_env[c4.protocol_client.PHASE_ENV] = "measured"
    measured_env[c4.protocol_client.PLAN_ENV] = str(plan_path.resolve())
    measured_env[c4.protocol_client.EVIDENCE_ENV] = str(evidence_path.resolve())
    artifacts = {}
    contracts = {}
    summaries = []
    block_paths = {}
    evidence_args = argparse.Namespace(**vars(args))
    for block in blocks:
        sequence = int(block["sequence"])
        arm = block["arm"]
        replicate = int(block["replicate"])
        key = f"{sequence:02d}_{arm.value}_r{replicate}"
        controller_reset = [
            _controller_reset(url) for url in args.endpoint_controller_url]
        controller_before = [
            _controller_get(url) for url in args.endpoint_controller_url]
        workload = workload_root / f"{key}.jsonl"
        raw_path = root / f"{key}.raw.json"
        c4._write_rows(workload, block["rows"])
        endpoint_evidence = c4._run_with_endpoint_evidence(
            c4._stream_command(
                args, module=PROTOCOL_MODULE, workload=workload,
                output=raw_path, run_id=f"{args.run_id}-{key}",
                max_workers=args.max_workers),
            args=evidence_args,
            env=measured_env,
            start_marker=(root / f"{key}.measurement-start.json").resolve(),
            first_arrival_offset_ms=min(
                float(row["arrival_offset_ms"]) for row in block["rows"]),
        )
        controller_after = [
            _controller_get(url) for url in args.endpoint_controller_url]
        block_contract, summary = _validate_block(
            raw_path,
            block,
            endpoint_evidence,
            controller_reset=controller_reset,
            controller_before=controller_before,
            controller_after=controller_after,
            semantic_contract=semantic_contract,
        )
        artifacts[key] = c4._artifact_binding(raw_path)
        contracts[key] = block_contract
        summaries.append(summary)
        block_paths[key] = raw_path.resolve()
        if sequence + 1 < len(blocks):
            time.sleep(args.cooldown_s)

    paired = _paired_gate(
        block_paths=block_paths, contracts=contracts,
        block_schema=str(screen["block_schema"]))
    payload = {
        "schema": screen["client_schema"],
        "run_id": args.run_id,
        "run_contract": str(contract_path),
        "run_contract_sha256": _sha256(contract_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "cache_plan": str(plan_path.resolve()),
        "cache_plan_sha256": _sha256(plan_path),
        "cache_runtime_evidence": str(evidence_path.resolve()),
        "cache_runtime_evidence_sha256": _sha256(evidence_path),
        "block_order": [
            {"arm": arm.value, "replicate": replicate}
            for arm, replicate in order
        ],
        "artifacts": artifacts,
        "contracts": contracts,
        "summaries": summaries,
        "paired_output_gate": paired,
        "blocks_completed": len(artifacts),
        "live_screen_correctness_pass": (
            len(artifacts) == 8
            and paired["all_four_arms_present"] is True
            and paired["prompt_and_output_digests_exact"] is True
        ),
        "live_screen_route_diversity_pass":
            paired["tempo_both_routes_exercised"],
        "calibration_only": True,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "unchanged_pd_data_plane": True,
    }
    if semantic_contract is not None:
        payload.update({
            "endpoint_routing_policy": "semantic_epoch_v1",
            "endpoint_service_profile_fingerprint_sha256":
                semantic_contract["endpoint_service_profile"][
                    "fingerprint_sha256"],
            "semantic_credit_contract": semantic_contract[
                "semantic_credit_contract"],
            "passive_external_endpoint_credit": True,
            "semantic_policy_authorized": True,
        })
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


def main() -> int:
    args = _parse()
    _require(args.mode == "tempo_auto", "adaptive screen requires tempo_auto")
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    _require(args.model.is_absolute(), "model path must be absolute")
    contract_path, contract, manifest_path, manifest, source_path, screen = (
        _load_contract())
    is_warmup = args.run_id.endswith("-warmup")
    if is_warmup:
        _require(
            args.workload.resolve().parent == args.output.resolve().parent
            and args.workload.name == "warmup.jsonl",
            "adaptive warmup must be lifecycle-local",
        )
    else:
        _require(args.workload.resolve() == source_path,
                 "adaptive runtime source workload differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True)
    templates = c4._load_templates(args.workload, tokenizer)
    if is_warmup:
        return c4.fixed._warmup(args, tokenizer, templates)
    return _measured(
        args, tokenizer, templates, contract_path, manifest_path, manifest,
        contract, screen)


if __name__ == "__main__":
    raise SystemExit(main())
