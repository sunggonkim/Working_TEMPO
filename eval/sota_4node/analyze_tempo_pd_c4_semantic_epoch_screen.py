#!/usr/bin/env python3
"""Fail-closed verdict for the calibration-only C4 semantic-epoch screen."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_phase_screen as base_analysis
from eval.sota_4node import build_tempo_pd_c4_semantic_epoch_run_contract as contract_builder
from eval.sota_4node import run_tempo_pd_c4_phase_screen_client as phase_client
from tempo.pd_endpoint_profile import SCHEMA_V2, load_endpoint_service_profile


SCHEMA = "tempo-pd-c4-semantic-epoch-screen-analysis-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROUTE = "decoder_local_chunked_prefill"
REMOTE_ROUTE = "official_lmcache_remote_prefill"
LOAD_SCHEMA = "tempo-frontend-semantic-load-v1"
LOAD_SOURCE = "frontend_pair_ledger_request_start_to_http_eof"
EPOCH_SCHEMA = "tempo-pd-semantic-epoch-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, *, name: str) -> dict[str, object]:
    _require(path.is_file(), f"{name} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _canonical_sha(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _bound_result_file(raw: object, *, root: Path, name: str) -> Path:
    _require(type(raw) is str and Path(raw).is_absolute(),
             f"{name} must be an absolute path")
    path = Path(raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} escapes the result root") from exc
    _require(path.is_file(), f"{name} is missing")
    return path


def _implementation_path(raw: object, *, name: str) -> Path:
    _require(type(raw) is str and raw, f"{name} path is missing")
    pure = PurePosixPath(raw)
    _require(
        not pure.is_absolute() and ".." not in pure.parts and str(pure) == raw,
        f"{name} path is not canonical and relative",
    )
    path = (REPO_ROOT / raw).resolve()
    _require(path.is_file(), f"{name} is missing")
    return path


def _validate_implementation(contract: Mapping[str, object]) -> int:
    entries = contract.get("implementation")
    _require(isinstance(entries, list) and entries,
             "semantic implementation inventory is missing")
    seen = set()
    for index, entry in enumerate(entries):
        _require(
            isinstance(entry, Mapping)
            and set(entry) == {"path", "sha256"},
            f"semantic implementation[{index}] binding differs",
        )
        raw = entry["path"]
        _require(type(raw) is str and raw not in seen,
                 "semantic implementation paths are duplicated")
        path = _implementation_path(
            raw, name=f"semantic implementation[{index}]")
        _require(
            _sha256(path) == _canonical_sha(
                entry["sha256"], name=f"semantic implementation[{index}] SHA"),
            f"semantic implementation drifted: {raw}",
        )
        seen.add(raw)
    required = {
        "tempo/pd_endpoint_controller.py",
        "tempo/pd_endpoint_profile.py",
        "eval/sota_4node/tempo_pd_elastic_frontend.py",
        "eval/sota_4node/tempo_pd_elastic_router.py",
        "eval/sota_4node/run_tempo_pd_c4_phase_screen_client.py",
        "eval/sota_4node/vllm_lmcache_pd_c4_phase_screen_node.py",
        "eval/sota_4node/analyze_tempo_pd_c4_phase_screen.py",
        "eval/sota_4node/analyze_tempo_pd_c4_semantic_epoch_screen.py",
        "eval/sota_4node/run_tempo_pd_c4_semantic_epoch_screen_in_allocation.sh",
        "eval/sota_4node/build_tempo_pd_semantic_epoch_endpoint_profile.py",
    }
    _require(required <= seen,
             "semantic implementation omits a verdict or live-path file")
    phase_client._validate_semantic_implementation(dict(contract))
    return len(entries)


def _validate_semantic_decision(
    decision: Mapping[str, object], *, contract: Mapping[str, object],
) -> tuple[str, str, int]:
    credit = contract["semantic_credit_contract"]
    _require(isinstance(credit, Mapping),
             "semantic credit contract is missing")
    route = decision.get("route")
    _require(route in {LOCAL_ROUTE, REMOTE_ROUTE},
             "semantic decision lacks a committed route")
    _require(
        decision.get("endpoint_routing_policy") == "semantic_epoch_v1"
        and decision.get("semantic_epoch_applied") is True
        and decision.get("semantic_epoch_schema") == EPOCH_SCHEMA
        and decision.get("semantic_epoch_policy") == "semantic_epoch_v1"
        and decision.get("frontend_semantic_load_schema") == LOAD_SCHEMA
        and decision.get("frontend_semantic_load_source") == LOAD_SOURCE,
        "semantic decision provenance differs",
    )
    endpoint_entry = contract.get("endpoint_service_profile")
    _require(
        isinstance(endpoint_entry, Mapping)
        and decision.get("semantic_epoch_profile_fingerprint_sha256")
        == endpoint_entry.get("fingerprint_sha256")
        and decision.get("endpoint_service_profile_fingerprint_sha256")
        == endpoint_entry.get("fingerprint_sha256"),
        "semantic decision profile binding differs",
    )
    active = decision.get("semantic_epoch_active_requests_before")
    decode = decision.get("semantic_epoch_decode_tokens_before")
    capacity = decision.get("semantic_epoch_max_num_seqs")
    _require(
        type(active) is int and active >= 0
        and type(decode) is int and decode >= 0
        and type(capacity) is int and capacity > 0
        and active == decision.get("frontend_semantic_active_requests_before")
        and decode == decision.get("frontend_semantic_decode_tokens_before")
        and capacity == decision.get("frontend_semantic_max_num_seqs"),
        "semantic decision load evidence differs",
    )
    high_numerator = credit["decoder_high_water_numerator"]
    high_denominator = credit["decoder_high_water_denominator"]
    low_numerator = credit["decoder_low_water_numerator"]
    low_denominator = credit["decoder_low_water_denominator"]
    _require(
        decision.get("semantic_epoch_decoder_high_water")
        is (active * high_denominator >= capacity * high_numerator)
        and decision.get("semantic_epoch_decoder_low_water")
        is (active * low_denominator < capacity * low_numerator)
        and decision.get("semantic_epoch_decoder_high_water_numerator")
        == high_numerator
        and decision.get("semantic_epoch_decoder_high_water_denominator")
        == high_denominator
        and decision.get("semantic_epoch_decoder_low_water_numerator")
        == low_numerator
        and decision.get("semantic_epoch_decoder_low_water_denominator")
        == low_denominator,
        "semantic watermarks differ from the frozen contract",
    )
    overload = decision.get("semantic_epoch_overload_multiplier")
    _require(
        not isinstance(overload, bool)
        and isinstance(overload, (int, float))
        and math.isfinite(float(overload))
        and float(overload)
        == float(credit["remote_overload_service_stretch"]),
        "semantic remote overload guard differs",
    )
    _require(
        decision.get("semantic_epoch_confirmation_requests")
        == credit["epoch_confirmation_requests"]
        and decision.get(
            "semantic_epoch_remote_external_credit_close_fraction")
        == credit["remote_external_credit_close_fraction"],
        "semantic epoch confirmation or external-credit guard differs",
    )
    credit_epoch = (
        credit.get("local_external_credit_opens_epoch") is True)
    if credit_epoch:
        local_external = decision.get(
            "semantic_epoch_local_external_utilization")
        _require(
            isinstance(local_external, (int, float))
            and not isinstance(local_external, bool)
            and math.isfinite(float(local_external))
            and decision.get("semantic_epoch_decision_basis")
            == "local_external_credit_nonzero"
            and decision.get(
                "semantic_epoch_local_external_credit_pressure")
            is (float(local_external) > 0.0)
            and decision.get(
                "semantic_epoch_local_external_credit_opens_epoch") is True
            and decision.get(
                "semantic_epoch_frontend_decoder_watermarks_policy_input")
            is False,
            "semantic credit epoch decision basis differs",
        )
    selected_endpoint = (
        "decoder_local_chunked_prefill"
        if route == LOCAL_ROUTE else "official_lmcache_remote_prefill")
    _require(
        decision.get("semantic_epoch_route_after") == selected_endpoint
        and decision.get("endpoint_decision_route") == route
        and decision.get("reason") == decision.get("semantic_epoch_reason")
        and decision.get("endpoint_request_local_allowed")
        is (route == LOCAL_ROUTE)
        and decision.get("endpoint_request_remote_allowed")
        is (route == REMOTE_ROUTE),
        "semantic route latch did not own the endpoint decision",
    )
    generation = decision.get("semantic_epoch_generation")
    reason = decision.get("semantic_epoch_reason")
    _require(type(generation) is int and generation >= 0,
             "semantic generation differs")
    expected_reason_prefix = (
        "semantic_credit_epoch_" if credit_epoch else "semantic_epoch_")
    _require(type(reason) is str and reason.startswith(expected_reason_prefix),
             "semantic reason differs")
    _require(
        decision.get("admission_credit_release_event")
        == "first_response_chunk"
        and decision.get("admission_credit_released_ns") is not None
        and decision.get("endpoint_feedback_event") == "first_response_chunk"
        and decision.get("endpoint_external_credit_registered") is False,
        "semantic TEMPO credit lifecycle differs",
    )
    return str(route), str(reason), generation


def _validate_external_decision(
    decision: Mapping[str, object], *, metadata: Mapping[str, object],
    request_id: str,
) -> tuple[str, str]:
    route = decision.get("route")
    _require(route in {LOCAL_ROUTE, REMOTE_ROUTE},
             "route-pinned external request lacks a committed route")
    _require(
        "-endpoint-observed-" in request_id
        and decision.get("endpoint_passive_feedback_enabled") is True
        and decision.get("endpoint_passive_registered") is True
        and decision.get("endpoint_external_credit_registered") is True
        and decision.get("endpoint_feedback_passive") is True
        and decision.get("endpoint_feedback_event")
        == "external_credit_passive_first_response_chunk"
        and decision.get("admission_credit_scope") is None
        and decision.get("admission_credit_release_event") is None,
        f"route-pinned external credit differs: {request_id}",
    )
    prompt = metadata.get("prompt_tokens")
    output = metadata.get("output_tokens")
    cache = metadata.get("cache_state")
    source_prompt = decision.get(
        "endpoint_passive_service_source_prompt_tokens")
    source_output = decision.get(
        "endpoint_passive_service_source_output_tokens")
    source_cache = decision.get(
        "endpoint_passive_service_source_cache_residency")
    mode = decision.get("endpoint_passive_service_lookup_mode")
    _require(
        type(prompt) is int and prompt >= 2
        and type(output) is int and output >= 2
        and type(source_prompt) is int and source_prompt >= prompt
        and type(source_output) is int and source_output >= output,
        f"external service proxy geometry differs: {request_id}",
    )
    expected = {
        "miss": ("miss_via_prefill_only_geometry_ceiling", "prefill_only"),
        "p_only": ("same_residency_geometry_ceiling", "prefill_only"),
    }.get(cache)
    _require(expected is not None and (mode, source_cache) == expected,
             f"external service proxy provenance differs: {request_id}")
    return str(route), str(mode)


def analyze(
    *, result_path: Path, expected_result_sha256: str,
    base_analysis_path: Path, expected_base_analysis_sha256: str,
) -> dict[str, object]:
    result_path = result_path.resolve()
    base_analysis_path = base_analysis_path.resolve()
    _require(
        _sha256(result_path)
        == _canonical_sha(expected_result_sha256, name="result SHA"),
        "semantic result digest differs",
    )
    result = _load(result_path, name="semantic node result")
    _require(
        result.get("schema") == base_analysis.NODE_SCHEMA
        and result.get("live_screen_correctness_pass") is True
        and result.get("blocks_completed") == 8
        and result.get("endpoint_routing_policy") == "semantic_epoch_v1"
        and result.get("passive_external_endpoint_credit") is True
        and result.get("performance_claim_allowed") is False
        and result.get("unchanged_pd_data_plane") is True
        and result.get("transport") == "LMCacheConnectorV1:UCX",
        "semantic node result contract differs",
    )
    contract_path = Path(str(result.get("run_contract"))).resolve()
    _require(
        contract_path.is_file()
        and _sha256(contract_path) == result.get("run_contract_sha256"),
        "semantic run-contract binding differs",
    )
    contract = _load(contract_path, name="semantic run contract")
    _require(
        contract.get("schema") == contract_builder.SCHEMA
        and contract.get("fingerprint_sha256")
        == contract_builder.contract_fingerprint(contract)
        and contract.get("calibration_only") is True
        and contract.get("controller_parameter_search_allowed") is False
        and contract.get("frozen_validation_allowed") is False
        and contract.get("controller_reset_before_each_measured_block")
        is True,
        "semantic run contract differs",
    )
    implementation_count = _validate_implementation(contract)
    phase_client._validate_semantic_runtime(contract)
    endpoint_entry = contract.get("endpoint_service_profile")
    source_endpoint_entry = contract.get("source_endpoint_service_profile")
    _require(
        isinstance(endpoint_entry, Mapping)
        and set(endpoint_entry) == {
            "path", "sha256", "schema", "fingerprint_sha256",
            "derived_from_sha256",
        }
        and isinstance(source_endpoint_entry, Mapping)
        and set(source_endpoint_entry) == {
            "path", "sha256", "schema", "fingerprint_sha256",
        },
        "semantic endpoint profile bindings differ",
    )
    endpoint_path = _implementation_path(
        endpoint_entry["path"], name="semantic endpoint profile")
    source_endpoint_path = _implementation_path(
        source_endpoint_entry["path"], name="semantic source endpoint profile")
    _require(
        _sha256(endpoint_path) == endpoint_entry["sha256"]
        and _sha256(source_endpoint_path) == source_endpoint_entry["sha256"]
        and endpoint_entry["derived_from_sha256"]
        == source_endpoint_entry["sha256"],
        "semantic endpoint profile lineage differs",
    )
    endpoint_profile = load_endpoint_service_profile(endpoint_path)
    source_endpoint_profile = load_endpoint_service_profile(
        source_endpoint_path)
    _require(
        endpoint_profile.schema == SCHEMA_V2
        and endpoint_profile.fingerprint_sha256
        == endpoint_entry["fingerprint_sha256"]
        and endpoint_profile.routing_policy is not None
        and endpoint_profile.routing_policy.as_dict()
        == contract.get("semantic_credit_contract")
        and source_endpoint_profile.fingerprint_sha256
        == source_endpoint_entry["fingerprint_sha256"],
        "semantic endpoint routing policy differs from the frozen profile",
    )

    _require(
        _sha256(base_analysis_path) == _canonical_sha(
            expected_base_analysis_sha256, name="base analysis SHA"),
        "base phase-screen analysis digest differs",
    )
    phase_analysis = _load(base_analysis_path, name="base phase analysis")
    _require(
        phase_analysis.get("schema") == base_analysis.SCHEMA
        and Path(str(phase_analysis.get("source_result"))).resolve()
        == result_path
        and phase_analysis.get("source_result_sha256")
        == expected_result_sha256
        and phase_analysis.get("live_screen_correctness_pass") is True
        and phase_analysis.get("performance_claim_allowed") is False,
        "base phase-screen analysis lineage differs",
    )

    result_root = result_path.parent.resolve()
    raw_path = _bound_result_file(
        result.get("raw"), root=result_root, name="semantic client raw")
    _require(_sha256(raw_path) == result.get("raw_sha256"),
             "semantic client raw digest differs")
    raw = _load(raw_path, name="semantic client raw")
    _require(
        raw.get("schema") == base_analysis.CLIENT_SCHEMA
        and raw.get("run_contract_sha256") == result["run_contract_sha256"]
        and raw.get("endpoint_routing_policy") == "semantic_epoch_v1"
        and raw.get("passive_external_endpoint_credit") is True
        and raw.get("controller_reset_before_each_block_exact") is True
        and raw.get("live_screen_correctness_pass") is True,
        "semantic client contract differs",
    )
    artifacts = raw.get("artifacts")
    contracts = raw.get("contracts")
    _require(
        isinstance(artifacts, Mapping) and len(artifacts) == 8
        and isinstance(contracts, Mapping) and set(contracts) == set(artifacts),
        "semantic block inventory differs",
    )
    base_bindings = {
        row["key"]: row for row in phase_analysis.get(
            "child_artifact_bindings", [])
    }
    _require(set(base_bindings) == set(artifacts),
             "base analysis child inventory differs")

    tempo_routes: Counter[str] = Counter()
    tempo_reasons: Counter[str] = Counter()
    external_routes: Counter[str] = Counter()
    external_proxy_modes: Counter[str] = Counter()
    tempo_generations = []
    semantic_requests = 0
    external_requests = 0
    for block_key in sorted(artifacts):
        block_path = _bound_result_file(
            artifacts[block_key], root=result_root,
            name=f"semantic block {block_key}")
        binding = base_bindings[block_key]
        _require(
            Path(binding["path"]).resolve() == block_path
            and binding["sha256"] == _sha256(block_path),
            f"semantic block/base analysis binding differs: {block_key}",
        )
        block = _load(block_path, name=f"semantic block {block_key}")
        block_contract = block.get("c4_phase_screen_contract")
        _require(
            isinstance(block_contract, Mapping)
            and block_contract == contracts[block_key]
            and block_contract.get("endpoint_routing_policy")
            == "semantic_epoch_v1"
            and block_contract.get("passive_external_endpoint_credit") is True,
            f"semantic block contract differs: {block_key}",
        )
        _require(
            block_contract.get("controller_reset_before_block_exact") is True,
            f"semantic block lacks an isolated controller epoch: {block_key}",
        )
        endpoint_series = block.get("c4_endpoint_series")
        _require(isinstance(endpoint_series, Mapping),
                 f"semantic endpoint series differs: {block_key}")
        reset_generations = phase_client._validate_controller_reset_evidence(
            endpoint_series.get("endpoint_controller_reset_before_block"))
        _require(len(reset_generations) == 2,
                 f"semantic pair reset inventory differs: {block_key}")
        request_index = block_contract.get("request_index")
        decisions = block.get("router_decisions")
        _require(isinstance(request_index, Mapping)
                 and isinstance(decisions, list),
                 f"semantic block rows differ: {block_key}")
        decision_index = {row.get("request_id"): row for row in decisions}
        _require(len(decision_index) == len(decisions)
                 and set(decision_index) == set(request_index),
                 f"semantic block decision IDs differ: {block_key}")
        foreground_arm = block_contract.get("arm")
        for request_id, metadata in request_index.items():
            _require(isinstance(metadata, Mapping),
                     "semantic request metadata differs")
            decision = decision_index[request_id]
            _require(
                decision.get("frontend_semantic_load_schema") == LOAD_SCHEMA
                and decision.get("frontend_semantic_load_source") == LOAD_SOURCE,
                f"semantic frontend ledger is missing: {request_id}",
            )
            tenant = metadata.get("tenant")
            if tenant == "foreground" and foreground_arm == "tempo":
                route, reason, generation = _validate_semantic_decision(
                    decision, contract=contract)
                tempo_routes[route] += 1
                tempo_reasons[reason] += 1
                tempo_generations.append(generation)
                semantic_requests += 1
            elif tenant != "foreground":
                external_route, proxy_mode = _validate_external_decision(
                    decision, metadata=metadata, request_id=str(request_id))
                external_routes[external_route] += 1
                external_proxy_modes[proxy_mode] += 1
                external_requests += 1

    credit_epoch = contract.get("semantic_credit_contract", {}).get(
        "local_external_credit_opens_epoch") is True
    open_reason = (
        "semantic_credit_epoch_open_remote_local_credit"
        if credit_epoch else "semantic_epoch_open_remote_high_water")
    close_reasons = (
        {
            "semantic_credit_epoch_close_remote_unavailable",
            "semantic_credit_epoch_close_local_credit_idle",
        }
        if credit_epoch else {
            "semantic_epoch_close_remote_unavailable",
            "semantic_epoch_close_decoder_low_water",
        }
    )
    exercised = {
        "both_tempo_routes": set(tempo_routes) == {LOCAL_ROUTE, REMOTE_ROUTE},
        "epoch_generation_advanced": bool(tempo_generations)
        and max(tempo_generations) > 0,
        "remote_epoch_opened": any(
            reason == open_reason
            for reason in tempo_reasons),
        "remote_epoch_closed": any(
            reason in close_reasons for reason in tempo_reasons),
        "both_external_routes_observed": (
            external_routes[LOCAL_ROUTE] > 0
            and external_routes[REMOTE_ROUTE] > 0),
    }
    semantic_correctness = all(exercised.values())
    performance_gate = (
        phase_analysis.get("controller_performance_gate_pass") is True
        and phase_analysis.get("original_goal_gate_on_live_tempo", {}).get(
            "all_pass") is True
    )
    promotion_allowed = semantic_correctness and performance_gate
    return {
        "schema": SCHEMA,
        "source_result": str(result_path),
        "source_result_sha256": expected_result_sha256,
        "source_client_raw": str(raw_path),
        "source_client_raw_sha256": _sha256(raw_path),
        "run_contract": str(contract_path),
        "run_contract_sha256": _sha256(contract_path),
        "run_contract_fingerprint_sha256": contract["fingerprint_sha256"],
        "endpoint_service_profile": str(endpoint_path),
        "endpoint_service_profile_sha256": _sha256(endpoint_path),
        "endpoint_service_profile_fingerprint_sha256": (
            endpoint_profile.fingerprint_sha256),
        "base_phase_analysis": str(base_analysis_path),
        "base_phase_analysis_sha256": expected_base_analysis_sha256,
        "implementation_file_count": implementation_count,
        "semantic_tempo_requests": semantic_requests,
        "external_route_pinned_requests": external_requests,
        "tempo_route_counts": dict(sorted(tempo_routes.items())),
        "tempo_reason_counts": dict(sorted(tempo_reasons.items())),
        "external_route_counts": dict(sorted(external_routes.items())),
        "external_service_proxy_mode_counts": dict(
            sorted(external_proxy_modes.items())),
        "maximum_epoch_generation": max(tempo_generations),
        "semantic_exercise_gates": exercised,
        "semantic_correctness_and_exercise_pass": semantic_correctness,
        "original_screen_performance_gate_pass": performance_gate,
        "authorizes_candidate_for_final_c4_integration": promotion_allowed,
        "calibration_only": True,
        "post_screen_parameter_tuning_allowed": False,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "unchanged_pd_data_plane": True,
        "transport": "LMCacheConnectorV1:UCX",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--base-analysis", type=Path, required=True)
    parser.add_argument("--expected-base-analysis-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite semantic analysis")
    value = analyze(
        result_path=args.result,
        expected_result_sha256=args.expected_result_sha256,
        base_analysis_path=args.base_analysis,
        expected_base_analysis_sha256=args.expected_base_analysis_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "sha256": _sha256(args.output.resolve()),
        "authorizes_candidate_for_final_c4_integration": value[
            "authorizes_candidate_for_final_c4_integration"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
