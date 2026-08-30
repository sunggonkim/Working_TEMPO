#!/usr/bin/env python3
"""Analyze Candidate B on newly calibrated C4 rows without making a claim."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_adaptive_screen as base
from eval.sota_4node import analyze_tempo_pd_c4_semantic_epoch_screen as semantic
from eval.sota_4node import build_tempo_pd_c4_semantic_integration_run_contract as contract_builder
from eval.sota_4node import run_tempo_pd_c4_adaptive_screen_client as client
from eval.sota_4node import vllm_lmcache_pd_c4_adaptive_screen_node as node
from tempo.pd_contention_workload import ForegroundArm, Tenant


SCHEMA = "tempo-pd-c4-semantic-integration-screen-analysis-v1"
NODE_SCHEMA = node.SEMANTIC_SCHEMA
REPO_ROOT = Path(__file__).resolve().parents[2]

_ADAPTIVE_IMPLEMENTATION_KEYS = frozenset({
    "adaptive_implementation_contract",
    "adaptive_implementation_contract_sha256",
    "adaptive_implementation_fingerprint_sha256",
    "adaptive_implementation_file_count",
})
_SEMANTIC_IMPLEMENTATION_KEYS = frozenset({
    "semantic_integration_implementation_contract",
    "semantic_integration_implementation_contract_sha256",
    "semantic_integration_implementation_fingerprint_sha256",
    "semantic_integration_implementation_file_count",
})
_SEMANTIC_RUNTIME_KEYS = frozenset({
    "endpoint_routing_policy",
    "endpoint_service_profile_fingerprint_sha256",
    "semantic_credit_contract",
    "passive_external_endpoint_credit",
    "semantic_policy_authorized",
})
_NODE_KEYS = (
    (base._NODE_KEYS - _ADAPTIVE_IMPLEMENTATION_KEYS)
    | _SEMANTIC_IMPLEMENTATION_KEYS
    | _SEMANTIC_RUNTIME_KEYS
)
_CLIENT_KEYS = base._CLIENT_KEYS | _SEMANTIC_RUNTIME_KEYS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analysis_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_run_contract(path: Path) -> dict[str, object]:
    value = base._load_object(path, name="semantic integration run contract")
    _require(
        value.get("schema") == contract_builder.SCHEMA
        and value.get("fingerprint_sha256")
        == contract_builder.contract_fingerprint(value)
        and value.get("semantic_policy_authorized") is True
        and value.get("same_allocation_calibration_required") is True
        and value.get("controller_parameter_search_allowed") is False
        and value.get("calibration_only") is True
        and value.get("performance_claim_allowed") is False,
        "semantic integration run contract is invalid",
    )
    arguments = {}
    for argument, entry_name in (
        ("adaptive_contract", "base_adaptive_run_contract"),
        ("semantic_analysis", "semantic_exploratory_analysis"),
        ("semantic_endpoint", "endpoint_service_profile"),
        ("implementation", "semantic_integration_implementation_contract"),
    ):
        artifact, entry = base._contract_entry(value, entry_name)
        arguments[f"{argument}_path"] = artifact
        arguments[f"{argument}_sha256"] = entry["sha256"]
    rebuilt = contract_builder.build_run_contract(
        **arguments, repo_root=REPO_ROOT)
    _require(rebuilt == value,
             "semantic integration run contract does not reproduce")
    return value


def _semantic_exercise(
    raw_blocks: list[tuple[Mapping[str, object], ForegroundArm]],
    *, contract: Mapping[str, object],
) -> dict[str, object]:
    tempo_routes: Counter[str] = Counter()
    tempo_reasons: Counter[str] = Counter()
    external_routes: Counter[str] = Counter()
    external_modes: Counter[str] = Counter()
    generations: list[int] = []
    tempo_requests = 0
    external_requests = 0
    for raw, arm in raw_blocks:
        block_contract = raw["c4_adaptive_screen_contract"]
        request_index = block_contract["request_index"]
        decisions = {
            row["request_id"]: row for row in raw["router_decisions"]}
        for request_id, metadata in request_index.items():
            decision = decisions[request_id]
            _require(
                decision.get("frontend_semantic_load_schema")
                == semantic.LOAD_SCHEMA
                and decision.get("frontend_semantic_load_source")
                == semantic.LOAD_SOURCE,
                f"semantic pair-local ledger is missing: {request_id}",
            )
            tenant = metadata["tenant"]
            if tenant == Tenant.FOREGROUND.value and arm is ForegroundArm.TEMPO:
                route, reason, generation = semantic._validate_semantic_decision(
                    decision, contract=contract)
                tempo_routes[route] += 1
                tempo_reasons[reason] += 1
                generations.append(generation)
                tempo_requests += 1
            elif tenant != Tenant.FOREGROUND.value:
                route, mode = semantic._validate_external_decision(
                    decision, metadata=metadata, request_id=request_id)
                external_routes[route] += 1
                external_modes[mode] += 1
                external_requests += 1
    gates = {
        "both_tempo_routes": set(tempo_routes) == {
            semantic.LOCAL_ROUTE, semantic.REMOTE_ROUTE},
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
            external_routes[semantic.LOCAL_ROUTE] > 0
            and external_routes[semantic.REMOTE_ROUTE] > 0),
    }
    return {
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


def analyze(
    result_path: Path, *, expected_result_sha256: str,
) -> dict[str, object]:
    result_path = result_path.resolve()
    expected_result_sha256 = base._canonical_sha(
        expected_result_sha256,
        name="semantic integration node result SHA-256",
    )
    _require(
        result_path.is_file() and _sha256(result_path) == expected_result_sha256,
        "semantic integration node result digest differs",
    )
    result = base._load_object(
        result_path, name="semantic integration node result")
    _require(set(result) == _NODE_KEYS and result.get("schema") == NODE_SCHEMA,
             "semantic integration node result inventory differs")
    result_root = result_path.parent
    raw_path = base._bound_path(
        result["raw"], result["raw_sha256"],
        name="semantic integration client raw", within=result_root)
    run_contract_path = base._bound_path(
        result["run_contract"], result["run_contract_sha256"],
        name="semantic integration run contract")
    run_contract = _validate_run_contract(run_contract_path)
    _require(
        result["run_contract_fingerprint_sha256"]
        == run_contract["fingerprint_sha256"],
        "semantic integration node/run-contract fingerprint differs",
    )
    manifest_path, manifest_entry = base._contract_entry(
        run_contract, "phase_manifest")
    manifest = base._load_object(
        manifest_path, name="semantic integration manifest")
    endpoint_path, endpoint_entry = base._contract_entry(
        run_contract, "endpoint_service_profile")
    del endpoint_path
    source_result_path, source_result_entry = base._contract_entry(
        run_contract, "source_node_result")
    source_result = base._load_object(source_result_path, name="source C4 result")
    _require(
        result.get("slurm_job_id") == source_result.get("slurm_job_id")
        and type(result.get("slurm_job_id")) is str
        and bool(result["slurm_job_id"].strip()),
        "C4 and semantic integration did not reuse one persistent allocation",
    )
    _require(
        result.get("source_workload")
        == run_contract["source_workload"]["path"]
        and result.get("source_workload_sha256")
        == run_contract["source_workload"]["sha256"]
        and result.get("phase_manifest") == str(manifest_path)
        and result.get("phase_manifest_sha256") == manifest_entry["sha256"]
        and result.get("elastic_profile")
        == run_contract["elastic_profile"]["path"]
        and result.get("elastic_profile_sha256")
        == run_contract["elastic_profile"]["sha256"]
        and result.get("endpoint_service_profile")
        == run_contract["endpoint_service_profile"]["path"]
        and result.get("endpoint_service_profile_sha256")
        == run_contract["endpoint_service_profile"]["sha256"]
        and result.get("endpoint_service_profile_fingerprint_sha256")
        == endpoint_entry["fingerprint_sha256"]
        and result.get("endpoint_routing_policy") == "semantic_epoch_v1"
        and result.get("semantic_credit_contract")
        == run_contract["semantic_credit_contract"]
        and result.get("passive_external_endpoint_credit") is True
        and result.get("semantic_policy_authorized") is True
        and result.get("fixed_runtime_environment")
        == run_contract["fixed_runtime_environment"]
        and result.get("block_count") == 8
        and result.get("correctness_gate_pass") is True
        and result.get("calibration_only") is True
        and result.get("performance_claim_allowed") is False
        and result.get("physical_switch_bottleneck_claim_allowed") is False
        and result.get("independent_validation_required") is True
        and result.get("unchanged_pd_data_plane") is True
        and result.get("transport") == "LMCacheConnectorV1:UCX"
        and isinstance(result.get("transport_environment"), dict)
        and 600.0 <= float(result["startup_readiness_timeout_s"]) <= 3600.0,
        "semantic integration node lineage or invariant differs",
    )
    implementation_path, implementation_entry = base._contract_entry(
        run_contract, "semantic_integration_implementation_contract")
    _require(
        result.get("semantic_integration_implementation_contract")
        == str(implementation_path)
        and result.get("semantic_integration_implementation_contract_sha256")
        == implementation_entry["sha256"]
        and result.get(
            "semantic_integration_implementation_fingerprint_sha256")
        == implementation_entry["fingerprint_sha256"],
        "semantic integration node implementation binding differs",
    )

    parent = base._load_object(
        raw_path, name="semantic integration client raw")
    _require(
        set(parent) == _CLIENT_KEYS
        and parent.get("schema") == client.SEMANTIC_SCHEMA,
        "semantic integration client raw inventory differs",
    )
    _require(
        Path(str(parent["run_contract"])).resolve() == run_contract_path
        and parent["run_contract_sha256"] == result["run_contract_sha256"]
        and Path(str(parent["manifest"])).resolve() == manifest_path
        and parent["manifest_sha256"] == manifest_entry["sha256"]
        and parent.get("endpoint_routing_policy") == "semantic_epoch_v1"
        and parent.get("semantic_credit_contract")
        == run_contract["semantic_credit_contract"]
        and parent.get("passive_external_endpoint_credit") is True
        and parent.get("semantic_policy_authorized") is True
        and parent.get("blocks_completed") == 8
        and parent.get("live_screen_correctness_pass") is True
        and parent.get("calibration_only") is True
        and parent.get("performance_claim_allowed") is False,
        "semantic integration client lineage or claim differs",
    )
    base._bound_path(
        parent["cache_plan"], parent["cache_plan_sha256"],
        name="semantic integration cache plan", within=raw_path.parent)
    runtime_path = base._bound_path(
        parent["cache_runtime_evidence"],
        parent["cache_runtime_evidence_sha256"],
        name="semantic integration cache runtime evidence",
        within=raw_path.parent)
    runtime = base._load_object(
        runtime_path, name="semantic integration cache runtime evidence")
    _require(
        runtime.get("schema") == client.c4.RUNTIME_EVIDENCE_SCHEMA
        and runtime.get("preparation_completed_before_measurement") is True
        and runtime.get("measurement_includes_preparation_requests") is False
        and runtime.get("ready_for_measurement") is True,
        "semantic integration cache runtime evidence differs",
    )
    validated = node._validate_client_artifacts(
        parent, client_raw_path=raw_path,
        block_schema=client.SEMANTIC_BLOCK_SCHEMA)
    _require(validated == result["block_artifacts"],
             "semantic integration node/client child bindings differ")

    expected_keys = [item[0] for item in node._EXPECTED_BLOCKS]
    artifacts = parent["artifacts"]
    contracts = parent["contracts"]
    _require(
        list(artifacts) == expected_keys
        and list(contracts) == expected_keys
        and parent["block_order"] == [
            {"arm": arm, "replicate": replicate}
            for _key, arm, replicate in node._EXPECTED_BLOCKS
        ],
        "semantic integration parent block inventory differs",
    )
    blocks = {}
    endpoint_rows = []
    request_rows = []
    generation_history = [[], []]
    block_bindings = []
    raw_blocks: list[tuple[Mapping[str, object], ForegroundArm]] = []
    for sequence, (key, arm_value, replicate) in enumerate(
        node._EXPECTED_BLOCKS
    ):
        arm = ForegroundArm(arm_value)
        entry = artifacts[key]
        path = base._bound_path(
            entry["path"], entry["sha256"],
            name=f"semantic integration block {key}", within=raw_path.parent)
        raw, foreground, block_endpoint_rows, block_request_rows = (
            base._validate_block(
                path,
                parent_contract=contracts[key],
                block_key=key,
                sequence=sequence,
                arm=arm,
                replicate=replicate,
                manifest=manifest,
                endpoint_fingerprint=endpoint_entry["fingerprint_sha256"],
                semantic_contract=run_contract,
            ))
        blocks[(replicate, arm_value)] = foreground
        endpoint_rows.extend(block_endpoint_rows)
        request_rows.extend(block_request_rows)
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
        generation_history == [list(range(1, 9)), list(range(1, 9))],
        "semantic integration controllers were not reset once per block",
    )
    _require(len(endpoint_rows) == 192 and len(request_rows) == 192,
             "semantic integration phase/tenant row inventory differs")
    samples = base._paired_samples(blocks)
    expected_foreground = sum(
        metadata["tenant"] == Tenant.FOREGROUND.value
        for metadata in contracts[expected_keys[0]]["request_index"].values()
    ) * 2
    _require(len(samples) == expected_foreground,
             "semantic integration paired foreground count differs")
    recomputed_paired = client._paired_gate(
        block_paths={key: Path(artifacts[key]["path"]) for key in expected_keys},
        contracts=contracts,
        block_schema=client.SEMANTIC_BLOCK_SCHEMA,
    )
    _require(parent["paired_output_gate"] == recomputed_paired,
             "semantic integration paired gate differs from child evidence")
    metrics, group_rows = base._screen_metrics(samples, manifest)
    exercise = _semantic_exercise(raw_blocks, contract=run_contract)
    _require(
        result.get("tempo_both_routes_exercised")
        == metrics["screen_gates"]["both_tempo_routes_exercised"]
        == exercise["gates"]["both_tempo_routes"]
        == parent.get("live_screen_route_diversity_pass"),
        "semantic integration route-diversity evidence differs",
    )
    authorizes_independent = (
        metrics["authorizes_independent_validation"] is True
        and exercise["all_pass"] is True)
    output: dict[str, object] = {
        "schema": SCHEMA,
        "source_node_result": {
            "path": str(result_path), "sha256": expected_result_sha256},
        "source_c4_node_result": {
            "path": str(source_result_path),
            "sha256": source_result_entry["sha256"],
        },
        "persistent_allocation_job_id": result["slurm_job_id"],
        "run_contract": {
            "path": str(run_contract_path),
            "sha256": result["run_contract_sha256"],
            "fingerprint_sha256": run_contract["fingerprint_sha256"],
        },
        "block_artifacts": block_bindings,
        "controller_generation_history": generation_history,
        "endpoint_phase_rows": endpoint_rows,
        "request_phase_tenant_rows": request_rows,
        "foreground_paired_samples": samples,
        "phase_geometry_rows": group_rows,
        "screen_metrics": metrics,
        "semantic_exercise": exercise,
        "authorizes_independent_validation": authorizes_independent,
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
    _require(not args.output.exists(),
             "refusing to overwrite semantic integration analysis")
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
