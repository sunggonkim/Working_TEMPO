#!/usr/bin/env python3
"""Analyze one preregistered, fresh-allocation C8 validation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from eval.sota_4node import analyze_tempo_go_c8_dual_regime as frozen
from eval.sota_4node import run_tempo_go_c8_independent_validation_client as client


SCHEMA = "tempo-go-c8-independent-validation-analysis-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    _require(0.0 <= percentile <= 1.0, "invalid percentile")
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _jain(values: Iterable[float]) -> float:
    usable = [float(value) for value in values]
    _require(bool(usable), "Jain population is empty")
    total = sum(usable)
    square = sum(value * value for value in usable)
    return 1.0 if square == 0.0 else total * total / (len(usable) * square)


def _background_report(
    full: dict[str, object], section: dict[str, object],
    gates: dict[str, object],
) -> dict[str, object]:
    blocks = {str(row["name"]): row for row in full["blocks"]}
    specs = {str(row["name"]): row for row in section["blocks"]}
    remote_name = str(section["independent_remote_block_name"])
    c7_names = [
        name for name, spec in specs.items()
        if name != remote_name and spec.get("hot_decoder_index") in (0, 1)
    ]
    _require(c7_names and remote_name in blocks,
             "independent background block population differs")

    aggregate: dict[str, dict[str, int]] = {}
    per_block = []
    service_lane_failures = 0
    c7_complete = 0
    c7_offered = 0
    minimum_block_tenant_fraction = 1.0
    for name in c7_names:
        counts_by_tenant = blocks[name]["terminal_counts_by_business_tenant"]
        for tenant, statuses in counts_by_tenant.items():
            if not str(tenant).startswith("background"):
                continue
            complete = int(statuses.get("complete", 0))
            offered = sum(int(value) for value in statuses.values())
            _require(offered > 0, "C7 background terminal population is empty")
            fraction = complete / offered
            minimum_block_tenant_fraction = min(
                minimum_block_tenant_fraction, fraction)
            service_lane_failures += int(statuses.get(
                "service_lane_failure", 0))
            c7_complete += complete
            c7_offered += offered
            target = aggregate.setdefault(str(tenant), {
                "complete": 0, "offered": 0,
            })
            target["complete"] += complete
            target["offered"] += offered
            per_block.append({
                "block": name,
                "tenant": tenant,
                "complete": complete,
                "offered": offered,
                "completion_fraction": fraction,
                "terminal_counts": statuses,
            })
    _require(c7_offered > 0 and len(aggregate) >= 2,
             "C7 tenant fairness population is incomplete")
    tenant_fractions = {
        tenant: values["complete"] / values["offered"]
        for tenant, values in aggregate.items()
    }
    c7_completion_fraction = c7_complete / c7_offered
    service_lane_failure_fraction = service_lane_failures / c7_offered
    jain = _jain(tenant_fractions.values())

    remote_counts = blocks[remote_name][
        "terminal_counts_by_business_tenant"]
    c8_background = remote_counts.get("background_local_decoder", {})
    c8_complete = int(c8_background.get("complete", 0))
    c8_offered = sum(int(value) for value in c8_background.values())
    _require(c8_offered > 0, "C8 background terminal population is empty")
    c8_completion_fraction = c8_complete / c8_offered
    c8_noncomplete = c8_offered - c8_complete

    c7_gate = (
        c7_completion_fraction
        >= float(gates["minimum_c7_background_completion_fraction"])
        and minimum_block_tenant_fraction
        >= float(gates["minimum_c7_block_tenant_completion_fraction"])
        and jain >= float(gates["minimum_c7_tenant_jain_fairness"])
        and service_lane_failure_fraction
        <= float(gates["maximum_c7_service_lane_failure_fraction"])
    )
    c8_gate = (
        c8_completion_fraction
        >= float(gates["minimum_c8_background_completion_fraction"])
        and c8_noncomplete <= int(gates["maximum_c8_background_noncomplete"])
    )
    return {
        "c7_blocks": per_block,
        "c7_tenants": {
            tenant: {**values, "completion_fraction": tenant_fractions[tenant]}
            for tenant, values in sorted(aggregate.items())
        },
        "c7_complete": c7_complete,
        "c7_offered": c7_offered,
        "c7_completion_fraction": c7_completion_fraction,
        "c7_minimum_block_tenant_completion_fraction": (
            minimum_block_tenant_fraction),
        "c7_tenant_jain_fairness": jain,
        "c7_service_lane_failures": service_lane_failures,
        "c7_service_lane_failure_fraction": service_lane_failure_fraction,
        "c7_background_gate": c7_gate,
        "c8_complete": c8_complete,
        "c8_offered": c8_offered,
        "c8_completion_fraction": c8_completion_fraction,
        "c8_noncomplete": c8_noncomplete,
        "c8_background_gate": c8_gate,
        "background_utility_and_fairness_gate": c7_gate and c8_gate,
    }


def _signal_status(
    decision: dict[str, object], signal_name: str,
) -> tuple[bool, bool]:
    payload = decision.get("frontend_tempo_go_decision")
    _require(isinstance(payload, dict), "global decision payload is missing")
    provenance = payload.get("telemetry_provenance")
    _require(isinstance(provenance, dict) and provenance,
             "global telemetry provenance is missing")
    found = []
    for endpoint in provenance.values():
        if not isinstance(endpoint, dict):
            continue
        cross_layer = endpoint.get("cross_layer")
        if not isinstance(cross_layer, dict):
            continue
        for signal in cross_layer.get("signals", []):
            if isinstance(signal, dict) and signal.get("name") == signal_name:
                found.append(signal)
    if not found:
        return False, False
    supported = any(
        row.get("support") == "supported"
        and isinstance(row.get("value"), (int, float))
        and math.isfinite(float(row["value"]))
        for row in found
    )
    explicit_unavailable = all(
        row.get("support") == "not_collected" and row.get("value") is None
        for row in found
    )
    return supported, supported or explicit_unavailable


def _telemetry_report(
    result_path: Path, contract: dict[str, object], gates: dict[str, object],
) -> dict[str, object]:
    wrapper = _json(result_path)
    bundle_path = Path(str(wrapper["raw"])).resolve()
    bundle = _json(bundle_path)
    receipt = bundle.get("independent_validation_execution")
    _require(
        isinstance(receipt, dict)
        and receipt.get("schema") == client.EXECUTION_SCHEMA,
        "independent workload execution receipt is missing",
    )
    remote_name = str(contract["independent_validation"][
        "remote_favorable_block"])
    raw_path = Path(str(bundle["artifacts"][remote_name])).resolve()
    raw = _json(raw_path)
    decisions = [
        row for row in raw["router_decisions"]
        if row.get("tempo_go_global_commit_applied") is True
    ]
    expected = int(bundle["analysis"]["remote_favorable"][
        "completed_victims"])
    _require(len(decisions) == expected and expected > 0,
             "independent remote decision population differs")

    preparation_statuses: dict[str, int] = {}
    collection_ms = []
    admission_wait_ms = []
    tie_breaks = 0
    for decision in decisions:
        preparation = decision.get("frontend_tempo_go_telemetry_preparation")
        _require(isinstance(preparation, dict),
                 "telemetry preparation receipt is missing")
        status = str(preparation.get("status"))
        preparation_statuses[status] = preparation_statuses.get(status, 0) + 1
        elapsed = preparation.get("collection_elapsed_ns")
        wait = decision.get("frontend_tempo_go_admission_wait_ns")
        _require(isinstance(elapsed, int) and elapsed >= 0
                 and isinstance(wait, int) and wait >= 0,
                 "telemetry overhead receipt is invalid")
        collection_ms.append(elapsed / 1_000_000.0)
        admission_wait_ms.append(wait / 1_000_000.0)
        payload = decision["frontend_tempo_go_decision"]
        bindings = payload.get("binding_resources", [])
        if "mesh_telemetry_uncertainty_source_virtual_service" in bindings:
            tie_breaks += 1

    required = [str(value) for value in gates["required_supported_signals"]]
    explicit = [
        str(value) for value in gates["required_explicit_status_signals"]
    ]
    support: dict[str, object] = {}
    supported_gate = True
    explicit_gate = True
    for name in list(dict.fromkeys(required + explicit)):
        supported = 0
        classified = 0
        for decision in decisions:
            is_supported, is_classified = _signal_status(decision, name)
            supported += int(is_supported)
            classified += int(is_classified)
        supported_fraction = supported / len(decisions)
        classified_fraction = classified / len(decisions)
        support[name] = {
            "supported_decisions": supported,
            "explicitly_classified_decisions": classified,
            "population": len(decisions),
            "supported_fraction": supported_fraction,
            "explicitly_classified_fraction": classified_fraction,
        }
        if name in required:
            supported_gate = supported_gate and supported_fraction >= float(
                gates["minimum_supported_signal_fraction"])
        if name in explicit:
            explicit_gate = explicit_gate and classified_fraction == 1.0

    batch_fraction = preparation_statuses.get("batch", 0) / len(decisions)
    tie_break_fraction = tie_breaks / len(decisions)
    collection_p99 = _percentile(collection_ms, 0.99)
    admission_p99 = _percentile(admission_wait_ms, 0.99)
    collection_p50 = _percentile(collection_ms, 0.50)
    admission_p50 = _percentile(admission_wait_ms, 0.50)
    assert (
        collection_p50 is not None and collection_p99 is not None
        and admission_p50 is not None and admission_p99 is not None
    )
    overhead_gate = (
        batch_fraction >= float(gates["minimum_complete_batch_fraction"])
        and collection_p50 <= float(gates["maximum_collection_p50_ms"])
        and collection_p99 <= float(gates["maximum_collection_p99_ms"])
        and admission_p50 <= float(gates["maximum_admission_wait_p50_ms"])
        and admission_p99 <= float(gates["maximum_admission_wait_p99_ms"])
    )
    tie_break_gate = tie_break_fraction >= float(
        gates["minimum_source_virtual_service_binding_fraction"])
    telemetry_gate = (
        supported_gate and explicit_gate and overhead_gate and tie_break_gate)
    return {
        "bundle": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "remote_raw": str(raw_path),
        "remote_raw_sha256": _sha256(raw_path),
        "decision_population": len(decisions),
        "preparation_statuses": dict(sorted(preparation_statuses.items())),
        "complete_batch_fraction": batch_fraction,
        "collection_ms": {
            "p50": collection_p50,
            "p99": collection_p99,
            "max": max(collection_ms),
        },
        "admission_wait_ms": {
            "p50": admission_p50,
            "p99": admission_p99,
            "max": max(admission_wait_ms),
        },
        "signal_support": support,
        "required_supported_signal_gate": supported_gate,
        "required_explicit_status_gate": explicit_gate,
        "source_virtual_service_bindings": tie_breaks,
        "source_virtual_service_binding_fraction": tie_break_fraction,
        "source_virtual_service_binding_gate": tie_break_gate,
        "telemetry_overhead_gate": overhead_gate,
        "telemetry_and_overhead_gate": telemetry_gate,
    }


def analyze_campaign(
    results: dict[str, Path], contract_path: Path,
) -> dict[str, object]:
    contract_path = contract_path.resolve()
    contract = _json(contract_path)
    _require(contract.get("schema") == frozen.CONTRACT_SCHEMA,
             "independent C8 contract schema differs")
    heldout = contract.get("independent_validation")
    _require(
        isinstance(heldout, dict)
        and heldout.get("schema") == client.INDEPENDENT_SCHEMA,
        "independent C8 metadata differs",
    )
    expected = [
        str(row["name"]) for row in contract["joint_control"]["arms"]
    ]
    _require(list(results) == expected,
             "independent C8 arm order differs from preregistration")

    parent = heldout["parent_discovery"]
    repo_root = Path(__file__).resolve().parents[2]
    parent_path = (repo_root / str(parent["analysis_path"])).resolve()
    _require(parent_path.is_file() and _sha256(parent_path) == parent["analysis_sha256"],
             "parent discovery analysis drifted")
    parent_analysis = _json(parent_path)
    parent_gate = (
        parent_analysis.get("c8_dual_regime_discovery_positive") is True
        and parent_analysis.get("performance_claim_allowed") is True
        and parent_analysis.get("contract_sha256")
        == parent["contract_sha256"]
    )

    job_ids = set()
    execution_gate = True
    result_receipts = {}
    for arm, path in results.items():
        wrapper = _json(path)
        job_ids.add(str(wrapper.get("slurm_job_id")))
        execution_gate = execution_gate and (
            wrapper.get("qualification_contract_sha256") == _sha256(contract_path)
        )
        bundle_path = Path(str(wrapper["raw"])).resolve()
        bundle = _json(bundle_path)
        receipt = bundle.get("independent_validation_execution")
        execution_gate = execution_gate and (
            isinstance(receipt, dict)
            and receipt.get("schema") == client.EXECUTION_SCHEMA
            and receipt.get("contract_sha256") == _sha256(contract_path)
            and receipt.get("request_seed") == heldout["request_seed"]
            and receipt.get("arrival_jitter") == heldout["arrival_jitter"]
            and receipt.get("p_only_prompt_namespace")
            == heldout["p_only_prompt_namespace"]
        )
        result_receipts[arm] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "raw": str(bundle_path),
            "raw_sha256": _sha256(bundle_path),
        }
    forbidden = {str(value) for value in heldout["forbidden_discovery_job_ids"]}
    fresh_allocation_gate = (
        len(job_ids) == 1
        and next(iter(job_ids)).isdigit()
        and not bool(job_ids & forbidden)
    )

    base = frozen.analyze_campaign(results, contract_path)
    base_performance_gate = (
        base.get("c8_dual_regime_discovery_positive") is True
        and base.get("performance_claim_allowed") is True
    )
    full = base["arms"][base["headline_full_arm"]]
    independent_gates = heldout["gates"]
    background = _background_report(
        full,
        {
            **contract["joint_control"],
            "independent_remote_block_name": heldout["remote_favorable_block"],
        },
        independent_gates["background"],
    )
    telemetry = _telemetry_report(
        results[str(base["headline_full_arm"])],
        contract,
        independent_gates["telemetry"],
    )
    positive = (
        parent_gate
        and fresh_allocation_gate
        and execution_gate
        and base_performance_gate
        and background["background_utility_and_fairness_gate"] is True
        and telemetry["telemetry_and_overhead_gate"] is True
    )
    return {
        "schema": SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "parent_discovery": {
            "path": str(parent_path),
            "sha256": _sha256(parent_path),
            "gate": parent_gate,
        },
        "slurm_job_ids": sorted(job_ids),
        "fresh_allocation_gate": fresh_allocation_gate,
        "one_shot_execution_receipt_gate": execution_gate,
        "results": result_receipts,
        "base_campaign": base,
        "base_performance_gate": base_performance_gate,
        "background": background,
        "telemetry": telemetry,
        "independent_validation_positive": positive,
        "performance_claim_allowed": positive,
        "independent_validation_claim_allowed": positive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    results: dict[str, Path] = {}
    for value in args.result:
        name, separator, raw_path = value.partition("=")
        _require(bool(separator) and name and raw_path,
                 "result must be ARM=PATH")
        _require(name not in results, "duplicate result arm")
        results[name] = Path(raw_path).resolve()
    analysis = analyze_campaign(results, args.contract)
    args.output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
