#!/usr/bin/env python3
"""Analyze strongest single fixed edge versus full C6 over one dynamic trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from eval.sota_4node import run_tempo_go_c6_performance_client as campaign


SCHEMA = "tempo-go-c6-performance-analysis-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-p0d1-result", type=Path, required=True)
    parser.add_argument("--fixed-p1d0-result", type=Path, required=True)
    parser.add_argument("--full-result", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_node(
    path: Path, *, arm: str, fixed_policy: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    node = json.loads(path.read_text(encoding="utf-8"))
    _require(node.get("schema") == "tempo-go-c6-performance-node-result-v1",
             f"{arm} node result schema differs")
    raw_path = Path(node["raw"]).resolve()
    _require(raw_path.is_file() and _sha256(raw_path) == node["raw_sha256"],
             f"{arm} raw bundle identity differs")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    _require(raw.get("schema") == campaign.SCHEMA and raw.get("arm") == arm,
             f"{arm} raw bundle differs")
    _require(raw.get("fixed_policy") == fixed_policy,
             f"{arm} fixed-policy binding differs")
    _require(raw.get("performance_claim_allowed") is True,
             f"{arm} bundle is not performance eligible")
    analysis = raw.get("analysis")
    _require(isinstance(analysis, dict)
             and analysis.get("schema") == campaign.ANALYSIS_SCHEMA,
             f"{arm} arm analysis is missing")
    return node, raw


def _blocks(raw: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = raw["analysis"]["blocks"]
    result = {}
    for row in rows:
        key = (row["policy"], row["logical_phase"])
        _require(key not in result, f"duplicate C6 block: {key}")
        result[key] = row
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    offered = sum(int(row["offered_victims"]) for row in rows)
    completed = sum(int(row["completed_victims"]) for row in rows)
    good = sum(int(row["slo_good_victims"]) for row in rows)
    rejects = sum(int(row["global_rejects"]) for row in rows)
    failures = sum(int(row["failures"]) for row in rows)
    p99_values = [float(row["e2e_ms"]["p99"]) for row in rows
                  if row["e2e_ms"]["p99"] is not None]
    p50_values = [float(row["e2e_ms"]["p50"]) for row in rows
                  if row["e2e_ms"]["p50"] is not None]
    return {
        "offered_victims": offered,
        "completed_victims": completed,
        "slo_good_victims": good,
        "slo_attainment_fraction_of_offered": good / offered,
        "global_rejects": rejects,
        "failures": failures,
        "worst_block_e2e_p99_ms": max(p99_values) if p99_values else None,
        "mean_block_e2e_p50_ms": (
            sum(p50_values) / len(p50_values) if p50_values else None),
    }


def analyze(
    fixed_p0d1_result: Path, fixed_p1d0_result: Path,
    full_result: Path, contract_path: Path,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == campaign.CONTRACT_SCHEMA,
             "C6 performance contract differs")
    section = contract["c6_performance"]
    thresholds = section["thresholds"]
    fixed_nodes = {}
    fixed_raws = {}
    for policy, path in (
        ("fixed_p0d1", fixed_p0d1_result),
        ("fixed_p1d0", fixed_p1d0_result),
    ):
        node, raw = _load_node(
            path, arm=campaign.FIXED_ARM, fixed_policy=policy)
        fixed_nodes[policy] = node
        fixed_raws[policy] = raw
    full_node, full_raw = _load_node(
        full_result, arm=campaign.FULL_ARM, fixed_policy=None)
    expected_contract_sha = _sha256(contract_path)
    bound_results = [
        (policy, fixed_nodes[policy], fixed_raws[policy])
        for policy in ("fixed_p0d1", "fixed_p1d0")
    ] + [("full", full_node, full_raw)]
    for name, node, raw in bound_results:
        _require(node["qualification_contract_sha256"] == expected_contract_sha
                 and raw["qualification_contract_sha256"] == expected_contract_sha,
                 f"{name} contract binding differs")

    fixed_blocks = {}
    for policy, raw in fixed_raws.items():
        policy_blocks = _blocks(raw)
        _require(all(key[0] == policy for key in policy_blocks),
                 f"{policy} epoch contains another fixed policy")
        fixed_blocks.update(policy_blocks)
    full_blocks = _blocks(full_raw)
    phases = ("normal", "hot_d0", "hot_d1")
    fixed_policies = ("fixed_p0d1", "fixed_p1d0")
    _require(set(fixed_blocks) == {
        (policy, phase) for policy in fixed_policies for phase in phases
    }, "fixed block matrix is incomplete")
    _require(set(full_blocks) == {(campaign.FULL_ARM, phase) for phase in phases},
             "full C6 block matrix is incomplete")

    # The winner is one deployable static edge over the entire changing
    # workload, never an oracle that picks a different edge after seeing the
    # phase label.
    fixed_summaries = {}
    for policy in fixed_policies:
        overload = _aggregate([
            fixed_blocks[(policy, "hot_d0")],
            fixed_blocks[(policy, "hot_d1")],
        ])
        all_phases = _aggregate([
            fixed_blocks[(policy, phase)] for phase in phases])
        fixed_summaries[policy] = {
            "overload": overload,
            "all_phases": all_phases,
        }
    strongest_fixed = max(
        fixed_policies,
        key=lambda policy: (
            fixed_summaries[policy]["overload"]["slo_good_victims"],
            -fixed_summaries[policy]["overload"][
                "worst_block_e2e_p99_ms"],
        ),
    )
    fixed_overload = fixed_summaries[strongest_fixed]["overload"]
    fixed_all = fixed_summaries[strongest_fixed]["all_phases"]
    full_overload = _aggregate([
        full_blocks[(campaign.FULL_ARM, "hot_d0")],
        full_blocks[(campaign.FULL_ARM, "hot_d1")],
    ])
    full_all = _aggregate([
        full_blocks[(campaign.FULL_ARM, phase)] for phase in phases])

    _require(fixed_overload["slo_good_victims"] > 0,
             "strongest fixed overload SLO-goodput is zero")
    _require(fixed_all["slo_good_victims"] > 0,
             "strongest fixed total SLO-goodput is zero")
    overload_goodput_ratio = (
        full_overload["slo_good_victims"]
        / fixed_overload["slo_good_victims"]
    )
    all_phase_goodput_ratio = (
        full_all["slo_good_victims"] / fixed_all["slo_good_victims"]
    )
    fixed_p99 = fixed_overload["worst_block_e2e_p99_ms"]
    full_p99 = full_overload["worst_block_e2e_p99_ms"]
    _require(fixed_p99 is not None and full_p99 is not None and fixed_p99 > 0,
             "overload p99 is unavailable")
    overload_p99_reduction = 1.0 - full_p99 / fixed_p99

    fixed_normal = fixed_blocks[(strongest_fixed, "normal")]
    full_normal = full_blocks[(campaign.FULL_ARM, "normal")]
    fixed_normal_p50 = fixed_normal["e2e_ms"]["p50"]
    full_normal_p50 = full_normal["e2e_ms"]["p50"]
    _require(fixed_normal_p50 is not None and full_normal_p50 is not None,
             "normal p50 is unavailable")
    normal_regression = full_normal_p50 / fixed_normal_p50 - 1.0
    fixed_bad = fixed_overload["global_rejects"] + fixed_overload["failures"]
    full_bad = full_overload["global_rejects"] + full_overload["failures"]

    # Verify prompt/timing/geometry identities independently of policy names.
    same_population = {}
    for phase in phases:
        hashes = {
            fixed_raws[policy]["contracts"][
                fixed_blocks[(policy, phase)]["name"]
            ][
                "semantic_schedule_sha256"]
            for policy in fixed_policies
        }
        hashes.add(
            full_raw["contracts"][
                full_blocks[(campaign.FULL_ARM, phase)]["name"]
            ]["semantic_schedule_sha256"]
        )
        same_population[phase] = len(hashes) == 1

    gates = {
        "same_population_all_phases": all(same_population.values()),
        "overload_slo_goodput_at_least_1_5x": overload_goodput_ratio
        >= float(thresholds["overload_slo_goodput_ratio"]),
        "overload_p99_reduction_at_least_30pct": overload_p99_reduction
        >= float(thresholds["overload_p99_reduction_fraction"]),
        "normal_e2e_regression_at_most_3pct": normal_regression
        <= float(thresholds["normal_e2e_regression_fraction"]),
        "reject_or_fail_does_not_increase": full_bad - fixed_bad
        <= int(thresholds["reject_or_fail_increase_allowed"]),
        "all_phase_incremental_goodput_at_least_1_1x": all_phase_goodput_ratio
        >= float(thresholds["full_incremental_slo_goodput_ratio"]),
    }
    return {
        "schema": SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": expected_contract_sha,
        "fixed_results": {
            "fixed_p0d1": str(fixed_p0d1_result.resolve()),
            "fixed_p1d0": str(fixed_p1d0_result.resolve()),
        },
        "full_result": str(full_result.resolve()),
        "strongest_single_fixed_policy": strongest_fixed,
        "fixed_policy_summaries": fixed_summaries,
        "full_c6": {"overload": full_overload, "all_phases": full_all},
        "effects": {
            "overload_slo_goodput_ratio": overload_goodput_ratio,
            "overload_p99_reduction_fraction": overload_p99_reduction,
            "all_phase_slo_goodput_ratio": all_phase_goodput_ratio,
            "normal_e2e_p50_regression_fraction": normal_regression,
            "overload_reject_or_fail_delta": full_bad - fixed_bad,
        },
        "same_population_by_phase": same_population,
        "full_c6_route_and_edge_counts": {
            phase: {
                "routes": full_blocks[(campaign.FULL_ARM, phase)]["route_counts"],
                "edges": full_blocks[(campaign.FULL_ARM, phase)]["edge_counts"],
            }
            for phase in phases
        },
        "gates": gates,
        "c6_performance_gate_pass": all(gates.values()),
        "independent_validation_claim_allowed": False,
    }


def main() -> int:
    args = _parse()
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    value = analyze(
        args.fixed_p0d1_result,
        args.fixed_p1d0_result,
        args.full_result,
        args.contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "strongest_single_fixed_policy": value["strongest_single_fixed_policy"],
        "effects": value["effects"],
        "c6_performance_gate_pass": value["c6_performance_gate_pass"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
