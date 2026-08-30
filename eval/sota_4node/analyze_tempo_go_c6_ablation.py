#!/usr/bin/env python3
"""Analyze the source-bound C6 fixed and dynamic ablation receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.sota_4node import analyze_tempo_go_c6_performance as fixed_analysis
from eval.sota_4node import run_tempo_go_c6_performance_client as campaign


SCHEMA = "tempo-go-c6-ablation-analysis-v1"
PHASES = ("normal", "hot_d0", "hot_d1")
INPUTS = {
    "fixed_p1d0": (campaign.FIXED_ARM, "fixed_p1d0"),
    "predictor": (campaign.PREDICTOR_ARM, None),
    "queue_gpu": (campaign.QUEUE_GPU_ARM, None),
    "network_request_only": (campaign.NETWORK_REQUEST_ONLY_ARM, None),
    "app_global_only": (campaign.APP_GLOBAL_ONLY_ARM, None),
    "full": (campaign.FULL_ARM, None),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUTS:
        parser.add_argument(
            "--" + name.replace("_", "-") + "-result",
            dest=name,
            type=Path,
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    offered = sum(int(row["offered_victims"]) for row in rows)
    completed = sum(int(row["completed_victims"]) for row in rows)
    good = sum(int(row["slo_good_victims"]) for row in rows)
    rejects = sum(int(row["global_rejects"]) for row in rows)
    failures = sum(int(row["failures"]) for row in rows)
    return {
        "offered_victims": offered,
        "completed_victims": completed,
        "slo_good_victims": good,
        "slo_attainment_fraction_of_offered": good / offered,
        "global_rejects": rejects,
        "failures": failures,
        "worst_e2e_p99_ms": max(float(row["e2e_ms"]["p99"]) for row in rows),
    }


def _summary(blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normal = blocks["normal"]
    overload = _aggregate([blocks["hot_d0"], blocks["hot_d1"]])
    all_phases = _aggregate([blocks[phase] for phase in PHASES])
    return {
        "normal_e2e_p50_ms": float(normal["e2e_ms"]["p50"]),
        "normal_e2e_p99_ms": float(normal["e2e_ms"]["p99"]),
        "hot_d0_e2e_p50_ms": float(blocks["hot_d0"]["e2e_ms"]["p50"]),
        "hot_d0_e2e_p99_ms": float(blocks["hot_d0"]["e2e_ms"]["p99"]),
        "hot_d1_e2e_p50_ms": float(blocks["hot_d1"]["e2e_ms"]["p50"]),
        "hot_d1_e2e_p99_ms": float(blocks["hot_d1"]["e2e_ms"]["p99"]),
        "overload": overload,
        "all_phases": all_phases,
        "route_counts": {
            phase: dict(blocks[phase]["route_counts"]) for phase in PHASES
        },
        "edge_counts": {
            phase: dict(blocks[phase]["edge_counts"]) for phase in PHASES
        },
    }


def _effect(full: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    full_overload = full["overload"]
    base_overload = baseline["overload"]
    full_all = full["all_phases"]
    base_all = baseline["all_phases"]
    _require(base_overload["slo_good_victims"] > 0,
             "baseline overload SLO-good count is zero")
    _require(base_all["slo_good_victims"] > 0,
             "baseline all-phase SLO-good count is zero")
    _require(base_overload["worst_e2e_p99_ms"] > 0,
             "baseline overload p99 is unavailable")
    bad_full = full_overload["global_rejects"] + full_overload["failures"]
    bad_base = base_overload["global_rejects"] + base_overload["failures"]
    return {
        "overload_slo_goodput_ratio": (
            full_overload["slo_good_victims"]
            / base_overload["slo_good_victims"]
        ),
        "all_phase_slo_goodput_ratio": (
            full_all["slo_good_victims"] / base_all["slo_good_victims"]
        ),
        "worst_overload_p99_reduction_fraction": 1.0 - (
            full_overload["worst_e2e_p99_ms"]
            / base_overload["worst_e2e_p99_ms"]
        ),
        "normal_e2e_p50_change_fraction": (
            full["normal_e2e_p50_ms"] / baseline["normal_e2e_p50_ms"] - 1.0
        ),
        "overload_reject_or_fail_delta": bad_full - bad_base,
    }


def _remote_requests(summary: dict[str, Any]) -> int:
    return sum(
        int(count)
        for phase in PHASES
        for route, count in summary["route_counts"][phase].items()
        if route == "official_lmcache_remote_prefill"
    )


def analyze(paths: dict[str, Path]) -> dict[str, Any]:
    _require(set(paths) == set(INPUTS), "C6 ablation input matrix differs")
    nodes: dict[str, dict[str, Any]] = {}
    raws: dict[str, dict[str, Any]] = {}
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    receipts: dict[str, dict[str, Any]] = {}

    for name, (arm, fixed_policy) in INPUTS.items():
        path = paths[name].resolve()
        node, raw = fixed_analysis._load_node(
            path, arm=arm, fixed_policy=fixed_policy)
        contract = Path(node["qualification_contract"]).resolve()
        _require(
            contract.is_file()
            and fixed_analysis._sha256(contract)
            == node["qualification_contract_sha256"],
            f"{name} qualification contract identity differs",
        )
        by_key = fixed_analysis._blocks(raw)
        policy = fixed_policy or arm
        _require(set(by_key) == {(policy, phase) for phase in PHASES},
                 f"{name} phase matrix differs")
        nodes[name] = node
        raws[name] = raw
        blocks[name] = {phase: by_key[(policy, phase)] for phase in PHASES}
        receipts[name] = {
            "result": str(path),
            "result_sha256": fixed_analysis._sha256(path),
            "raw": node["raw"],
            "raw_sha256": node["raw_sha256"],
            "qualification_contract": str(contract),
            "qualification_contract_sha256": (
                node["qualification_contract_sha256"]),
            "slurm_job_id": node["slurm_job_id"],
        }

    source_workloads = {node["source_workload"] for node in nodes.values()}
    profile_hashes = {node["profile_sha256"] for node in nodes.values()}
    _require(len(source_workloads) == 1, "source workload differs across arms")
    _require(len(profile_hashes) == 1, "Elastic profile differs across arms")

    same_population: dict[str, bool] = {}
    for phase in PHASES:
        semantic_hashes = {
            raws[name]["contracts"][blocks[name][phase]["name"]][
                "semantic_schedule_sha256"]
            for name in INPUTS
        }
        same_population[phase] = len(semantic_hashes) == 1
    _require(all(same_population.values()),
             "semantic offered population differs across arms")

    summaries = {name: _summary(value) for name, value in blocks.items()}
    effects = {
        name: _effect(summaries["full"], summaries[name])
        for name in INPUTS if name != "full"
    }
    remote = {name: _remote_requests(value) for name, value in summaries.items()}

    fixed_gate = effects["fixed_p1d0"]
    predictor_gate = effects["predictor"]
    queue_gate = effects["queue_gpu"]
    app_gate = effects["app_global_only"]
    gates = {
        "same_population_all_phases": all(same_population.values()),
        "full_vs_fixed_material_overload": (
            fixed_gate["overload_slo_goodput_ratio"] >= 1.5
            and fixed_gate["worst_overload_p99_reduction_fraction"] >= 0.30
            and fixed_gate["normal_e2e_p50_change_fraction"] <= 0.03
            and fixed_gate["overload_reject_or_fail_delta"] <= 0
        ),
        "full_vs_predictor_material_overload": (
            predictor_gate["overload_slo_goodput_ratio"] >= 1.5
            and predictor_gate["worst_overload_p99_reduction_fraction"] >= 0.30
            and predictor_gate["overload_reject_or_fail_delta"] <= 0
        ),
        "full_vs_predictor_normal_regression_at_most_3pct": (
            predictor_gate["normal_e2e_p50_change_fraction"] <= 0.03),
        "full_vs_queue_gpu_tail_reduction_at_least_30pct": (
            queue_gate["worst_overload_p99_reduction_fraction"] >= 0.30),
        "full_vs_queue_gpu_slo_good_not_lower": (
            queue_gate["overload_slo_goodput_ratio"] >= 1.0),
        "full_vs_queue_gpu_normal_regression_at_most_3pct": (
            queue_gate["normal_e2e_p50_change_fraction"] <= 0.03),
        "full_vs_app_global_incremental_slo_at_least_1_1x": (
            app_gate["overload_slo_goodput_ratio"] >= 1.1),
        "full_vs_app_global_worst_p99_reduction_at_least_5pct": (
            app_gate["worst_overload_p99_reduction_fraction"] >= 0.05),
        "full_activates_remote_lmcache_route": remote["full"] > 0,
    }
    same_allocation_dynamic = len({
        nodes[name]["slurm_job_id"] for name in INPUTS if name != "fixed_p1d0"
    }) == 1
    return {
        "schema": SCHEMA,
        "receipts": receipts,
        "population": {
            "source_workload": next(iter(source_workloads)),
            "profile_sha256": next(iter(profile_hashes)),
            "same_semantic_schedule_by_phase": same_population,
            "same_allocation_dynamic_arms": same_allocation_dynamic,
            "fixed_and_full_same_allocation": (
                nodes["fixed_p1d0"]["slurm_job_id"]
                == nodes["full"]["slurm_job_id"]
            ),
        },
        "arm_summaries": summaries,
        "full_effects_against_baseline": effects,
        "remote_lmcache_request_counts": remote,
        "gates": gates,
        "decoder_global_discovery_positive": (
            gates["full_vs_fixed_material_overload"]
            and gates["full_vs_predictor_material_overload"]
        ),
        "strict_full_superiority_over_queue_gpu": (
            gates["full_vs_queue_gpu_tail_reduction_at_least_30pct"]
            and gates["full_vs_queue_gpu_slo_good_not_lower"]
            and gates["full_vs_queue_gpu_normal_regression_at_most_3pct"]
        ),
        "cross_layer_incremental_superiority_over_app_global": (
            gates["full_vs_app_global_incremental_slo_at_least_1_1x"]
            and gates[
                "full_vs_app_global_worst_p99_reduction_at_least_5pct"]
        ),
        "remote_cross_layer_activation_in_full": (
            gates["full_activates_remote_lmcache_route"]),
        "independent_validation_claim_allowed": False,
    }


def main() -> int:
    args = _parse()
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    paths = {name: getattr(args, name) for name in INPUTS}
    value = analyze(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "decoder_global_discovery_positive": (
            value["decoder_global_discovery_positive"]),
        "strict_full_superiority_over_queue_gpu": (
            value["strict_full_superiority_over_queue_gpu"]),
        "cross_layer_incremental_superiority_over_app_global": (
            value["cross_layer_incremental_superiority_over_app_global"]),
        "remote_cross_layer_activation_in_full": (
            value["remote_cross_layer_activation_in_full"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
