#!/usr/bin/env python3
"""Analyze C7 receiver-incast arms and the same-population campaign."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


ARM_SCHEMA = "tempo-go-c7-joint-control-arm-analysis-v1"
CAMPAIGN_SCHEMA = "tempo-go-c7-joint-control-campaign-analysis-v1"
BUNDLE_SCHEMA = "tempo-go-c7-joint-control-client-v1"
CONTRACT_SCHEMA = "tempo-go-c7-joint-control-contract-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: list[float], fraction: float) -> float:
    _require(bool(values), "quantile population is empty")
    ordered = sorted(values)
    return ordered[max(0, min(
        len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def _metric(row: dict[str, Any]) -> dict[str, float]:
    arrivals = row.get("token_arrival_offsets_ns")
    dispatch = row.get("dispatch_offset_ns")
    end = row.get("stream_end_offset_ns")
    _require(isinstance(arrivals, list) and arrivals,
             "completed victim lacks token arrivals")
    _require(isinstance(dispatch, int) and isinstance(end, int),
             "completed victim lacks client timestamps")
    return {
        "ttft_ms": (arrivals[0] - dispatch) / 1e6,
        "e2e_ms": (end - dispatch) / 1e6,
        "tpot_ms": (
            (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e6
            if len(arrivals) > 1 else 0.0
        ),
    }


def _summary(metrics: list[dict[str, float]]) -> dict[str, object]:
    if not metrics:
        empty: dict[str, object] = {"count": 0}
        for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
            empty[name] = {
                "p50": None, "p95": None, "p99": None, "mean": None,
            }
        return empty
    result: dict[str, object] = {"count": len(metrics)}
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [row[name] for row in metrics]
        result[name] = {
            "p50": _quantile(values, 0.50),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
            "mean": statistics.fmean(values),
        }
    return result


def _edge(decision: dict[str, object]) -> str:
    route = decision.get("route")
    committed_edge = decision.get("tempo_go_global_commit_edge_id")
    if committed_edge is not None:
        source = decision.get("tempo_go_global_commit_prefill_index")
        decoder = decision.get("tempo_go_global_commit_decoder_index")
        committed_route = decision.get("tempo_go_global_commit_route")
        _require(
            type(source) is int
            and type(decoder) is int
            and committed_route == route,
            "global edge receipt lacks canonical P/D identity",
        )
        expected = (
            f"local:d{decoder}"
            if route == "decoder_local_chunked_prefill"
            else f"remote:p{source}->d{decoder}"
            if route == "official_lmcache_remote_prefill"
            else None
        )
        _require(
            isinstance(committed_edge, str)
            and committed_edge == expected,
            "global edge receipt is not canonical",
        )
        return committed_edge
    if route == "decoder_local_chunked_prefill":
        return f"local:d{int(decision['local_decoder_index'])}"
    if route == "official_lmcache_remote_prefill":
        return (
            f"remote:p{int(decision['frontend_pair_index'])}"
            f"->d{int(decision['remote_decoder_index'])}"
        )
    raise ValueError("admitted decision has an invalid route")


def _load_block(
    *, path: Path, block_contract: dict[str, object],
    section: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["requests"]
    decisions = raw["router_decisions"]
    row_index = {row["request_id"]: row for row in rows}
    decision_index = {row["request_id"]: row for row in decisions}
    request_index = block_contract["request_index"]
    _require(
        len(row_index) == len(rows)
        and len(decision_index) == len(decisions)
        and set(row_index) == set(decision_index) == set(request_index),
        "joint-control block identities differ",
    )
    metrics: list[dict[str, float]] = []
    routes: collections.Counter[str] = collections.Counter()
    edges: collections.Counter[str] = collections.Counter()
    decoder_counts: collections.Counter[str] = collections.Counter()
    rejects = failures = 0
    slo_good = 0
    slo = section["victim"]["slo"]
    terminal_by_role: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    terminal_by_tenant: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    for request_id, metadata in request_index.items():
        row = row_index[request_id]
        terminal = str(row.get("terminal_kind") or (
            "complete" if row.get("valid") is True else "invalid"))
        role = str(metadata["role"])
        tenant = str(metadata["business_tenant"])
        terminal_by_role[role][terminal] += 1
        terminal_by_tenant[tenant][terminal] += 1
    for request_id, metadata in request_index.items():
        if metadata["role"] != "victim":
            continue
        row = row_index[request_id]
        if row.get("terminal_kind") == "global_reject":
            rejects += 1
            continue
        # The stream collector deliberately keeps ``valid`` true for a
        # receipted service-lane failure: the terminal contract is valid even
        # though no completion tokens exist.  Such a row is an execution
        # failure, not a metric-bearing completion, and must not be passed to
        # _metric().
        if row.get("terminal_kind") in {
            "service_lane_failure", "route_failure",
        }:
            failures += 1
            continue
        if row.get("valid") is not True:
            failures += 1
            continue
        value = _metric(row)
        metrics.append(value)
        if (
            value["e2e_ms"] <= float(slo["e2e_ms"])
            and value["tpot_ms"] <= float(slo["tpot_ms"])
        ):
            slo_good += 1
        decision = decision_index[request_id]
        routes[str(decision["route"])] += 1
        edge = _edge(decision)
        edges[edge] += 1
        decoder = (
            decision["local_decoder_index"]
            if decision["route"] == "decoder_local_chunked_prefill"
            else decision["remote_decoder_index"]
        )
        decoder_counts[str(decoder)] += 1
    offered = int(block_contract["request_counts"]["victim"])
    _require(len(metrics) + rejects + failures == offered,
             "victim terminal accounting differs")
    return ({
        "name": block_contract["name"],
        "hot_decoder_index": block_contract["hot_decoder_index"],
        "aggressor_rate_per_s": block_contract["aggressor_rate_per_s"],
        "offered_victims": offered,
        "completed_victims": len(metrics),
        "global_rejects": rejects,
        "failures": failures,
        "slo_good_victims": slo_good,
        "slo_attainment_fraction_of_offered": slo_good / offered,
        "victim": _summary(metrics),
        "route_counts": dict(sorted(routes.items())),
        "edge_counts": dict(sorted(edges.items())),
        "decoder_counts": dict(sorted(decoder_counts.items())),
        "terminal_counts_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(terminal_by_role.items())
        },
        "terminal_counts_by_business_tenant": {
            tenant: dict(sorted(counts.items()))
            for tenant, counts in sorted(terminal_by_tenant.items())
        },
        "raw": str(path.resolve()),
        "raw_sha256": _sha256(path),
    }, metrics)


def analyze_arm_bundle(
    bundle: dict[str, object], contract_path: Path,
) -> dict[str, object]:
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "joint bundle schema differs")
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == CONTRACT_SCHEMA,
             "joint contract schema differs")
    section = contract["joint_control"]
    artifacts = bundle["artifacts"]
    contracts = bundle["contracts"]
    expected = [row["name"] for row in section["blocks"]]
    _require(list(artifacts) == expected == list(contracts),
             "joint block order differs")
    blocks = []
    populations: dict[str, list[dict[str, float]]] = {}
    for name in expected:
        block, metrics = _load_block(
            path=Path(artifacts[name]),
            block_contract=contracts[name],
            section=section,
        )
        blocks.append(block)
        populations[name] = metrics

    def group(names: list[str]) -> dict[str, object]:
        selected = [row for row in blocks if row["name"] in names]
        metrics = [value for name in names for value in populations[name]]
        offered = sum(int(row["offered_victims"]) for row in selected)
        completed = sum(int(row["completed_victims"]) for row in selected)
        slo_good = sum(int(row["slo_good_victims"]) for row in selected)
        return {
            "block_names": names,
            "offered_victims": offered,
            "completed_victims": completed,
            "global_rejects": sum(int(row["global_rejects"]) for row in selected),
            "failures": sum(int(row["failures"]) for row in selected),
            "slo_good_victims": slo_good,
            "slo_attainment_fraction_of_offered": slo_good / offered,
            "victim": _summary(metrics),
        }

    normal_names = [
        row["name"] for row in section["blocks"]
        if row["hot_decoder_index"] is None
    ]
    hot_names = [
        row["name"] for row in section["blocks"]
        if row["hot_decoder_index"] in (0, 1)
    ]
    all_names = expected
    route_counts: collections.Counter[str] = collections.Counter()
    edge_counts: collections.Counter[str] = collections.Counter()
    for row in blocks:
        route_counts.update(row["route_counts"])
        edge_counts.update(row["edge_counts"])
    return {
        "schema": ARM_SCHEMA,
        "arm": bundle["arm"],
        "blocks": blocks,
        "normal": group(normal_names),
        "hot": group(hot_names),
        "all": group(all_names),
        "route_counts": dict(sorted(route_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
        "terminal_contract_valid_for_every_block": True,
        "same_population_ready_for_campaign_analysis": True,
        "actual_native_transport": True,
        "performance_claim_allowed": False,
    }


def _fractional_reduction(
    full_value: object, baseline_value: object,
) -> float | None:
    if not isinstance(full_value, (int, float)) or not isinstance(
        baseline_value, (int, float)
    ) or float(baseline_value) <= 0.0:
        return None
    return 1.0 - float(full_value) / float(baseline_value)


def _effect(
    full: dict[str, object], baseline: dict[str, object],
) -> dict[str, float | None]:
    full_hot = full["hot"]
    base_hot = baseline["hot"]
    full_normal = full["normal"]
    base_normal = baseline["normal"]
    base_slo = int(base_hot["slo_good_victims"])
    full_slo = int(full_hot["slo_good_victims"])
    return {
        "hot_slo_good_ratio": (
            full_slo / base_slo if base_slo > 0
            else math.inf if full_slo > 0 else 1.0
        ),
        "hot_e2e_p99_reduction_fraction": _fractional_reduction(
            full_hot["victim"]["e2e_ms"]["p99"],
            base_hot["victim"]["e2e_ms"]["p99"],
        ),
        "hot_e2e_p50_reduction_fraction": _fractional_reduction(
            full_hot["victim"]["e2e_ms"]["p50"],
            base_hot["victim"]["e2e_ms"]["p50"],
        ),
        "normal_e2e_p50_regression_fraction": (
            None
            if _fractional_reduction(
                full_normal["victim"]["e2e_ms"]["p50"],
                base_normal["victim"]["e2e_ms"]["p50"],
            ) is None
            else -float(_fractional_reduction(
                full_normal["victim"]["e2e_ms"]["p50"],
                base_normal["victim"]["e2e_ms"]["p50"],
            ))
        ),
        "pooled_e2e_p50_reduction_fraction": _fractional_reduction(
            full["all"]["victim"]["e2e_ms"]["p50"],
            baseline["all"]["victim"]["e2e_ms"]["p50"],
        ),
    }


def analyze_campaign(
    results: dict[str, Path], contract_path: Path,
) -> dict[str, object]:
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == CONTRACT_SCHEMA,
             "joint campaign contract schema differs")
    section = contract["joint_control"]
    expected = [row["name"] for row in section["arms"]]
    _require(list(results) == expected, "joint campaign arm order differs")
    arms: dict[str, dict[str, object]] = {}
    sources = {}
    for arm, path in results.items():
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        analysis = wrapper.get("analysis")
        _require(isinstance(analysis, dict)
                 and analysis.get("schema") == ARM_SCHEMA
                 and analysis.get("arm") == arm,
                 f"joint arm analysis differs: {arm}")
        arms[arm] = analysis
        sources[arm] = {"path": str(path.resolve()), "sha256": _sha256(path)}

    fixed_names = [
        row["name"] for row in section["arms"] if row["kind"] == "fixed"
    ]
    strongest_fixed = max(fixed_names, key=lambda name: (
        int(arms[name]["hot"]["slo_good_victims"]),
        -float(arms[name]["hot"]["victim"]["e2e_ms"]["p99"]),
        -float(arms[name]["all"]["victim"]["e2e_ms"]["p50"]),
    ))
    headline_full_arm = str(section.get("headline_full_arm", "full_c7"))
    _require(headline_full_arm in arms, "headline full arm is missing")
    full = arms[headline_full_arm]
    comparison_names = [
        strongest_fixed, "predictor", "queue_gpu",
        "network_request_only", "app_global_only",
    ]
    if headline_full_arm != "full_c7":
        comparison_names.append("full_c7")
    comparison_names = list(dict.fromkeys(comparison_names))
    effects = {name: _effect(full, arms[name]) for name in comparison_names}
    gates = section["performance_gates"]

    def robustness(name: str) -> bool:
        effect = effects[name]
        normal_regression = effect["normal_e2e_p50_regression_fraction"]
        hot_p99_reduction = effect["hot_e2e_p99_reduction_fraction"]
        return (
            isinstance(normal_regression, (int, float))
            and normal_regression
            <= float(gates["normal_p50_regression_fraction"])
            and (
                (
                    isinstance(hot_p99_reduction, (int, float))
                    and hot_p99_reduction
                    >= float(gates["hot_p99_reduction_fraction"])
                )
                or effect["hot_slo_good_ratio"]
                >= float(gates["hot_slo_good_ratio"])
            )
        )

    incremental = all(
        (
            isinstance(
                effects[name]["hot_e2e_p99_reduction_fraction"],
                (int, float),
            )
            and effects[name]["hot_e2e_p99_reduction_fraction"]
            >= float(gates["incremental_hot_p99_reduction_fraction"])
        )
        or effects[name]["hot_slo_good_ratio"]
        >= float(gates["incremental_hot_slo_good_ratio"])
        for name in ("network_request_only", "app_global_only")
    )
    route_counts = full["route_counts"]
    both_routes = (
        int(route_counts.get("decoder_local_chunked_prefill", 0)) > 0
        and int(route_counts.get("official_lmcache_remote_prefill", 0)) > 0
    )
    # C7-v1 had one hot block per decoder.  The activation matrix deliberately
    # has multiple hot blocks per decoder (remote-cool and combined-hot), so
    # aggregate by decoder rather than baking block names into the gate.
    hot_by_decoder: dict[str, collections.Counter[str]] = {
        "0": collections.Counter(), "1": collections.Counter()
    }
    for row in full["blocks"]:
        hot_decoder = row["hot_decoder_index"]
        if hot_decoder in (0, 1):
            hot_by_decoder[str(hot_decoder)].update(row["decoder_counts"])
    d0_hot = hot_by_decoder["0"]
    d1_hot = hot_by_decoder["1"]
    switch_valid = (
        int(d0_hot.get("1", 0)) > int(d0_hot.get("0", 0))
        and int(d1_hot.get("0", 0)) > int(d1_hot.get("1", 0))
    )
    correctness = all(
        int(analysis["all"]["failures"]) == 0
        and analysis["terminal_contract_valid_for_every_block"] is True
        for analysis in arms.values()
    )
    positive = (
        correctness
        and robustness(strongest_fixed)
        and robustness("predictor")
        and robustness("queue_gpu")
        and incremental
        and both_routes
        and switch_valid
    )
    return {
        "schema": CAMPAIGN_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "sources": sources,
        "arms": arms,
        "headline_full_arm": headline_full_arm,
        "strongest_fixed_arm": strongest_fixed,
        "full_effects": effects,
        "correctness_gate": correctness,
        "full_uses_both_local_and_remote": both_routes,
        "full_switches_away_from_hot_receiver": switch_valid,
        "full_vs_strongest_fixed_robustness_gate": robustness(strongest_fixed),
        "full_vs_predictor_robustness_gate": robustness("predictor"),
        "full_vs_queue_gpu_robustness_gate": robustness("queue_gpu"),
        "cross_layer_incremental_gate": incremental,
        "c7_joint_control_discovery_positive": positive,
        "independent_validation_claim_allowed": False,
        "performance_claim_allowed": positive,
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
