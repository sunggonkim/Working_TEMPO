#!/usr/bin/env python3
"""Analyze the C8 local-protection plus P_ONLY remote-activation campaign."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from eval.sota_4node import analyze_tempo_go_c7_joint_control as c7


ARM_SCHEMA = "tempo-go-c8-dual-regime-arm-analysis-v1"
CAMPAIGN_SCHEMA = "tempo-go-c8-dual-regime-campaign-analysis-v1"
BUNDLE_SCHEMA = "tempo-go-c8-dual-regime-client-v1"
CONTRACT_SCHEMA = "tempo-go-c8-dual-regime-contract-v1"
REMOTE_ROUTE = "official_lmcache_remote_prefill"
LOCAL_ROUTE = "decoder_local_chunked_prefill"
REMOTE_REGIME = "dual_decoder_hot_p_only_remote_favorable"
BUSINESS_PRIORITY_SERVICE_LANE_BINDING = (
    "vllm_priority_business_dual_route_service_lane")
BUSINESS_PRIORITY_SERVICE_LANE_REASONS = frozenset({
    "global_priority_business_dual_route_service_lane_route_committed",
    "global_priority_business_dual_route_service_lane_promoted",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _priority_lane_receipt(
    decision: dict[str, object],
    global_decision: dict[str, object] | None,
    expected_managed_priority: int | None,
) -> bool:
    """Validate either the legacy or v2 business-dual-route lane receipt.

    The v2 business lane intentionally does not rewrite vLLM's upstream
    request priority.  It is a global queue lease plus an accepted reservation
    recorded in the decision binding.  Requiring the legacy ``strong_remote``
    fields here incorrectly rejects valid v2 native receipts.
    """
    if expected_managed_priority is None or not isinstance(global_decision, dict):
        return False
    legacy = (
        decision.get("upstream_priority_effective")
        == expected_managed_priority
        and decision.get("strong_remote_catchup_priority_applied") is True
        and decision.get("upstream_priority_class") == "strong_remote_catchup"
        and decision.get("global_priority_service_lane_committed") is True
        and global_decision.get("reason") in {
            "global_priority_remote_cache_service_lane_route_committed",
            "global_priority_remote_cache_service_lane_promoted",
        }
        and global_decision.get("queue_lease") is True
        and "vllm_priority_remote_cache_service_lane"
        in global_decision.get("binding_resources", [])
    )
    if legacy:
        return True
    reservation = decision.get("tempo_go_service_lane_reservation")
    if not isinstance(reservation, dict):
        return False
    return (
        decision.get("tempo_go_service_lane_reservation_status") == "accepted"
        and reservation.get("status") == "accepted"
        and reservation.get("queue_lease") is True
        and reservation.get("global_route") == REMOTE_ROUTE
        and decision.get("tempo_go_global_commit_queue_lease") is True
        and global_decision.get("reason") in BUSINESS_PRIORITY_SERVICE_LANE_REASONS
        and BUSINESS_PRIORITY_SERVICE_LANE_BINDING
        in global_decision.get("binding_resources", [])
    )


def _group(
    names: list[str], *, blocks: list[dict[str, object]],
    populations: dict[str, list[dict[str, float]]],
) -> dict[str, object]:
    selected = [row for row in blocks if row["name"] in names]
    metrics = [value for name in names for value in populations[name]]
    offered = sum(int(row["offered_victims"]) for row in selected)
    completed = sum(int(row["completed_victims"]) for row in selected)
    slo_good = sum(int(row["slo_good_victims"]) for row in selected)
    _require(offered > 0, "C8 analysis group is empty")
    routes: collections.Counter[str] = collections.Counter()
    edges: collections.Counter[str] = collections.Counter()
    for row in selected:
        routes.update(row["route_counts"])
        edges.update(row["edge_counts"])
    return {
        "block_names": names,
        "offered_victims": offered,
        "completed_victims": completed,
        "global_rejects": sum(int(row["global_rejects"]) for row in selected),
        "failures": sum(int(row["failures"]) for row in selected),
        "slo_good_victims": slo_good,
        "slo_attainment_fraction_of_offered": slo_good / offered,
        "victim": c7._summary(metrics),
        "route_counts": dict(sorted(routes.items())),
        "edge_counts": dict(sorted(edges.items())),
    }


def _remote_hit_receipt(
    *, raw_path: Path, block_contract: dict[str, object],
    expected_managed_priority: int | None,
    source_balance_required: bool,
) -> dict[str, int | bool]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = {row["request_id"]: row for row in raw["requests"]}
    decisions = {row["request_id"]: row for row in raw["router_decisions"]}
    request_index = block_contract["request_index"]
    remote_completed = 0
    exact_full_hits = 0
    priority_lane_receipts = 0
    cross_pair_remote_receipts = 0
    source_balance_receipts = 0
    cross_pair_source_balance_receipts = 0
    decoder_business_admission_receipts = 0
    for request_id, metadata in request_index.items():
        if metadata["role"] != "victim":
            continue
        row = rows[request_id]
        decision = decisions[request_id]
        if (
            row.get("valid") is True
            and row.get("terminal_kind") not in {
                "global_reject", "service_lane_failure"
            }
            and decision.get("route") == REMOTE_ROUTE
        ):
            remote_completed += 1
            if (
                decision.get("request_cache_contract") == "p_only"
                and decision.get("decision_cache_residency") == "prefill_only"
                and decision.get("completion_cache_residency") == "prefill_only"
                and decision.get("lmcache_source_full_hit_observed") is True
                and decision.get("lmcache_source_cached_tokens")
                == int(metadata["prompt_tokens"])
            ):
                exact_full_hits += 1
            global_decision = decision.get("frontend_tempo_go_decision")
            decoder_admission = decision.get(
                "frontend_decoder_business_admission")
            if (
                isinstance(decoder_admission, dict)
                and decoder_admission.get("status") == "released"
                and decoder_admission.get("admission_class") == "protected"
                and decoder_admission.get("pair_index")
                == decision.get("tempo_go_global_commit_decoder_index")
            ):
                decoder_business_admission_receipts += 1
            if (
                isinstance(global_decision, dict)
                and global_decision.get("prefill_index")
                != global_decision.get("decoder_index")
            ):
                cross_pair_remote_receipts += 1
            source_balance_receipt = bool(
                isinstance(global_decision, dict)
                and global_decision.get("mesh_near_tie_source_balanced")
                is True
                and isinstance(
                    global_decision.get("mesh_near_tie_score_window_ms"),
                    (int, float),
                )
                and isinstance(
                    global_decision.get("mesh_near_tie_score_delta_ms"),
                    (int, float),
                )
                and 0.0
                <= float(global_decision["mesh_near_tie_score_delta_ms"])
                <= float(global_decision["mesh_near_tie_score_window_ms"])
                and "mesh_telemetry_uncertainty_source_virtual_service"
                in global_decision.get("binding_resources", [])
            )
            if source_balance_receipt:
                source_balance_receipts += 1
                if (
                    global_decision.get("prefill_index")
                    != global_decision.get("decoder_index")
                ):
                    cross_pair_source_balance_receipts += 1
            if _priority_lane_receipt(
                decision, global_decision, expected_managed_priority):
                priority_lane_receipts += 1
    return {
        "remote_completed_victims": remote_completed,
        "exact_official_lmcache_full_source_hits": exact_full_hits,
        "all_remote_completions_exact_full_source_hits": (
            remote_completed > 0 and exact_full_hits == remote_completed
        ),
        "managed_priority_lane_receipts": priority_lane_receipts,
        "cross_pair_remote_receipts": cross_pair_remote_receipts,
        "source_balance_receipts": source_balance_receipts,
        "cross_pair_source_balance_receipts": (
            cross_pair_source_balance_receipts),
        "all_cross_pair_remote_completions_use_source_balance": (
            not source_balance_required
            or cross_pair_remote_receipts > 0
            and cross_pair_source_balance_receipts
            == cross_pair_remote_receipts
        ),
        "decoder_business_admission_receipts": (
            decoder_business_admission_receipts),
        "all_managed_remote_completions_use_decoder_business_admission": (
            expected_managed_priority is None
            or remote_completed > 0
            and decoder_business_admission_receipts == remote_completed
        ),
        "all_managed_remote_completions_use_priority_lane": (
            expected_managed_priority is None
            or remote_completed > 0
            and priority_lane_receipts == remote_completed
        ),
    }


def analyze_arm_bundle(
    bundle: dict[str, object], contract_path: Path,
) -> dict[str, object]:
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "C8 bundle schema differs")
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == CONTRACT_SCHEMA,
             "C8 contract schema differs")
    section = contract["joint_control"]
    artifacts = bundle["artifacts"]
    contracts = bundle["contracts"]
    expected = [row["name"] for row in section["blocks"]]
    _require(list(artifacts) == expected == list(contracts),
             "C8 block order differs")

    blocks: list[dict[str, object]] = []
    populations: dict[str, list[dict[str, float]]] = {}
    specs = {str(row["name"]): row for row in section["blocks"]}
    expected_managed_priority = (
        int(section["remote_activation"]["managed_remote_priority"])
        if bundle["arm"] == section["headline_full_arm"] else None
    )
    source_balance_required = bool(
        expected_managed_priority is not None
        and section["remote_activation"].get(
            "mesh_near_tie_source_balance_mode")
        == "telemetry_uncertainty_virtual_service_v1"
    )
    for name in expected:
        raw_path = Path(artifacts[name])
        block, metrics = c7._load_block(
            path=raw_path,
            block_contract=contracts[name],
            section=section,
        )
        spec = specs[name]
        block["pressure_regime"] = spec.get("pressure_regime", "control")
        block["victim_cache_state"] = spec.get(
            "victim_cache_state", section["victim"]["cache_state"])
        block["hot_decoder_indices"] = list(spec.get(
            "hot_decoder_indices",
            [] if spec.get("hot_decoder_index") is None
            else [int(spec["hot_decoder_index"])],
        ))
        if block["pressure_regime"] == REMOTE_REGIME:
            block["remote_activation_receipt"] = _remote_hit_receipt(
                raw_path=raw_path,
                block_contract=contracts[name],
                expected_managed_priority=expected_managed_priority,
                source_balance_required=source_balance_required,
            )
        blocks.append(block)
        populations[name] = metrics

    normal_names = [
        name for name in expected
        if specs[name].get("hot_decoder_index") is None
        and not specs[name].get("hot_decoder_indices")
    ]
    miss_hot_names = [
        name for name in expected
        if specs[name].get("pressure_regime") != REMOTE_REGIME
        and specs[name].get("hot_decoder_index") in (0, 1)
        and specs[name].get(
            "victim_cache_state", section["victim"]["cache_state"]
        ) == "miss"
    ]
    remote_names = [
        name for name in expected
        if specs[name].get("pressure_regime") == REMOTE_REGIME
    ]
    _require(normal_names and miss_hot_names and remote_names,
             "C8 requires normal, MISS-hot, and remote-favorable regimes")
    hot_names = miss_hot_names + remote_names

    route_counts: collections.Counter[str] = collections.Counter()
    edge_counts: collections.Counter[str] = collections.Counter()
    for row in blocks:
        route_counts.update(row["route_counts"])
        edge_counts.update(row["edge_counts"])
    return {
        "schema": ARM_SCHEMA,
        "arm": bundle["arm"],
        "blocks": blocks,
        "normal": _group(
            normal_names, blocks=blocks, populations=populations),
        "miss_hot": _group(
            miss_hot_names, blocks=blocks, populations=populations),
        "remote_favorable": _group(
            remote_names, blocks=blocks, populations=populations),
        "hot": _group(hot_names, blocks=blocks, populations=populations),
        "all": _group(expected, blocks=blocks, populations=populations),
        "route_counts": dict(sorted(route_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
        "terminal_contract_valid_for_every_block": True,
        "same_population_ready_for_campaign_analysis": True,
        "actual_native_transport": True,
        "performance_claim_allowed": False,
    }


def _reduction(full: object, baseline: object) -> float | None:
    if not isinstance(full, (int, float)) or not isinstance(
        baseline, (int, float)
    ) or float(baseline) <= 0.0:
        return None
    return 1.0 - float(full) / float(baseline)


def _regime_effect(
    full: dict[str, object], baseline: dict[str, object], regime: str,
) -> dict[str, float | None]:
    foreground = full[regime]
    reference = baseline[regime]
    base_slo = int(reference["slo_good_victims"])
    full_slo = int(foreground["slo_good_victims"])
    return {
        "slo_good_ratio": (
            full_slo / base_slo if base_slo > 0
            else math.inf if full_slo > 0 else 1.0
        ),
        "e2e_p50_reduction_fraction": _reduction(
            foreground["victim"]["e2e_ms"]["p50"],
            reference["victim"]["e2e_ms"]["p50"],
        ),
        "e2e_p99_reduction_fraction": _reduction(
            foreground["victim"]["e2e_ms"]["p99"],
            reference["victim"]["e2e_ms"]["p99"],
        ),
    }


def _normal_regression(
    full: dict[str, object], baseline: dict[str, object],
) -> float | None:
    reduction = _reduction(
        full["normal"]["victim"]["e2e_ms"]["p50"],
        baseline["normal"]["victim"]["e2e_ms"]["p50"],
    )
    return None if reduction is None else -reduction


def analyze_campaign(
    results: dict[str, Path], contract_path: Path,
) -> dict[str, object]:
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == CONTRACT_SCHEMA,
             "C8 campaign contract schema differs")
    section = contract["joint_control"]
    expected = [row["name"] for row in section["arms"]]
    _require(list(results) == expected, "C8 campaign arm order differs")
    arms: dict[str, dict[str, object]] = {}
    sources = {}
    for arm, path in results.items():
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        analysis = wrapper.get("analysis")
        _require(isinstance(analysis, dict)
                 and analysis.get("schema") == ARM_SCHEMA
                 and analysis.get("arm") == arm,
                 f"C8 arm analysis differs: {arm}")
        # The node wrapper stores a compact analysis snapshot, but the raw
        # bundle is authoritative for provenance gates.  Refresh remote
        # activation receipts here so a newer receipt schema (for example the
        # business-dual-route v2 lane) cannot be rejected merely because the
        # node-side snapshot was produced by an older analyzer.
        raw_bundle_path = Path(wrapper.get("raw", ""))
        if raw_bundle_path.is_file():
            raw_bundle = json.loads(raw_bundle_path.read_text(encoding="utf-8"))
            raw_artifacts = raw_bundle.get("artifacts", {})
            raw_contracts = raw_bundle.get("contracts", {})
            expected_priority = (
                int(section["remote_activation"]["managed_remote_priority"])
                if arm == section["headline_full_arm"] else None
            )
            source_balance_required = bool(
                expected_priority is not None
                and section["remote_activation"].get(
                    "mesh_near_tie_source_balance_mode")
                == "telemetry_uncertainty_virtual_service_v1"
            )
            for block in analysis.get("blocks", []):
                name = block.get("name")
                if (
                    block.get("pressure_regime") == REMOTE_REGIME
                    and isinstance(name, str)
                    and isinstance(raw_artifacts, dict)
                    and isinstance(raw_contracts, dict)
                    and name in raw_artifacts
                    and name in raw_contracts
                ):
                    block["remote_activation_receipt"] = _remote_hit_receipt(
                        raw_path=Path(raw_artifacts[name]),
                        block_contract=raw_contracts[name],
                        expected_managed_priority=expected_priority,
                        source_balance_required=source_balance_required,
                    )
        arms[arm] = analysis
        sources[arm] = {"path": str(path.resolve()), "sha256": _sha256(path)}

    fixed_names = [
        row["name"] for row in section["arms"] if row["kind"] == "fixed"
    ]
    _require(fixed_names and "predictor" in arms and "queue_gpu" in arms,
             "C8 comparison arms are incomplete")
    headline = str(section["headline_full_arm"])
    _require(headline in arms, "C8 headline arm is missing")
    full = arms[headline]

    def strongest(regime: str) -> str:
        return max(fixed_names, key=lambda name: (
            int(arms[name][regime]["slo_good_victims"]),
            -float(arms[name][regime]["victim"]["e2e_ms"]["p99"]),
        ))

    strongest_miss = strongest("miss_hot")
    strongest_remote = strongest("remote_favorable")
    comparison_names = list(dict.fromkeys(
        [strongest_miss, strongest_remote, "predictor", "queue_gpu"]
    ))
    effects = {
        name: {
            "miss_hot": _regime_effect(full, arms[name], "miss_hot"),
            "remote_favorable": _regime_effect(
                full, arms[name], "remote_favorable"),
            "normal_e2e_p50_regression_fraction": _normal_regression(
                full, arms[name]),
        }
        for name in comparison_names
    }
    gates = section["performance_gates"]
    c8_gates = section["remote_activation_gates"]

    def robustness(name: str, regime: str) -> bool:
        effect = effects[name][regime]
        regression = effects[name]["normal_e2e_p50_regression_fraction"]
        p99 = effect["e2e_p99_reduction_fraction"]
        return (
            isinstance(regression, (int, float))
            and regression <= float(gates["normal_p50_regression_fraction"])
            and (
                isinstance(p99, (int, float))
                and p99 >= float(gates["hot_p99_reduction_fraction"])
                or effect["slo_good_ratio"] >= float(gates["hot_slo_good_ratio"])
            )
        )

    remote_group = full["remote_favorable"]
    remote_completed = int(remote_group["completed_victims"])
    remote_routes = int(remote_group["route_counts"].get(REMOTE_ROUTE, 0))
    remote_fraction = (
        remote_routes / remote_completed if remote_completed else 0.0)
    remote_blocks = [
        row for row in full["blocks"]
        if row.get("pressure_regime") == REMOTE_REGIME
    ]
    exact_hits = sum(
        int(row["remote_activation_receipt"][
            "exact_official_lmcache_full_source_hits"])
        for row in remote_blocks
    )
    full_hit_gate = (
        remote_routes > 0 and exact_hits == remote_routes
        and all(row["remote_activation_receipt"][
            "all_remote_completions_exact_full_source_hits"]
                for row in remote_blocks)
    )
    priority_lane_gate = (
        remote_routes > 0
        and sum(
            int(row["remote_activation_receipt"][
                "managed_priority_lane_receipts"])
            for row in remote_blocks
        ) == remote_routes
        and all(row["remote_activation_receipt"][
            "all_managed_remote_completions_use_priority_lane"]
                for row in remote_blocks)
    )
    cross_pair_remote_receipts = sum(
        int(row["remote_activation_receipt"][
            "cross_pair_remote_receipts"])
        for row in remote_blocks
    )
    cross_pair_remote_fraction = (
        cross_pair_remote_receipts / remote_routes if remote_routes else 0.0)
    cross_pair_remote_gate = (
        cross_pair_remote_receipts > 0
        and cross_pair_remote_fraction >= float(
            c8_gates["minimum_cross_pair_remote_fraction"])
    )
    source_balance_required = (
        section["remote_activation"].get(
            "mesh_near_tie_source_balance_mode")
        == "telemetry_uncertainty_virtual_service_v1"
    )
    source_balance_receipts = sum(
        int(row["remote_activation_receipt"].get(
            "source_balance_receipts", 0))
        for row in remote_blocks
    )
    cross_pair_source_balance_receipts = sum(
        int(row["remote_activation_receipt"].get(
            "cross_pair_source_balance_receipts", 0))
        for row in remote_blocks
    )
    source_balance_gate = (
        not source_balance_required
        or cross_pair_remote_receipts > 0
        and cross_pair_source_balance_receipts
        == cross_pair_remote_receipts
        and all(row["remote_activation_receipt"].get(
            "all_cross_pair_remote_completions_use_source_balance") is True
                for row in remote_blocks)
    )
    decoder_business_admission_gate = (
        remote_routes > 0
        and sum(
            int(row["remote_activation_receipt"][
                "decoder_business_admission_receipts"])
            for row in remote_blocks
        ) == remote_routes
        and all(row["remote_activation_receipt"][
            "all_managed_remote_completions_use_decoder_business_admission"]
                for row in remote_blocks)
    )
    best_remote = arms[strongest_remote]["remote_favorable"]
    remote_noninferior = (
        int(remote_group["slo_good_victims"])
        >= float(c8_gates["best_fixed_slo_retention_fraction"])
        * int(best_remote["slo_good_victims"])
        and float(remote_group["victim"]["e2e_ms"]["p99"])
        <= float(c8_gates["best_fixed_p99_ratio_ceiling"])
        * float(best_remote["victim"]["e2e_ms"]["p99"])
    )
    both_routes = (
        int(full["route_counts"].get(LOCAL_ROUTE, 0)) > 0
        and int(full["route_counts"].get(REMOTE_ROUTE, 0)) > 0
    )
    correctness = all(
        int(value["all"]["failures"]) == 0
        and value["terminal_contract_valid_for_every_block"] is True
        and value["actual_native_transport"] is True
        for value in arms.values()
    )
    miss_fixed = robustness(strongest_miss, "miss_hot")
    miss_predictor = robustness("predictor", "miss_hot")
    remote_predictor = robustness("predictor", "remote_favorable")
    activation = remote_fraction >= float(
        c8_gates["minimum_full_remote_fraction"])
    positive = (
        correctness
        and miss_fixed
        and miss_predictor
        and activation
        and full_hit_gate
        and priority_lane_gate
        and decoder_business_admission_gate
        and cross_pair_remote_gate
        and source_balance_gate
        and remote_noninferior
        and remote_predictor
        and both_routes
    )
    return {
        "schema": CAMPAIGN_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "sources": sources,
        "arms": arms,
        "headline_full_arm": headline,
        "strongest_fixed_miss_hot_arm": strongest_miss,
        "strongest_fixed_remote_favorable_arm": strongest_remote,
        "effects": effects,
        "correctness_gate": correctness,
        "miss_hot_vs_strongest_fixed_robustness_gate": miss_fixed,
        "miss_hot_vs_predictor_robustness_gate": miss_predictor,
        "remote_favorable_vs_predictor_robustness_gate": remote_predictor,
        "remote_favorable_best_fixed_noninferiority_gate": remote_noninferior,
        "remote_favorable_remote_fraction": remote_fraction,
        "remote_favorable_activation_gate": activation,
        "remote_favorable_exact_lmcache_full_hit_gate": full_hit_gate,
        "remote_favorable_priority_service_lane_gate": priority_lane_gate,
        "remote_favorable_decoder_business_admission_gate": (
            decoder_business_admission_gate),
        "remote_favorable_cross_pair_remote_receipts": (
            cross_pair_remote_receipts),
        "remote_favorable_cross_pair_remote_fraction": (
            cross_pair_remote_fraction),
        "remote_favorable_cross_pair_remote_gate": cross_pair_remote_gate,
        "remote_favorable_source_balance_required": source_balance_required,
        "remote_favorable_source_balance_receipts": source_balance_receipts,
        "remote_favorable_cross_pair_source_balance_receipts": (
            cross_pair_source_balance_receipts),
        "remote_favorable_source_balance_gate": source_balance_gate,
        "full_uses_both_local_and_remote": both_routes,
        "c8_dual_regime_discovery_positive": positive,
        "performance_claim_allowed": positive,
        "independent_validation_claim_allowed": False,
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
