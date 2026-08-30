#!/usr/bin/env python3
"""Analyze one frozen C9 whole-system causal-burst ABBA discovery campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


REGIMES = ("normal", "miss_hot", "remote_favorable")
STRESSED = ("miss_hot", "remote_favorable")
CROSS_LAYER_SIGNAL_NAMES = {
    "nccl_collective_p99_ms",
    "nccl_arrival_spread_ms",
    "lmcache_transfer_p99_ms",
    "cassini_rx_pause_fraction_max",
    "cassini_tx_pause_fraction_max",
    "cassini_by_nic_pause_fraction_max",
    "cassini_host_posted_cycles_per_packet_max",
    "cassini_ecn_fraction_max",
    "cassini_retries",
    "cassini_timeouts",
    "lmcache_remote_semantic_ops_inflight",
    "lmcache_remote_kv_bytes_inflight",
}
OBSERVER_SIGNAL_NAMES = {
    "nccl_collective_p99_ms",
    "nccl_arrival_spread_ms",
    "lmcache_transfer_p99_ms",
}

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        "finite metric required",
    )
    return float(value)


def _optional_finite(value: Any) -> float | None:
    """Validate a metric while preserving an empty completed population.

    Under the native overload contract a regime may have offered requests but
    zero completed victims.  That is an observable negative outcome, not a
    reason for the analyzer itself to crash.  Keep it as JSON null; callers
    must make the corresponding performance gate fail closed.
    """

    if value is None:
        return None
    return _finite(value)


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        _finite(row[key]) for row in rows if row.get(key) is not None
    ]
    return mean(values) if values else None


def _contribution_names(plan: Any) -> set[str]:
    if not isinstance(plan, dict):
        return set()
    names: set[str] = set()
    for item in plan.get("signal_contributions", []):
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
            names.add(item[0])
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(item["name"])
    return names


def _normalized_cross_layer_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.rsplit(".", 1)[-1]
    return name if name in CROSS_LAYER_SIGNAL_NAMES else None


def _provenance_cross_layer_names(decision: Any) -> set[str]:
    """Return supported endpoint or shared-fabric signals used by a decision."""
    if not isinstance(decision, dict):
        return set()
    provenance = decision.get("telemetry_provenance")
    if not isinstance(provenance, dict):
        return set()
    names: set[str] = set()
    for record in provenance.values():
        if not isinstance(record, dict):
            continue
        cross_layer = record.get("cross_layer")
        if isinstance(cross_layer, dict):
            for signal in cross_layer.get("signals", []):
                if not isinstance(signal, dict):
                    continue
                name = _normalized_cross_layer_name(signal.get("name"))
                if (
                    name is not None
                    and signal.get("support") == "supported"
                    and signal.get("value") is not None
                ):
                    names.add(name)
        groups = record.get("groups")
        if isinstance(groups, dict):
            for group in groups.values():
                if not isinstance(group, dict):
                    continue
                for item in group.get("contributions", []):
                    if not isinstance(item, dict):
                        continue
                    name = _normalized_cross_layer_name(item.get("name"))
                    if name is not None:
                        names.add(name)
    return names


def _shared_provenance_controls(decision: Any) -> dict[str, bool]:
    controls = {
        "dispatch_stagger": False,
        "pair_activation_suppressed": False,
        "limited": False,
    }
    if not isinstance(decision, dict):
        return controls
    provenance = decision.get("telemetry_provenance")
    if not isinstance(provenance, dict):
        return controls
    for record in provenance.values():
        if not isinstance(record, dict):
            continue
        groups = record.get("groups")
        if not isinstance(groups, dict):
            continue
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            controls["dispatch_stagger"] |= (
                int(group.get("dispatch_stagger_us", 0) or 0) > 0
            )
            controls["pair_activation_suppressed"] |= (
                group.get("suppress_pair_activation") is True
            )
            controls["limited"] |= group.get("limited") is True
    return controls


def _business_terminal_outcome(
    terminal: dict[str, Any], *, expected_output_tokens: int,
) -> str:
    """Classify service outcome independently of terminal parse validity.

    A fail-closed HTTP 503 can be a valid terminal-contract receipt, but it
    is not a completed business request.  Counting every ``valid=True`` row
    as completed inflated C9 completion whenever endpoint service-lane
    preflight returned a well-formed failure response.
    """
    _require(expected_output_tokens >= 0, "nonnegative output count required")
    if terminal.get("terminal_kind") == "global_reject":
        return "global_reject"
    output_values = terminal.get("output_token_values", [])
    _require(isinstance(output_values, list), "terminal output list required")
    if (
        terminal.get("terminal_kind") in {None, "complete"}
        and
        terminal.get("valid") is True
        and terminal.get("http_status") == 200
        and terminal.get("done_seen") is True
        and not terminal.get("transport_error")
        and len(output_values) == expected_output_tokens
    ):
        return "completed"
    return "failure"


def _decision_telemetry(result: dict[str, Any]) -> dict[str, Any]:
    analysis = result["analysis"]
    victim_decisions = 0
    victim_admits = 0
    victim_rejects = 0
    cross_layer_available = 0
    plans_with_cross_layer = 0
    cross_layer_actuated = 0
    signal_counts: dict[str, int] = {}
    observer_signal_counts: dict[str, int] = {}
    staggered = 0
    pair_activation_suppressed = 0
    critical = 0
    constrained = 0
    raw_files = []
    by_block: dict[str, dict[str, int]] = {}
    business_by_block: dict[str, dict[str, dict[str, int]]] = {}
    for block in analysis["blocks"]:
        raw_path = Path(block["raw"]).resolve()
        _require(raw_path.is_file(), f"raw block missing: {raw_path}")
        _require(_sha256(raw_path) == block["raw_sha256"],
                 f"raw block SHA differs: {raw_path}")
        raw_files.append(str(raw_path))
        raw = _load(raw_path)
        workload_contract = raw.get("c8_dual_regime_contract")
        if workload_contract is None:
            workload_contract = raw.get("c7_joint_control_contract")
        _require(isinstance(workload_contract, dict),
                 f"raw workload contract missing: {raw_path}")
        request_index = workload_contract.get("request_index")
        _require(isinstance(request_index, dict),
                 f"raw request index missing: {raw_path}")
        block_name = str(block["name"])
        block_decisions = {
            "victim_global_decisions": 0,
            "observer_supported_decisions": 0,
            "observer_actuated_decisions": 0,
        }
        request_rows = {
            row.get("request_id"): row
            for row in raw.get("requests", [])
            if isinstance(row, dict) and isinstance(row.get("request_id"), str)
        }
        _require(
            len(request_rows) == len(request_index)
            and set(request_rows) == set(request_index),
            f"raw request terminals differ: {raw_path}",
        )
        block_business = {
            "foreground": {
                "offered": 0, "completed": 0, "global_rejects": 0,
                "failures": 0, "offered_output_tokens": 0,
                "completed_output_tokens": 0,
            },
            "background": {
                "offered": 0, "completed": 0, "global_rejects": 0,
                "failures": 0, "offered_output_tokens": 0,
                "completed_output_tokens": 0,
            },
        }
        for request_id, identity in request_index.items():
            _require(isinstance(identity, dict),
                     f"raw request identity differs: {raw_path}")
            business_class = (
                "foreground" if identity.get("role") == "victim"
                else "background"
            )
            metrics = block_business[business_class]
            metrics["offered"] += 1
            offered_tokens = int(identity.get("output_tokens", 0) or 0)
            _require(offered_tokens >= 0,
                     f"raw offered output tokens differ: {raw_path}")
            metrics["offered_output_tokens"] += offered_tokens
            terminal = request_rows[request_id]
            terminal_outcome = _business_terminal_outcome(
                terminal, expected_output_tokens=offered_tokens)
            if terminal_outcome == "global_reject":
                metrics["global_rejects"] += 1
            elif terminal_outcome == "completed":
                output_values = terminal.get("output_token_values", [])
                _require(isinstance(output_values, list),
                         f"raw output terminal differs: {raw_path}")
                metrics["completed"] += 1
                metrics["completed_output_tokens"] += len(output_values)
            else:
                metrics["failures"] += 1
        for business_class, metrics in block_business.items():
            _require(
                metrics["completed"]
                + metrics["global_rejects"]
                + metrics["failures"]
                == metrics["offered"],
                f"business terminal partition differs for "
                f"{business_class}: {raw_path}",
            )
        for row in raw["router_decisions"]:
            if not isinstance(row, dict):
                continue
            request_id = row.get("request_id")
            identity = request_index.get(request_id, {})
            if identity.get("role") != "victim":
                continue
            decision = row.get("frontend_tempo_go_decision")
            if not (
                isinstance(decision, dict)
                and decision.get("schema") == "tempo-go-global-orchestrator-v1"
                and decision.get("kind") in {"admit", "reject"}
            ):
                continue
            victim_decisions += 1
            block_decisions["victim_global_decisions"] += 1
            if decision["kind"] == "admit":
                victim_admits += 1
            else:
                victim_rejects += 1
            plan = row.get("tempo_go_global_commit_actuation_plan")
            if not isinstance(plan, dict):
                plan = decision.get("joint_actuation")
            plan_names = {
                name
                for raw_name in _contribution_names(plan)
                if (name := _normalized_cross_layer_name(raw_name)) is not None
            }
            names = plan_names | _provenance_cross_layer_names(decision)
            if names:
                cross_layer_available += 1
                for name in names:
                    signal_counts[name] = signal_counts.get(name, 0) + 1
            observer_names = names & OBSERVER_SIGNAL_NAMES
            if observer_names:
                plans_with_cross_layer += 1
                block_decisions["observer_supported_decisions"] += 1
                for name in observer_names:
                    observer_signal_counts[name] = (
                        observer_signal_counts.get(name, 0) + 1
                    )
            shared_controls = _shared_provenance_controls(decision)
            decision_staggered = bool(
                (isinstance(plan, dict)
                 and int(plan.get("dispatch_stagger_us", 0) or 0) > 0)
                or int(decision.get("receiver_stagger_us", 0) or 0) > 0
                or shared_controls["dispatch_stagger"]
            )
            if decision_staggered:
                staggered += 1
            if shared_controls["pair_activation_suppressed"]:
                pair_activation_suppressed += 1
            decision_constrained = False
            if isinstance(plan, dict):
                if plan.get("critical_guard") is True:
                    critical += 1
                limits = (
                    ("local_prefill_token_ms_limit", "enforced_local_prefill_token_ms_limit"),
                    ("remote_prefill_token_ms_limit", "enforced_remote_prefill_token_ms_limit"),
                    ("remote_kv_bytes_limit", "enforced_remote_kv_bytes_limit"),
                    ("remote_semantic_ops_limit", "enforced_remote_semantic_ops_limit"),
                )
                decision_constrained = any(
                    isinstance(plan.get(raw_name), int)
                    and isinstance(plan.get(enforced_name), int)
                    and plan[enforced_name] < plan[raw_name]
                    for raw_name, enforced_name in limits
                )
                if decision_constrained:
                    constrained += 1
            decision_constrained |= shared_controls["limited"]
            if observer_names and (
                plan_names
                or decision_staggered
                or shared_controls["pair_activation_suppressed"]
                or decision_constrained
            ):
                cross_layer_actuated += 1
                block_decisions["observer_actuated_decisions"] += 1
        by_block[block_name] = block_decisions
        business_by_block[block_name] = block_business
    return {
        "victim_global_decisions": victim_decisions,
        "victim_global_admits": victim_admits,
        "victim_global_rejects": victim_rejects,
        "cross_layer_telemetry_available_decisions": cross_layer_available,
        "cross_layer_supported_decisions": plans_with_cross_layer,
        "cross_layer_actuated_decisions": cross_layer_actuated,
        "cross_layer_supported_fraction": (
            plans_with_cross_layer / victim_decisions if victim_decisions else 0.0
        ),
        "signal_decision_counts": dict(sorted(signal_counts.items())),
        "observer_signal_decision_counts": dict(
            sorted(observer_signal_counts.items())
        ),
        "dispatch_stagger_decisions": staggered,
        "pair_activation_suppressed_decisions": pair_activation_suppressed,
        "critical_guard_decisions": critical,
        "constrained_limit_decisions": constrained,
        "by_block": by_block,
        "business_by_block": business_by_block,
        "raw_files": raw_files,
    }


def _sum_business(
    blocks: list[dict[str, Any]], *, names: set[str] | None = None,
) -> dict[str, dict[str, int | float]]:
    fields = (
        "offered", "completed", "global_rejects", "failures",
        "offered_output_tokens", "completed_output_tokens",
    )
    totals: dict[str, dict[str, int | float]] = {}
    for business_class in ("foreground", "background"):
        values = {field: 0 for field in fields}
        for block in blocks:
            by_block = block["decision_telemetry"]["business_by_block"]
            for block_name, classes in by_block.items():
                if names is not None and block_name not in names:
                    continue
                row = classes[business_class]
                for field in fields:
                    values[field] += int(row[field])
        offered = int(values["offered"])
        completed = int(values["completed"])
        values["completion_fraction"] = (
            completed / offered if offered else 1.0
        )
        offered_output_tokens = int(values["offered_output_tokens"])
        completed_output_tokens = int(values["completed_output_tokens"])
        values["output_token_completion_fraction"] = (
            completed_output_tokens / offered_output_tokens
            if offered_output_tokens else 1.0
        )
        totals[business_class] = values
    return totals


def _observer_counts(
    blocks: list[dict[str, Any]], *, names: set[str] | None = None,
) -> tuple[int, int, int]:
    decisions = 0
    supported = 0
    actuated = 0
    for block in blocks:
        rows = block["decision_telemetry"]["by_block"]
        for block_name, value in rows.items():
            if names is not None and block_name not in names:
                continue
            decisions += int(value["victim_global_decisions"])
            supported += int(value["observer_supported_decisions"])
            actuated += int(value["observer_actuated_decisions"])
    return decisions, supported, actuated


def _regime_summary(analysis: dict[str, Any], name: str) -> dict[str, Any]:
    value = analysis[name]
    victim = value["victim"]
    return {
        "offered": int(value["offered_victims"]),
        "completed": int(value["completed_victims"]),
        "slo_good": int(value["slo_good_victims"]),
        "failures": int(value["failures"]),
        "global_rejects": int(value["global_rejects"]),
        "e2e_p50_ms": _optional_finite(victim["e2e_ms"]["p50"]),
        "e2e_p99_ms": _optional_finite(victim["e2e_ms"]["p99"]),
        "ttft_p99_ms": _optional_finite(victim["ttft_ms"]["p99"]),
        "tpot_p99_ms": _optional_finite(victim["tpot_ms"]["p99"]),
        "route_counts": value["route_counts"],
        "edge_counts": value["edge_counts"],
    }


def _arm_aggregate(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime in REGIMES:
        rows = [block["regimes"][regime] for block in blocks]
        offered = sum(row["offered"] for row in rows)
        result[regime] = {
            "runs": len(rows),
            "offered": offered,
            "completed": sum(row["completed"] for row in rows),
            "slo_good": sum(row["slo_good"] for row in rows),
            "slo_good_fraction": (
                sum(row["slo_good"] for row in rows) / offered if offered else 0.0
            ),
            "mean_e2e_p50_ms": _mean_optional(rows, "e2e_p50_ms"),
            "mean_e2e_p99_ms": _mean_optional(rows, "e2e_p99_ms"),
            "mean_ttft_p99_ms": _mean_optional(rows, "ttft_p99_ms"),
            "mean_tpot_p99_ms": _mean_optional(rows, "tpot_p99_ms"),
        }
    return result


def _effect(
    full: dict[str, Any], blind: dict[str, Any]
) -> dict[str, float | None]:
    full_slo = _finite(full["slo_good_fraction"])
    blind_slo = _finite(blind["slo_good_fraction"])
    full_p50 = _optional_finite(full["mean_e2e_p50_ms"])
    blind_p50 = _optional_finite(blind["mean_e2e_p50_ms"])
    full_p99 = _optional_finite(full["mean_e2e_p99_ms"])
    blind_p99 = _optional_finite(blind["mean_e2e_p99_ms"])
    return {
        "full_minus_blind_p50_fraction": (
            (full_p50 - blind_p50) / blind_p50
            if full_p50 is not None and blind_p50 not in (None, 0.0)
            else None
        ),
        "full_p99_reduction_fraction": (
            (blind_p99 - full_p99) / blind_p99
            if full_p99 is not None and blind_p99 not in (None, 0.0)
            else None
        ),
        "full_slo_good_fraction": full_slo,
        "blind_slo_good_fraction": blind_slo,
        "full_slo_good_ratio": (
            full_slo / blind_slo
            if blind_slo > 0.0
            else (1_000_000_000.0 if full_slo > 0.0 else 1.0)
        ),
    }


def _analyze_population_campaign(
    *, contract_path: Path, root: Path, output: Path,
    contract: dict[str, Any], blocks: list[dict[str, Any]],
    all_correct: bool, all_transport: bool, all_sidecars_correct: bool,
) -> int:
    """Analyze a C9 campaign containing fixed, predictor, and TEMPO arms.

    The original C9 receipt format was an ABBA comparison between
    ``app_global_only`` and ``full_c7_managed_background``.  Candidate K needs
    the stronger final comparison: the same causal burst must contain the
    strongest fixed routes and the simple request predictor as well.  Keep the
    legacy four-block analyzer unchanged and use this branch only when a
    source-bound contract explicitly enumerates the larger population.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for block in blocks:
        grouped.setdefault(str(block["arm"]), []).append(block)
    full_name = "full_c7_managed_background"
    fixed_names = sorted(name for name in grouped if name.startswith("fixed_"))
    _require(full_name in grouped, "population campaign lacks TEMPO arm")
    _require(bool(fixed_names), "population campaign lacks fixed arms")
    _require("predictor" in grouped, "population campaign lacks predictor arm")

    aggregates = {
        name: _arm_aggregate(rows) for name, rows in sorted(grouped.items())
    }
    full = aggregates[full_name]
    predictor = aggregates["predictor"]
    best_fixed_arm: dict[str, str] = {}
    best_fixed: dict[str, dict[str, Any]] = {}
    effects_best_fixed: dict[str, dict[str, float | None]] = {}
    effects_predictor: dict[str, dict[str, float | None]] = {}
    for regime in REGIMES:
        selected = min(
            fixed_names,
            key=lambda name: aggregates[name][regime]["mean_e2e_p99_ms"],
        )
        best_fixed_arm[regime] = selected
        best_fixed[regime] = aggregates[selected][regime]
        effects_best_fixed[regime] = _effect(full[regime], best_fixed[regime])
        effects_predictor[regime] = _effect(full[regime], predictor[regime])

    offered_vectors = {
        regime: {
            int(aggregate[regime]["offered"])
            for aggregate in aggregates.values()
        }
        for regime in REGIMES
    }
    thresholds = contract["gates"]
    full_blocks = grouped[full_name]
    full_decisions, full_supported, full_observer_actuated = _observer_counts(
        full_blocks)
    supported_fraction = (
        full_supported / full_decisions if full_decisions else 0.0
    )
    baseline_supported = sum(
        block["decision_telemetry"]["cross_layer_supported_decisions"]
        for name in fixed_names + ["predictor"]
        for block in grouped[name]
    )
    full_actuation = (
        full_observer_actuated > 0
        and any(
            block["decision_telemetry"]["cross_layer_actuated_decisions"] > 0
            for block in full_blocks
        )
    )
    stressed_fixed = [effects_best_fixed[name] for name in STRESSED]
    stressed_predictor = [effects_predictor[name] for name in STRESSED]
    normal_effects = [
        effects_best_fixed["normal"], effects_predictor["normal"]
    ]
    minimum_reduction = _finite(
        thresholds.get("minimum_stressed_p99_reduction_fraction", 0.0))
    minimum_slo_ratio = _finite(
        thresholds.get("minimum_stressed_slo_good_ratio", 1.0))
    normal_limit = _finite(
        thresholds.get("maximum_normal_p50_regression_fraction", 1.0))

    def measured(value: Any) -> bool:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))

    full_business = _sum_business(full_blocks)
    baseline_business = {
        name: _sum_business(grouped[name])
        for name in fixed_names + ["predictor"]
    }
    gates = {
        "correctness": all_correct and all_sidecars_correct,
        "native_transport": all_transport,
        "same_population": all(len(values) == 1 for values in offered_vectors.values()),
        "fixed_baseline_present": bool(fixed_names),
        "predictor_baseline_present": True,
        "full_observer_supported": supported_fraction >= _finite(
            thresholds.get("minimum_full_supported_observer_fraction", 0.0)),
        "full_cross_layer_actuation": full_actuation,
        "baseline_cross_layer_blind": baseline_supported == 0,
        "stressed_p99_reduction_vs_best_fixed": all(
            measured(value["full_p99_reduction_fraction"])
            and value["full_p99_reduction_fraction"] >= minimum_reduction
            for value in stressed_fixed
        ),
        "stressed_p99_reduction_vs_predictor": all(
            measured(value["full_p99_reduction_fraction"])
            and value["full_p99_reduction_fraction"] >= minimum_reduction
            for value in stressed_predictor
        ),
        "stressed_slo_not_lower_vs_best_fixed": all(
            measured(value["full_slo_good_ratio"])
            and value["full_slo_good_ratio"] >= minimum_slo_ratio
            for value in stressed_fixed
        ),
        "stressed_slo_not_lower_vs_predictor": all(
            measured(value["full_slo_good_ratio"])
            and value["full_slo_good_ratio"] >= minimum_slo_ratio
            for value in stressed_predictor
        ),
        "normal_p50_regression_bounded": all(
            measured(value["full_minus_blind_p50_fraction"])
            and value["full_minus_blind_p50_fraction"] <= normal_limit
            for value in normal_effects
        ),
    }
    positive = all(gates.values())
    payload = {
        "schema": "tempo-go-c9-causal-burst-analysis-v1",
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "root": str(root),
        "blocks": blocks,
        "aggregates": aggregates,
        "comparison_baselines": {
            "fixed_arms": fixed_names,
            "best_fixed_arm_by_regime": best_fixed_arm,
            "predictor_arm": "predictor",
        },
        "effects": {
            "vs_best_fixed": effects_best_fixed,
            "vs_predictor": effects_predictor,
        },
        "business": {
            "full_tempo": full_business,
            "baselines": baseline_business,
        },
        "telemetry": {
            "observer_support_scope": "all_victim_global_decisions",
            "full_victim_global_decisions": full_decisions,
            "full_cross_layer_supported_decisions": full_supported,
            "full_cross_layer_supported_fraction": supported_fraction,
            "full_cross_layer_actuation_observed": full_actuation,
            "baseline_cross_layer_supported_decisions": baseline_supported,
        },
        "gates": gates,
        "causal_discovery_positive": positive,
        "claim_boundary": contract["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    root = args.root.resolve()
    output = args.output.resolve()
    _require(not output.exists(), "refusing to overwrite analysis")
    contract = _load(contract_path)
    _require(contract.get("schema") == "tempo-go-c9-causal-burst-discovery-v1",
             "contract schema differs")
    order = contract["execution"]["order"]
    _require(len(order) > 0, "C9 campaign requires at least one block")

    blocks: list[dict[str, Any]] = []
    all_correct = True
    all_transport = True
    all_sidecars_correct = True
    for index, spec in enumerate(order):
        block_root = root / spec["name"]
        result_path = block_root / "inference/result.json"
        cojob_path = block_root / "cojob/result.json"
        failure_path = block_root / "cojob/cojob_failure.json"
        observer_path = block_root / "cojob/nccl_observer.json"
        transport_path = block_root / "cojob/native_transport_receipt.json"
        execution_path = block_root / "block_execution_receipt.json"
        for path in (result_path, observer_path, transport_path, execution_path):
            _require(path.is_file(), f"campaign artifact missing: {path}")
        result = _load(result_path)
        observer = _load(observer_path)
        transport = _load(transport_path)
        execution = _load(execution_path)
        analysis = result["analysis"]
        _require(analysis["arm"] == spec["arm"],
                 f"arm differs in block {spec['name']}")
        _require(
            execution.get("schema")
            == "tempo-go-c9-causal-burst-block-execution-v1"
            and execution.get("block") == spec["name"]
            and execution.get("arm") == spec["arm"]
            and execution.get("inference_status") == "complete"
            and execution.get("measured_arm_retried") is False,
            f"execution receipt differs in block {spec['name']}",
        )
        _require(observer.get("source_epoch") ==
                 f"slurm-{result['slurm_job_id']}-c9-causal-{index}",
                 f"observer epoch differs in block {spec['name']}")
        cojob_outcome = execution.get("cojob_outcome")
        _require(cojob_outcome in {"complete", "overload_timeout"},
                 f"co-job outcome differs in block {spec['name']}")
        cojob: dict[str, Any] | None = None
        failure: dict[str, Any] | None = None
        if cojob_outcome == "complete":
            _require(cojob_path.is_file(),
                     f"complete co-job result missing: {cojob_path}")
            cojob = _load(cojob_path)
            sidecar_outcome_valid = bool(
                execution.get("cojob_exit_code") == 0
                and observer.get("producer_state") == "complete"
                and cojob.get("overall_correctness_met") is True
            )
        else:
            _require(failure_path.is_file(),
                     f"co-job timeout receipt missing: {failure_path}")
            failure = _load(failure_path)
            stderr_paths = sorted((block_root / "cojob").glob(
                "cojob-rank-*.stderr.log"))
            exact_timeout = any(
                "official LMCache/NIXL batched_write exceeded" in
                path.read_text(encoding="utf-8", errors="replace")
                for path in stderr_paths
            )
            sidecar_outcome_valid = bool(
                isinstance(execution.get("cojob_exit_code"), int)
                and execution["cojob_exit_code"] != 0
                and observer.get("producer_state") == "active"
                and observer.get("correctness_met") is True
                and int(observer.get("sequence", 0)) >= 1
                and failure.get("failure") == "cojob_step_failed"
                and exact_timeout
            )
        paired_observer_paths = [observer_path]
        paired_transport_paths = [transport_path]
        paired_sidecars_valid = True
        if execution.get("cojob_pair_count") == 2:
            observer_values = execution.get("observers")
            observer_sha_values = execution.get("observer_sha256s")
            transport_values = execution.get("transport_receipts")
            outcome_values = execution.get("cojob_pair_outcomes")
            dir_values = execution.get("cojob_dirs")
            _require(
                isinstance(observer_values, list)
                and len(observer_values) == 2
                and isinstance(observer_sha_values, list)
                and len(observer_sha_values) == 2
                and isinstance(transport_values, list)
                and len(transport_values) == 2
                and isinstance(outcome_values, list)
                and len(outcome_values) == 2
                and isinstance(dir_values, list)
                and len(dir_values) == 2,
                f"two-pair receipt inventory missing: {execution_path}",
            )
            paired_observer_paths = [Path(value).resolve() for value in observer_values]
            paired_transport_paths = [Path(value).resolve() for value in transport_values]
            root_resolved = block_root.resolve()
            for pair_index in (0, 1):
                _require(
                    paired_observer_paths[pair_index].is_file()
                    and paired_observer_paths[pair_index].is_relative_to(root_resolved),
                    f"pair observer escapes block root: {execution_path}",
                )
                _require(
                    _sha256(paired_observer_paths[pair_index])
                    == observer_sha_values[pair_index],
                    f"pair observer SHA differs: {paired_observer_paths[pair_index]}",
                )
                _require(
                    paired_transport_paths[pair_index].is_file()
                    and paired_transport_paths[pair_index].is_relative_to(root_resolved),
                    f"pair transport receipt escapes block root: {execution_path}",
                )
                pair_transport = _load(paired_transport_paths[pair_index])
                _require(
                    pair_transport.get("production_transport_verified") is True
                    and pair_transport.get("slingshot_path")
                    == "nersc-nccl-ofi-libfabric",
                    f"pair transport receipt invalid: {paired_transport_paths[pair_index]}",
                )
                pair_outcome = str(outcome_values[pair_index])
                _require(
                    pair_outcome in {"complete", "overload_timeout"},
                    f"pair co-job outcome differs: {execution_path}",
                )
                pair_dir = Path(dir_values[pair_index]).resolve()
                _require(pair_dir.is_relative_to(root_resolved),
                         f"pair co-job directory escapes block root: {execution_path}")
                if pair_outcome == "complete":
                    pair_result = pair_dir / "result.json"
                    _require(pair_result.is_file(),
                             f"pair complete result missing: {pair_result}")
                    _require(
                        _load(pair_result).get("overall_correctness_met") is True,
                        f"pair complete correctness failed: {pair_result}",
                    )
                else:
                    pair_failure = pair_dir / "cojob_failure.json"
                    pair_stderr = sorted(pair_dir.glob("cojob-rank-*.stderr.log"))
                    _require(
                        pair_failure.is_file()
                        and _load(pair_failure).get("failure")
                        == "cojob_step_failed"
                        and any(
                            "official LMCache/NIXL batched_write exceeded"
                            in path.read_text(encoding="utf-8", errors="replace")
                            for path in pair_stderr
                        ),
                        f"pair overload receipt invalid: {pair_dir}",
                    )
            paired_sidecars_valid = True
        sidecar_outcome_valid = sidecar_outcome_valid and paired_sidecars_valid
        block_correct = bool(
            analysis.get("terminal_contract_valid_for_every_block")
            and analysis.get("same_population_ready_for_campaign_analysis")
        )
        transport_correct = bool(
            transport.get("production_transport_verified") is True
            and transport.get("slingshot_path") == "nersc-nccl-ofi-libfabric"
            and (
                cojob is None
                or cojob.get("baseline", {}).get("name")
                == "LMCache NixlChannel"
            )
        )
        all_correct = all_correct and block_correct
        all_sidecars_correct = all_sidecars_correct and sidecar_outcome_valid
        all_transport = all_transport and transport_correct
        blocks.append({
            "index": index,
            "name": spec["name"],
            "arm": spec["arm"],
            "result": str(result_path),
            "result_sha256": _sha256(result_path),
            "cojob_result": str(cojob_path) if cojob is not None else None,
            "cojob_result_sha256": (
                _sha256(cojob_path) if cojob is not None else None),
            "cojob_failure": (
                str(failure_path) if failure is not None else None),
            "cojob_failure_sha256": (
                _sha256(failure_path) if failure is not None else None),
            "observer": str(observer_path),
            "observer_sha256": _sha256(observer_path),
            "transport_receipt": str(transport_path),
            "transport_receipt_sha256": _sha256(transport_path),
            "block_execution_receipt": str(execution_path),
            "block_execution_receipt_sha256": _sha256(execution_path),
            "correct": block_correct,
            "cojob_outcome": cojob_outcome,
            "cojob_outcome_valid": sidecar_outcome_valid,
            "native_transport": transport_correct,
            "cojob_pair_count": len(paired_observer_paths),
            "observers": [str(path) for path in paired_observer_paths],
            "transport_receipts": [str(path) for path in paired_transport_paths],
            "cojob": {
                "blocks": (
                    len(cojob["blocks"])
                    if cojob is not None else int(observer["sequence"])),
                "active_loop": (
                    cojob["active_loop"] if cojob is not None else None),
                "nccl_collective_p99_ms": _finite(
                    observer["nccl_collective_p99_ms"]),
                "lmcache_transfer_p99_ms": _finite(
                    observer["lmcache_transfer_p99_ms"]),
                "observer_sequence": int(observer["sequence"]),
                "nixl_timeout_lower_bound_ms": (
                    float(contract["burst"]["nixl_transfer_timeout_s"])
                    * 1000.0
                    if cojob_outcome == "overload_timeout" else None),
            },
            "regimes": {
                name: _regime_summary(analysis, name) for name in REGIMES
            },
            "decision_telemetry": _decision_telemetry(result),
        })

    full_blocks = [
        block for block in blocks
        if block["arm"] == "full_c7_managed_background"
    ]
    blind_blocks = [
        block for block in blocks if block["arm"] == "app_global_only"
    ]
    if not (len(order) == 4 and len(full_blocks) == len(blind_blocks) == 2):
        return _analyze_population_campaign(
            contract_path=contract_path,
            root=root,
            output=output,
            contract=contract,
            blocks=blocks,
            all_correct=all_correct,
            all_transport=all_transport,
            all_sidecars_correct=all_sidecars_correct,
        )
    _require(len(full_blocks) == len(blind_blocks) == 2,
             "ABBA arm counts differ")
    full = _arm_aggregate(full_blocks)
    blind = _arm_aggregate(blind_blocks)
    effects = {name: _effect(full[name], blind[name]) for name in REGIMES}

    thresholds = contract["gates"]
    observer_scope = thresholds.get(
        "observer_support_scope", "all_victim_global_decisions")
    _require(observer_scope in {
        "all_victim_global_decisions",
        "remote_favorable_victim_global_decisions",
    }, "observer support scope differs")
    observer_block_names: set[str] | None = None
    if observer_scope == "remote_favorable_victim_global_decisions":
        observer_block_names = set(
            _load(Path(full_blocks[0]["result"]))["analysis"]
            ["remote_favorable"]["block_names"]
        )
    full_decisions, full_supported, full_observer_actuated = _observer_counts(
        full_blocks, names=observer_block_names)
    _, blind_supported, _ = _observer_counts(
        blind_blocks, names=observer_block_names)
    supported_fraction = (
        full_supported / full_decisions if full_decisions else 0.0
    )
    full_actuation = any(
        block["decision_telemetry"]["cross_layer_actuated_decisions"] > 0
        for block in full_blocks
    ) and full_observer_actuated > 0
    full_business = _sum_business(full_blocks)
    blind_business = _sum_business(blind_blocks)
    remote_names = set(
        _load(Path(full_blocks[0]["result"]))["analysis"]
        ["remote_favorable"]["block_names"]
    )
    full_remote_business = _sum_business(full_blocks, names=remote_names)
    blind_remote_business = _sum_business(blind_blocks, names=remote_names)
    stressed_reductions = [
        effects[name]["full_p99_reduction_fraction"] for name in STRESSED
    ]
    stressed_slo_ratios = [
        effects[name]["full_slo_good_ratio"] for name in STRESSED
    ]
    gates = {
        "correctness": all_correct and all_sidecars_correct,
        "native_transport": all_transport,
        "same_population": all(
            full[name]["offered"] == blind[name]["offered"] for name in REGIMES
        ),
        "full_observer_supported": supported_fraction >= _finite(
            thresholds["minimum_full_supported_observer_fraction"]),
        "ablation_cross_layer_blind": blind_supported == 0,
        "full_cross_layer_actuation": full_actuation,
        "at_least_one_stressed_p99_reduction": max(stressed_reductions)
        >= _finite(thresholds["minimum_stressed_p99_reduction_fraction"]),
        "stressed_slo_not_lower": min(stressed_slo_ratios)
        >= _finite(thresholds["minimum_stressed_slo_good_ratio"]),
        "normal_p50_regression_bounded": effects["normal"][
            "full_minus_blind_p50_fraction"
        ] <= _finite(thresholds["maximum_normal_p50_regression_fraction"]),
    }
    if "minimum_remote_background_completion_fraction" in thresholds:
        gates["remote_background_minimum_service"] = _finite(
            full_remote_business["background"]["completion_fraction"]
        ) >= _finite(
            thresholds["minimum_remote_background_completion_fraction"])
    if "minimum_remote_background_completion_ratio_to_blind" in thresholds:
        full_background_fraction = _finite(
            full_remote_business["background"]["completion_fraction"])
        blind_background_fraction = _finite(
            blind_remote_business["background"]["completion_fraction"])
        background_ratio = (
            full_background_fraction / blind_background_fraction
            if blind_background_fraction > 0.0 else 1.0
        )
        gates["remote_background_retained_vs_blind"] = background_ratio >= _finite(
            thresholds[
                "minimum_remote_background_completion_ratio_to_blind"])
    positive = all(gates.values())
    payload = {
        "schema": "tempo-go-c9-causal-burst-analysis-v1",
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "root": str(root),
        "blocks": blocks,
        "aggregates": {
            "full_c9": full,
            "app_global_only": blind,
        },
        "effects": effects,
        "business": {
            "all": {"full_c9": full_business,
                    "app_global_only": blind_business},
            "remote_favorable": {
                "full_c9": full_remote_business,
                "app_global_only": blind_remote_business,
            },
        },
        "telemetry": {
            "observer_support_scope": observer_scope,
            "full_victim_global_decisions": full_decisions,
            "full_cross_layer_supported_decisions": full_supported,
            "full_cross_layer_supported_fraction": supported_fraction,
            "blind_cross_layer_supported_decisions": blind_supported,
            "full_cross_layer_actuation_observed": full_actuation,
        },
        "gates": gates,
        "causal_discovery_positive": positive,
        "claim_boundary": contract["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
