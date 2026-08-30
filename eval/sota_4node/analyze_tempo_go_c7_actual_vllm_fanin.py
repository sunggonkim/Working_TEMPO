#!/usr/bin/env python3
"""Analyze the frozen actual-vLLM two-prefill receiver-incast qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


SCHEMA = "tempo-go-c7-actual-vllm-fanin-analysis-v1"
BUNDLE_SCHEMA = "tempo-go-c7-actual-vllm-fanin-client-v1"
CONTRACT_SCHEMA = "tempo-go-c7-actual-vllm-fanin-contract-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: list[float], fraction: float) -> float:
    _require(bool(values), "quantile population is empty")
    _require(0.0 < fraction <= 1.0, "quantile fraction is invalid")
    ordered = sorted(values)
    index = max(0, min(
        len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _request_metric(row: dict[str, Any]) -> dict[str, float]:
    arrivals = row.get("token_arrival_offsets_ns")
    dispatch = row.get("dispatch_offset_ns")
    end = row.get("stream_end_offset_ns")
    _require(isinstance(arrivals, list) and arrivals,
             "completed request lacks token arrivals")
    _require(isinstance(dispatch, int) and isinstance(end, int),
             "completed request lacks client timestamps")
    return {
        "ttft_ms": (arrivals[0] - dispatch) / 1e6,
        "e2e_ms": (end - dispatch) / 1e6,
        "tpot_ms": (
            (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e6
            if len(arrivals) > 1 else 0.0
        ),
    }


def _metric_summary(metrics: list[dict[str, float]]) -> dict[str, object]:
    _require(bool(metrics), "metric population is empty")
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


def _load_block(
    *, name: str, path: Path, block_contract: dict[str, object],
    section: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(rows, list) and isinstance(decisions, list),
             f"{name} terminal evidence is missing")
    row_index = {row.get("request_id"): row for row in rows}
    decision_index = {row.get("request_id"): row for row in decisions}
    request_index = block_contract.get("request_index")
    _require(isinstance(request_index, dict) and request_index,
             f"{name} request index is missing")
    _require(
        len(row_index) == len(rows)
        and len(decision_index) == len(decisions)
        and set(row_index) == set(decision_index) == set(request_index),
        f"{name} request identities differ",
    )
    victim_metrics: list[dict[str, float]] = []
    aggressor_metrics: list[dict[str, float]] = []
    source_counts = {"0": 0, "1": 0}
    victim_source_counts = {"0": 0, "1": 0}
    arrival_offsets: list[float] = []
    for request_id, metadata in request_index.items():
        row = row_index[request_id]
        decision = decision_index[request_id]
        _require(row.get("valid") is True,
                 f"{name} has an invalid request")
        _require(row.get("terminal_kind") != "global_reject",
                 f"{name} has an unexpected reject")
        _require(decision.get("route") == "official_lmcache_remote_prefill",
                 f"{name} escaped official LMCache remote")
        _require(decision.get("remote_decoder_index") == 0,
                 f"{name} escaped decoder D0")
        source = str(metadata.get("source_prefill_index"))
        _require(source in source_counts,
                 f"{name} has an invalid source identity")
        _require(decision.get("frontend_pair_index") == int(source),
                 f"{name} source P identity differs")
        metric = _request_metric(row)
        role = metadata.get("role")
        if role == "victim":
            victim_metrics.append(metric)
            victim_source_counts[source] += 1
        elif role == "aggressor":
            aggressor_metrics.append(metric)
            source_counts[source] += 1
            arrival_offsets.append(float(metadata["arrival_offset_ms"]))
        else:
            raise ValueError(f"{name} has an invalid request role")

    victim_slo = section["victim"]["slo"]
    slo_good = sum(
        row["e2e_ms"] <= float(victim_slo["e2e_ms"])
        and row["tpot_ms"] <= float(victim_slo["tpot_ms"])
        for row in victim_metrics
    )
    aggressor_rate = float(block_contract["aggressor_rate_per_s"])
    duration_ms = float(section["phase_duration_ms"])
    active_span_ms = (
        max(arrival_offsets) - min(arrival_offsets)
        if len(arrival_offsets) > 1 else 0.0
    )
    two_sender = (
        aggressor_rate == 0.0
        or source_counts["0"] > 0 and source_counts["1"] > 0
    )
    return ({
        "name": name,
        "aggressor_rate_per_s": aggressor_rate,
        "offered_victims": len(victim_metrics),
        "completed_victims": len(victim_metrics),
        "victim": _metric_summary(victim_metrics),
        "victim_slo_good": slo_good,
        "victim_slo_attainment": slo_good / len(victim_metrics),
        "offered_aggressors": len(aggressor_metrics),
        "completed_aggressors": len(aggressor_metrics),
        "aggressor": (
            _metric_summary(aggressor_metrics) if aggressor_metrics else None),
        "aggressor_source_counts": source_counts,
        "victim_source_counts": victim_source_counts,
        "target_decoder_index": 0,
        "actual_two_prefill_fanin": two_sender,
        "aggressor_arrival_span_ms": active_span_ms,
        "service_horizon_covered": (
            aggressor_rate == 0.0
            or active_span_ms >= duration_ms * 0.95
        ),
        "raw": str(path.resolve()),
        "raw_sha256": _sha256(path),
    }, victim_metrics)


def analyze_bundle(
    bundle: dict[str, object], contract_path: Path,
) -> dict[str, object]:
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "bundle schema differs")
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == CONTRACT_SCHEMA,
             "C7 fan-in contract schema differs")
    section = contract.get("actual_vllm_fanin")
    _require(isinstance(section, dict), "C7 fan-in section is missing")
    artifacts = bundle.get("artifacts")
    contracts = bundle.get("contracts")
    _require(isinstance(artifacts, dict) and isinstance(contracts, dict),
             "bundle artifacts/contracts are missing")
    expected_order = [row["name"] for row in section["blocks"]]
    _require(list(artifacts) == expected_order == list(contracts),
             "C7 block order differs")

    summaries: list[dict[str, object]] = []
    victim_populations: dict[str, list[dict[str, float]]] = {}
    for name in expected_order:
        summary, metrics = _load_block(
            name=name,
            path=Path(str(artifacts[name])),
            block_contract=contracts[name],
            section=section,
        )
        summaries.append(summary)
        victim_populations[name] = metrics

    controls = [
        name for name in expected_order
        if float(contracts[name]["aggressor_rate_per_s"]) == 0.0
    ]
    _require(len(controls) == 2, "C7 requires two drift controls")
    control_metrics = [
        metric for name in controls for metric in victim_populations[name]
    ]
    control = _metric_summary(control_metrics)
    victim_slo = section["victim"]["slo"]
    control_slo = sum(
        row["e2e_ms"] <= float(victim_slo["e2e_ms"])
        and row["tpot_ms"] <= float(victim_slo["tpot_ms"])
        for row in control_metrics
    ) / len(control_metrics)
    gates = section["qualification_gates"]
    effects = []
    for summary in summaries:
        rate = float(summary["aggressor_rate_per_s"])
        if rate == 0.0:
            continue
        p50 = float(summary["victim"]["e2e_ms"]["p50"])
        p99 = float(summary["victim"]["e2e_ms"]["p99"])
        p50_degradation = p50 / float(control["e2e_ms"]["p50"]) - 1.0
        p99_ratio = p99 / float(control["e2e_ms"]["p99"])
        slo_drop_pp = (
            control_slo - float(summary["victim_slo_attainment"])) * 100.0
        material = (
            p50_degradation
            >= float(gates["victim_e2e_p50_degradation_fraction"])
            or p99_ratio >= float(gates["victim_e2e_p99_ratio"])
            or slo_drop_pp
            >= float(gates["victim_slo_attainment_drop_percentage_points"])
        )
        effects.append({
            "aggressor_rate_per_s": rate,
            "victim_e2e_p50_degradation_fraction": p50_degradation,
            "victim_e2e_p99_ratio": p99_ratio,
            "victim_slo_attainment_drop_percentage_points": slo_drop_pp,
            "material_victim_degradation": material,
        })

    actual_fanin = all(
        bool(row["actual_two_prefill_fanin"])
        for row in summaries if float(row["aggressor_rate_per_s"]) > 0.0
    )
    horizon = all(
        bool(row["service_horizon_covered"])
        for row in summaries if float(row["aggressor_rate_per_s"]) > 0.0
    )
    same_victim_population = len({
        int(row["offered_victims"]) for row in summaries
    }) == 1
    material = any(row["material_victim_degradation"] for row in effects)
    qualification_pass = (
        actual_fanin and horizon and same_victim_population and material)
    knee = next((
        row["aggressor_rate_per_s"] for row in effects
        if row["material_victim_degradation"]
    ), None)
    return {
        "schema": SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "blocks": summaries,
        "drift_control": {
            "block_names": controls,
            "victim": control,
            "victim_slo_attainment": control_slo,
        },
        "rate_effects": effects,
        "first_material_knee_rate_per_s": knee,
        "actual_two_prefill_to_one_decoder_fanin": actual_fanin,
        "service_horizon_covered": horizon,
        "same_victim_population": same_victim_population,
        "material_independent_victim_degradation": material,
        "c7_actual_vllm_fanin_qualification_pass": qualification_pass,
        "joint_control_discovery_run_allowed": qualification_pass,
        "performance_claim_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    result = analyze_bundle(bundle, args.contract)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
