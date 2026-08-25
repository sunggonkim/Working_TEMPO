#!/usr/bin/env python3
"""Analyze a 60-second actual-vLLM decoder victim ABBA qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from eval.sota_4node import run_tempo_pd_contention_fixed_client as fixed
from tempo.pd_contention_workload import Tenant


SCHEMA = "tempo-go-c6-decoder-victim-abba-analysis-v1"
BUNDLE_SCHEMA = "tempo-go-c6-decoder-victim-client-v1"
BLOCK_SCHEMA = "tempo-go-c6-decoder-victim-block-v1"
CONTRACT_SCHEMA = "tempo-go-c6-qualification-contract-v1"
RAW_SCHEMA = "tempo-pd-stream-metrics-raw-1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "percentile requires samples")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    _require(bool(values), "metric has no samples")
    return {
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _victim_metric(row: dict[str, Any], *, tpot_slo_ms: float, e2e_slo_ms: float) -> dict[str, Any]:
    arrivals = row.get("token_arrival_offsets_ns")
    _require(isinstance(arrivals, list) and len(arrivals) == 128, "victim output is not 128 tokens")
    _require(all(type(value) is int for value in arrivals), "victim token offsets are invalid")
    _require(arrivals == sorted(arrivals), "victim token offsets are not monotonic")
    dispatch = row.get("dispatch_offset_ns")
    stream_end = row.get("stream_end_offset_ns")
    _require(type(dispatch) is int and type(stream_end) is int, "victim terminal offsets are invalid")
    _require(dispatch <= arrivals[0] <= arrivals[-1] <= stream_end, "victim offset order differs")
    ttft_ms = (arrivals[0] - dispatch) / 1_000_000.0
    decode_completion_ms = (arrivals[-1] - arrivals[0]) / 1_000_000.0
    tpot_ms = decode_completion_ms / 127.0
    e2e_ms = (stream_end - dispatch) / 1_000_000.0
    return {
        "request_id": row["request_id"],
        "ttft_ms": ttft_ms,
        "decode_completion_ms": decode_completion_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "slo_pass": tpot_ms <= tpot_slo_ms and e2e_ms <= e2e_slo_ms,
    }


def _block(
    name: str,
    path: Path,
    arm_spec: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    raw = _load(path)
    _require(raw.get("schema") == RAW_SCHEMA, f"{name} raw schema differs")
    _require(
        raw.get("validation", {}).get("performance_claim_allowed") is True,
        f"{name} native child correctness failed",
    )
    block_contract = raw.get("c6_decoder_victim_contract")
    _require(isinstance(block_contract, dict), f"{name} block contract is missing")
    _require(block_contract.get("schema") == BLOCK_SCHEMA, f"{name} block schema differs")
    _require(block_contract.get("name") == name, f"{name} block identity differs")
    _require(
        block_contract.get("aggressor") is arm_spec["aggressor"]
        and block_contract.get("replicate") == arm_spec["replicate"],
        f"{name} ABBA arm differs",
    )
    fixed._validate_endpoint_evidence_bundle(raw.get("endpoint_evidence"))

    request_index = block_contract.get("request_index")
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(request_index, dict), f"{name} request index is missing")
    _require(isinstance(requests, list), f"{name} requests are missing")
    _require(isinstance(decisions, list), f"{name} decisions are missing")
    rows = {row.get("request_id"): row for row in requests}
    decision_rows = {row.get("request_id"): row for row in decisions}
    _require(
        len(rows) == len(requests)
        and len(decision_rows) == len(decisions)
        and set(rows) == set(request_index) == set(decision_rows),
        f"{name} terminal identities differ",
    )

    victim_signature = []
    victim_metrics = []
    background_count = 0
    for request_id, metadata in request_index.items():
        row = rows[request_id]
        decision = decision_rows[request_id]
        _require(row.get("valid") is True, f"{name} request is invalid")
        _require(isinstance(row.get("router"), dict), f"{name} router receipt is missing")
        expected_route = fixed._expected_route(metadata)
        _require(row["router"].get("route") == expected_route, f"{name} route escaped pin")
        _require(decision.get("route") == expected_route, f"{name} decision route escaped pin")
        _require(
            fixed._cold_completion_valid(
                decision, require_explicit_miss=True,
            ),
            f"{name} cold completion is not exact",
        )
        values = row.get("output_token_values")
        _require(isinstance(values, list), f"{name} output values are missing")
        if metadata["tenant"] == Tenant.FOREGROUND.value:
            _require(len(values) == 128, f"{name} victim output count differs")
            victim_signature.append((
                metadata["ordinal"],
                metadata["arrival_offset_ms"],
                metadata["prompt_tokens"],
                metadata["output_tokens"],
                metadata["cache_state"],
                metadata["arm"],
            ))
            victim_metrics.append(_victim_metric(
                row,
                tpot_slo_ms=float(contract["decoder_victim_abba"]["slo"]["tpot_ms"]),
                e2e_slo_ms=float(contract["decoder_victim_abba"]["slo"]["e2e_ms"]),
            ))
        else:
            background_count += 1
            _require(len(values) == 2, f"{name} aggressor output count differs")

    expected_victims = int(
        contract["decoder_victim_abba"]["victim"]["offered_rate_per_s"]
        * contract["decoder_victim_abba"]["phase_duration_ms"]
        / 1000.0
    )
    expected_background = (
        int(
            contract["decoder_victim_abba"]["aggressor"]["offered_rate_per_s"]
            * contract["decoder_victim_abba"]["phase_duration_ms"]
            / 1000.0
        )
        if arm_spec["aggressor"]
        else 0
    )
    _require(len(victim_metrics) == expected_victims, f"{name} victim count differs")
    _require(background_count == expected_background, f"{name} aggressor count differs")
    success_fraction = sum(item["slo_pass"] for item in victim_metrics) / len(victim_metrics)
    return {
        "name": name,
        "aggressor": arm_spec["aggressor"],
        "replicate": arm_spec["replicate"],
        "victim_count": len(victim_metrics),
        "aggressor_count": background_count,
        "victim_signature": sorted(victim_signature),
        "metrics": {
            "ttft_ms": _summary([item["ttft_ms"] for item in victim_metrics]),
            "decode_completion_ms": _summary([
                item["decode_completion_ms"] for item in victim_metrics
            ]),
            "tpot_ms": _summary([item["tpot_ms"] for item in victim_metrics]),
            "e2e_ms": _summary([item["e2e_ms"] for item in victim_metrics]),
            "slo_attainment_fraction": success_fraction,
        },
        "raw": str(path.resolve()),
        "raw_sha256": _sha256(path),
    }


def analyze_bundle(bundle: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _load(contract_path)
    _require(contract.get("schema") == CONTRACT_SCHEMA, "qualification contract differs")
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "decoder victim bundle schema differs")
    specs = contract.get("decoder_victim_abba", {}).get("arms")
    _require(isinstance(specs, list) and len(specs) == 4, "decoder ABBA arms differ")
    _require(
        [item.get("aggressor") for item in specs] == [False, True, True, False],
        "decoder ABBA order differs",
    )
    artifacts = bundle.get("artifacts")
    _require(isinstance(artifacts, dict), "decoder victim artifacts are missing")
    arms = [
        _block(spec["name"], Path(artifacts[spec["name"]]), spec, contract)
        for spec in specs
    ]
    signatures = [item.pop("victim_signature") for item in arms]
    _require(all(value == signatures[0] for value in signatures[1:]), "victim populations differ")

    baseline = [arms[0], arms[3]]
    loaded = [arms[1], arms[2]]
    paired = []
    for clean, hot in zip(baseline, loaded, strict=True):
        clean_p50 = clean["metrics"]["decode_completion_ms"]["p50"]
        hot_p50 = hot["metrics"]["decode_completion_ms"]["p50"]
        clean_p99 = clean["metrics"]["decode_completion_ms"]["p99"]
        hot_p99 = hot["metrics"]["decode_completion_ms"]["p99"]
        paired.append({
            "clean": clean["name"],
            "hot": hot["name"],
            "p50_degradation_fraction": hot_p50 / clean_p50 - 1.0,
            "p99_ratio": hot_p99 / clean_p99,
            "slo_attainment_drop_percentage_points": 100.0 * (
                clean["metrics"]["slo_attainment_fraction"]
                - hot["metrics"]["slo_attainment_fraction"]
            ),
        })
    p50_degradation = statistics.median(
        item["p50_degradation_fraction"] for item in paired
    )
    p99_ratio = statistics.median(item["p99_ratio"] for item in paired)
    slo_drop_pp = statistics.median(
        item["slo_attainment_drop_percentage_points"] for item in paired
    )
    measured_p95_first_response_ms = max(
        item["metrics"]["ttft_ms"]["p95"] for item in arms
    )
    required_phase_duration_ms = max(30_000.0, 3.0 * measured_p95_first_response_ms)
    frozen_phase_duration_ms = float(contract["decoder_victim_abba"]["phase_duration_ms"])
    gates = {
        "same_victim_population_abba": True,
        "all_native_requests_correct_and_cold_exact": True,
        "phase_at_least_30s_and_3xp95_first_response": (
            frozen_phase_duration_ms >= required_phase_duration_ms
        ),
        "recovery_gap_at_least_5s": float(
            contract["decoder_victim_abba"]["cooldown_s"]
        ) >= 5.0,
        "victim_p50_degradation_at_least_25pct": p50_degradation >= 0.25,
        "victim_p99_at_least_2x": p99_ratio >= 2.0,
        "victim_slo_drop_at_least_20pp": slo_drop_pp >= 20.0,
    }
    q1_pass = any(gates[key] for key in (
        "victim_p50_degradation_at_least_25pct",
        "victim_p99_at_least_2x",
        "victim_slo_drop_at_least_20pp",
    ))
    q3_pass = (
        gates["phase_at_least_30s_and_3xp95_first_response"]
        and gates["recovery_gap_at_least_5s"]
    )
    return {
        "schema": SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "arms": arms,
        "paired_effects": paired,
        "aggregate_effect": {
            "median_p50_degradation_fraction": p50_degradation,
            "median_p99_ratio": p99_ratio,
            "median_slo_attainment_drop_percentage_points": slo_drop_pp,
            "measured_p95_first_response_ms": measured_p95_first_response_ms,
            "required_phase_duration_ms": required_phase_duration_ms,
            "frozen_phase_duration_ms": frozen_phase_duration_ms,
        },
        "gates": gates,
        "q1_decoder_output_completion_victim_pass": q1_pass,
        "q3_service_horizon_pass": q3_pass,
        "controller_performance_run_allowed": False,
        "performance_claim_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    value = analyze_bundle(_load(args.input), args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "q1_decoder_output_completion_victim_pass": value[
            "q1_decoder_output_completion_victim_pass"
        ],
        "q3_service_horizon_pass": value["q3_service_horizon_pass"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
