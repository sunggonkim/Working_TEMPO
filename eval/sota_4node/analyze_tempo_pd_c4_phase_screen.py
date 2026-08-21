#!/usr/bin/env python3
"""Fail-closed analysis for the calibration-only C4 live phase screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


SCHEMA = "tempo-pd-c4-phase-screen-analysis-v2"
NODE_SCHEMA = "tempo-pd-c4-phase-screen-node-v1"
CLIENT_SCHEMA = "tempo-pd-c4-phase-screen-client-v1"
BLOCK_SCHEMA = "tempo-pd-c4-phase-screen-block-v1"
ARMS = ("local", "remote", "predictor", "tempo")
FIXED_ARMS = ("local", "remote")
LOCAL_ROUTE = "decoder_local_chunked_prefill"
REMOTE_ROUTE = "official_lmcache_remote_prefill"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _bounded_file(raw: object, *, root: Path, name: str) -> Path:
    _require(isinstance(raw, str) and Path(raw).is_absolute(),
             f"{name} must be an absolute path")
    path = Path(raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} escapes the result root") from exc
    _require(path.is_file(), f"{name} is missing")
    return path


def _nearest_rank(values: list[float], fraction: float) -> float:
    _require(bool(values), "cannot summarize an empty sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latencies(row: dict[str, Any]) -> tuple[float, float, float]:
    dispatch = int(row["dispatch_offset_ns"])
    end = int(row["stream_end_offset_ns"])
    arrivals = [int(value) for value in row["token_arrival_offsets_ns"]]
    _require(bool(arrivals), "valid request has no token arrivals")
    output_tokens = len(row["output_token_values"])
    return (
        (end - dispatch) / 1_000_000.0,
        (arrivals[0] - dispatch) / 1_000_000.0,
        (end - arrivals[0]) / 1_000_000.0 / max(1, output_tokens - 1),
    )


def _sample(
    row: dict[str, Any], metadata: dict[str, Any], *, arm: str, replicate: int,
    slo: dict[str, float],
) -> dict[str, Any]:
    e2e, ttft, tpot = _latencies(row)
    router = row.get("router")
    _require(isinstance(router, dict), "foreground request lacks router provenance")
    route = router.get("route")
    _require(route in {LOCAL_ROUTE, REMOTE_ROUTE}, "foreground route is invalid")
    return {
        "arm": arm,
        "replicate": replicate,
        "pair_key": str(metadata["pair_key"]),
        "phase": str(metadata["phase"]),
        "prompt_tokens": int(metadata["prompt_tokens"]),
        "output_tokens": int(metadata["output_tokens"]),
        "cache_state": str(metadata["cache_state"]),
        "route": route,
        "reason": str(router.get("reason")),
        "e2e_ms": e2e,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "good": (
            e2e <= slo["e2e_ms"]
            and ttft <= slo["ttft_ms"]
            and tpot <= slo["tpot_ms"]
        ),
        "prompt_sha256": str(row["prompt_sha256"]),
        "output_text_sha256": str(row["output_text_sha256"]),
        "output_token_values": tuple(str(value)
                                     for value in row["output_token_values"]),
    }


def _metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(samples), "metric sample is empty")
    e2e = [float(value["e2e_ms"]) for value in samples]
    ttft = [float(value["ttft_ms"]) for value in samples]
    tpot = [float(value["tpot_ms"]) for value in samples]
    good = sum(bool(value["good"]) for value in samples)
    return {
        "requests": len(samples),
        "e2e_median_ms": statistics.median(e2e),
        "e2e_mean_ms": statistics.fmean(e2e),
        "e2e_p99_ms": _nearest_rank(e2e, 0.99),
        "ttft_median_ms": statistics.median(ttft),
        "ttft_p99_ms": _nearest_rank(ttft, 0.99),
        "tpot_median_ms": statistics.median(tpot),
        "tpot_p99_ms": _nearest_rank(tpot, 0.99),
        "goodput_requests": good,
        "goodput_fraction": good / len(samples),
    }


def _comparison(
    candidate: dict[tuple[int, str], dict[str, Any]],
    baseline: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    _require(set(candidate) == set(baseline), "paired comparison keys differ")
    keys = sorted(candidate)
    candidate_values = [candidate[key] for key in keys]
    baseline_values = [baseline[key] for key in keys]
    candidate_metrics = _metrics(candidate_values)
    baseline_metrics = _metrics(baseline_values)
    deltas = [
        float(candidate[key]["e2e_ms"]) - float(baseline[key]["e2e_ms"])
        for key in keys
    ]
    phases = sorted({str(candidate[key]["phase"]) for key in keys})
    by_phase: dict[str, Any] = {}
    for phase in phases:
        phase_keys = [key for key in keys if candidate[key]["phase"] == phase]
        wins = sum(candidate[key]["e2e_ms"] < baseline[key]["e2e_ms"]
                   for key in phase_keys)
        by_phase[phase] = {
            "pairs": len(phase_keys),
            "paired_wins": wins,
            "paired_win_fraction": wins / len(phase_keys),
            "candidate": _metrics([candidate[key] for key in phase_keys]),
            "baseline": _metrics([baseline[key] for key in phase_keys]),
        }
    baseline_median = float(baseline_metrics["e2e_median_ms"])
    baseline_good = float(baseline_metrics["goodput_fraction"])
    wins = sum(candidate[key]["e2e_ms"] < baseline[key]["e2e_ms"]
               for key in keys)
    return {
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "e2e_median_improvement_fraction": (
            baseline_median - float(candidate_metrics["e2e_median_ms"])
        ) / baseline_median,
        "goodput_relative_improvement_fraction": (
            (float(candidate_metrics["goodput_fraction"]) - baseline_good)
            / baseline_good if baseline_good else None
        ),
        "paired_wins": wins,
        "paired_win_fraction": wins / len(keys),
        "worst_paired_e2e_regression_ms": max(deltas),
        "e2e_p99_regression_fraction": (
            float(candidate_metrics["e2e_p99_ms"])
            / float(baseline_metrics["e2e_p99_ms"]) - 1.0
        ),
        "tpot_p99_regression_fraction": (
            float(candidate_metrics["tpot_p99_ms"])
            / float(baseline_metrics["tpot_p99_ms"]) - 1.0
        ),
        "by_phase": by_phase,
    }


def _route_counts(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    phases = sorted({str(value["phase"]) for value in samples})
    for phase in phases:
        rows = [value for value in samples if value["phase"] == phase]
        local = sum(value["route"] == LOCAL_ROUTE for value in rows)
        remote = sum(value["route"] == REMOTE_ROUTE for value in rows)
        result[phase] = {"local": local, "remote": remote, "requests": len(rows)}
    return result


def _counterfactual_route_quality(
    tempo: dict[tuple[int, str], dict[str, Any]],
    local: dict[tuple[int, str], dict[str, Any]],
    remote: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route_name, route_value, selected, opposite in (
        ("local", LOCAL_ROUTE, local, remote),
        ("remote", REMOTE_ROUTE, remote, local),
    ):
        keys = sorted(key for key, value in tempo.items()
                      if value["route"] == route_value)
        selected_values = [selected[key] for key in keys]
        opposite_values = [opposite[key] for key in keys]
        wins = sum(selected[key]["e2e_ms"] < opposite[key]["e2e_ms"]
                   for key in keys)
        selected_median = statistics.median(
            float(selected[key]["e2e_ms"]) for key in keys)
        opposite_median = statistics.median(
            float(opposite[key]["e2e_ms"]) for key in keys)
        result[route_name] = {
            "selected_requests": len(keys),
            "fixed_counterfactual_wins": wins,
            "fixed_counterfactual_win_fraction": wins / len(keys),
            "selected_route_e2e_median_ms": selected_median,
            "opposite_route_e2e_median_ms": opposite_median,
            "selected_route_median_improvement_fraction": (
                opposite_median - selected_median
            ) / opposite_median,
            "selected_route_metrics": _metrics(selected_values),
            "opposite_route_metrics": _metrics(opposite_values),
        }
    return result


def _cross_fit_phase_router(
    by_arm: dict[str, dict[tuple[int, str], dict[str, Any]]],
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    output: dict[tuple[int, str], dict[str, Any]] = {}
    route_maps: dict[str, Any] = {}
    for target in (0, 1):
        calibration = 1 - target
        phases = sorted({value["phase"] for key, value in by_arm["local"].items()
                         if key[0] == calibration})
        route_maps[str(target)] = {}
        for phase in phases:
            medians = {}
            for arm in FIXED_ARMS:
                values = [
                    sample for key, sample in by_arm[arm].items()
                    if key[0] == calibration and sample["phase"] == phase
                ]
                medians[arm] = float(_metrics(values)["e2e_median_ms"])
            winner = min(FIXED_ARMS, key=lambda arm: medians[arm])
            route_maps[str(target)][phase] = {
                "calibration_replicate": calibration,
                "selected_arm": winner,
                "calibration_fixed_medians_ms": medians,
            }
            for key, sample in by_arm[winner].items():
                if key[0] == target and sample["phase"] == phase:
                    output[key] = sample
    _require(len(output) == 360, "cross-fit phase router did not cover all pairs")
    return output, route_maps


def _cross_fit_context_router(
    by_arm: dict[str, dict[tuple[int, str], dict[str, Any]]],
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    output: dict[tuple[int, str], dict[str, Any]] = {}
    route_maps: dict[str, Any] = {}
    context_fields = ("phase", "prompt_tokens", "output_tokens", "cache_state")
    for target in (0, 1):
        calibration = 1 - target
        contexts = sorted({
            tuple(value[field] for field in context_fields)
            for key, value in by_arm["local"].items()
            if key[0] == calibration
        })
        route_maps[str(target)] = []
        for context in contexts:
            medians = {}
            for arm in FIXED_ARMS:
                values = [
                    sample for key, sample in by_arm[arm].items()
                    if key[0] == calibration
                    and tuple(sample[field] for field in context_fields) == context
                ]
                _require(len(values) == 10, "cross-fit context calibration is incomplete")
                medians[arm] = float(_metrics(values)["e2e_median_ms"])
            winner = min(FIXED_ARMS, key=lambda arm: medians[arm])
            route_maps[str(target)].append({
                "calibration_replicate": calibration,
                "context": dict(zip(context_fields, context, strict=True)),
                "selected_arm": winner,
                "calibration_fixed_medians_ms": medians,
            })
            selected = [
                (key, sample) for key, sample in by_arm[winner].items()
                if key[0] == target
                and tuple(sample[field] for field in context_fields) == context
            ]
            _require(len(selected) == 10, "cross-fit target context is incomplete")
            output.update(selected)
    _require(len(output) == 360, "cross-fit context router did not cover all pairs")
    return output, route_maps


def _full_goal_gate(
    vs_fixed: dict[str, Any], vs_predictor: dict[str, Any],
    route_quality: dict[str, Any],
) -> dict[str, bool]:
    gate = {
        "vs_strongest_fixed_median_10pct": (
            vs_fixed["e2e_median_improvement_fraction"] >= 0.10),
        "vs_predictor_median_5pct": (
            vs_predictor["e2e_median_improvement_fraction"] >= 0.05),
        "vs_strongest_fixed_goodput_5pct": (
            (vs_fixed["goodput_relative_improvement_fraction"] or 0.0) >= 0.05),
        "paired_win_75pct": vs_fixed["paired_win_fraction"] >= 0.75,
        "every_phase_paired_win_60pct": all(
            value["paired_win_fraction"] >= 0.60
            for value in vs_fixed["by_phase"].values()),
        "every_phase_e2e_p99_regression_at_most_5pct": all(
            value["candidate"]["e2e_p99_ms"]
            <= 1.05 * value["baseline"]["e2e_p99_ms"]
            for value in vs_fixed["by_phase"].values()),
        "every_phase_tpot_p99_regression_at_most_5pct": all(
            value["candidate"]["tpot_p99_ms"]
            <= 1.05 * value["baseline"]["tpot_p99_ms"]
            for value in vs_fixed["by_phase"].values()),
        "worst_regression_at_most_100ms": (
            vs_fixed["worst_paired_e2e_regression_ms"] <= 100.0),
        "selected_local_median_gain_5pct": (
            route_quality["local"]["selected_route_median_improvement_fraction"]
            >= 0.05),
        "selected_remote_median_gain_5pct": (
            route_quality["remote"]["selected_route_median_improvement_fraction"]
            >= 0.05),
    }
    gate["all_pass"] = all(gate.values())
    return gate


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    result_path = args.result.resolve()
    result_root = result_path.parent.resolve()
    _require(not args.output.exists(), "refusing to overwrite analysis output")
    result = _load(result_path, "C4 node result")
    _require(result.get("schema") == NODE_SCHEMA, "node result schema differs")
    _require(result.get("blocks_completed") == 8
             and result.get("live_screen_correctness_pass") is True,
             "C4 node correctness gate did not pass")
    _require(result.get("performance_claim_allowed") is False
             and result.get("physical_switch_bottleneck_claim_allowed") is False,
             "C4 node result permits a forbidden claim")
    _require(result.get("transport") == "LMCacheConnectorV1:UCX"
             and result.get("unchanged_pd_data_plane") is True,
             "C4 node data-plane contract differs")
    raw_path = _bounded_file(result.get("raw"), root=result_root, name="client raw")
    _require(_sha256(raw_path) == result.get("raw_sha256"),
             "client raw digest differs")
    raw = _load(raw_path, "C4 client raw")
    _require(raw.get("schema") == CLIENT_SCHEMA, "client raw schema differs")
    _require(raw.get("blocks_completed") == 8
             and raw.get("stopped_after_first_invalid_block") is None
             and raw.get("live_screen_correctness_pass") is True,
             "client raw correctness gate did not pass")
    _require(raw.get("performance_claim_allowed") is False
             and raw.get("physical_switch_bottleneck_claim_allowed") is False,
             "client raw permits a forbidden claim")
    paired_gate = raw.get("paired_output_gate")
    _require(isinstance(paired_gate, dict)
             and paired_gate.get("paired_foreground_requests") == 360
             and paired_gate.get("all_four_arms_present") is True
             and paired_gate.get("failures") == [],
             "paired output gate differs")

    manifest_path = Path(str(raw["manifest"])).resolve()
    _require(manifest_path.is_file()
             and _sha256(manifest_path) == raw.get("manifest_sha256"),
             "phase manifest binding differs")
    manifest = _load(manifest_path, "phase manifest")
    measurement = manifest["measurement"]
    slo = {
        "e2e_ms": float(measurement["e2e_slo_ms"]),
        "ttft_ms": float(measurement["ttft_slo_ms"]),
        "tpot_ms": float(measurement["tpot_slo_ms"]),
    }

    artifacts = raw.get("artifacts")
    contracts = raw.get("contracts")
    _require(isinstance(artifacts, dict) and len(artifacts) == 8,
             "C4 artifact inventory differs")
    _require(isinstance(contracts, dict) and set(contracts) == set(artifacts),
             "C4 contract inventory differs")
    by_arm: dict[str, dict[tuple[int, str], dict[str, Any]]] = {
        arm: {} for arm in ARMS
    }
    child_bindings = []
    block_invariants = []
    for key in sorted(artifacts):
        block_path = _bounded_file(artifacts[key], root=result_root,
                                   name=f"block {key}")
        block = _load(block_path, f"block {key}")
        contract = block.get("c4_phase_screen_contract")
        _require(isinstance(contract, dict) and contract == contracts[key],
                 f"block {key} contract binding differs")
        _require(contract.get("schema") == BLOCK_SCHEMA,
                 f"block {key} schema differs")
        arm = str(contract["arm"])
        replicate = int(contract["replicate"])
        _require(arm in ARMS and replicate in {0, 1},
                 f"block {key} arm/replicate differs")
        required_true = (
            "all_requests_valid", "router_decisions_exact",
            "one_way_route_commit_exact", "p_only_full_source_hits_exact",
            "cold_remote_source_misses_exact",
            "credit_release_and_quiescence_exact", "preseed_outside_measurement",
            "actual_inference_background_only", "official_lmcache_connector_v1_ucx",
        )
        _require(all(contract.get(name) is True for name in required_true),
                 f"block {key} invariant differs")
        _require(contract.get("synthetic_network_background") is False
                 and contract.get("cross_endpoint_clock_subtraction_allowed") is False
                 and contract.get("child_return_code") == 0,
                 f"block {key} invalid path evidence")
        rows = block.get("requests")
        request_index = contract.get("request_index")
        _require(isinstance(rows, list) and len(rows) == 1283,
                 f"block {key} request count differs")
        _require(isinstance(request_index, dict) and len(request_index) == 1283,
                 f"block {key} request index differs")
        rows_by_id = {str(row["request_id"]): row for row in rows}
        paired = 0
        for request_id, metadata in request_index.items():
            _require(request_id in rows_by_id, f"block {key} request is missing")
            if metadata.get("pair_key") is None:
                continue
            sample = _sample(rows_by_id[request_id], metadata, arm=arm,
                             replicate=replicate, slo=slo)
            pair_key = (replicate, sample["pair_key"])
            _require(pair_key not in by_arm[arm], "duplicate foreground pair")
            by_arm[arm][pair_key] = sample
            paired += 1
        _require(paired == 180, f"block {key} foreground count differs")
        child_bindings.append({
            "key": key,
            "path": str(block_path),
            "sha256": _sha256(block_path),
            "bytes": block_path.stat().st_size,
            "arm": arm,
            "replicate": replicate,
        })
        block_invariants.append({
            "key": key, "requests": len(rows), "foreground": paired,
            "all_invariants_exact": True,
        })
    _require(all(len(by_arm[arm]) == 360 for arm in ARMS),
             "foreground arm coverage differs")
    pair_keys = set(by_arm["local"])
    _require(all(set(by_arm[arm]) == pair_keys for arm in ARMS),
             "paired foreground keys differ")
    for key in pair_keys:
        values = {
            (by_arm[arm][key]["prompt_sha256"],
             by_arm[arm][key]["output_text_sha256"],
             by_arm[arm][key]["output_token_values"])
            for arm in ARMS
        }
        _require(len(values) == 1, "paired foreground output differs")

    pooled = {arm: _metrics(list(by_arm[arm].values())) for arm in ARMS}
    strongest_fixed = min(FIXED_ARMS,
                          key=lambda arm: pooled[arm]["e2e_median_ms"])
    tempo_vs_fixed = _comparison(by_arm["tempo"], by_arm[strongest_fixed])
    tempo_vs_predictor = _comparison(by_arm["tempo"], by_arm["predictor"])
    tempo_route_quality = _counterfactual_route_quality(
        by_arm["tempo"], by_arm["local"], by_arm["remote"])
    cross_fit, route_maps = _cross_fit_phase_router(by_arm)
    cross_fit_vs_fixed = _comparison(cross_fit, by_arm[strongest_fixed])
    cross_fit_vs_predictor = _comparison(cross_fit, by_arm["predictor"])
    cross_fit_route_quality = _counterfactual_route_quality(
        cross_fit, by_arm["local"], by_arm["remote"])
    context_fit, context_route_maps = _cross_fit_context_router(by_arm)
    context_fit_vs_fixed = _comparison(context_fit, by_arm[strongest_fixed])
    context_fit_vs_predictor = _comparison(context_fit, by_arm["predictor"])
    context_fit_route_quality = _counterfactual_route_quality(
        context_fit, by_arm["local"], by_arm["remote"])
    phase_metrics = {
        phase: {
            arm: _metrics([
                sample for sample in by_arm[arm].values()
                if sample["phase"] == phase
            ]) for arm in ARMS
        }
        for phase in manifest["phase_order"]
    }

    original_gate = _full_goal_gate(
        tempo_vs_fixed, tempo_vs_predictor, tempo_route_quality)
    cross_fit_headline = {
        "vs_strongest_fixed_median_10pct": (
            cross_fit_vs_fixed["e2e_median_improvement_fraction"] >= 0.10),
        "vs_predictor_median_5pct": (
            cross_fit_vs_predictor["e2e_median_improvement_fraction"] >= 0.05),
        "vs_strongest_fixed_goodput_5pct": (
            (cross_fit_vs_fixed["goodput_relative_improvement_fraction"] or 0.0)
            >= 0.05),
    }
    cross_fit_headline["all_pass"] = all(cross_fit_headline.values())
    context_fit_headline = {
        "vs_strongest_fixed_median_10pct": (
            context_fit_vs_fixed["e2e_median_improvement_fraction"] >= 0.10),
        "vs_predictor_median_5pct": (
            context_fit_vs_predictor["e2e_median_improvement_fraction"] >= 0.05),
        "vs_strongest_fixed_goodput_5pct": (
            (context_fit_vs_fixed["goodput_relative_improvement_fraction"] or 0.0)
            >= 0.05),
    }
    context_fit_headline["all_pass"] = all(context_fit_headline.values())

    payload = {
        "schema": SCHEMA,
        "source_result": str(result_path),
        "source_result_sha256": _sha256(result_path),
        "source_raw": str(raw_path),
        "source_raw_sha256": _sha256(raw_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "slurm_job_id": result.get("slurm_job_id"),
        "child_artifact_bindings": child_bindings,
        "block_invariants": block_invariants,
        "requests_per_block": 1283,
        "requests_total": 8 * 1283,
        "paired_foreground_requests": 360,
        "pooled_arm_metrics": pooled,
        "phase_arm_metrics": phase_metrics,
        "strongest_fixed_arm": strongest_fixed,
        "tempo_vs_strongest_fixed": tempo_vs_fixed,
        "tempo_vs_predictor": tempo_vs_predictor,
        "tempo_route_counts_by_phase": _route_counts(
            list(by_arm["tempo"].values())),
        "tempo_selected_route_counterfactual": tempo_route_quality,
        "cross_fit_phase_router": {
            "diagnostic_only": True,
            "route_maps": route_maps,
            "pooled_metrics": _metrics(list(cross_fit.values())),
            "vs_strongest_fixed": cross_fit_vs_fixed,
            "vs_predictor": cross_fit_vs_predictor,
            "selected_route_counterfactual": cross_fit_route_quality,
            "headline_screen": cross_fit_headline,
            "full_goal_gate": _full_goal_gate(
                cross_fit_vs_fixed, cross_fit_vs_predictor,
                cross_fit_route_quality),
        },
        "cross_fit_phase_geometry_router": {
            "diagnostic_only": True,
            "context_fields": [
                "phase", "prompt_tokens", "output_tokens", "cache_state"],
            "route_maps": context_route_maps,
            "pooled_metrics": _metrics(list(context_fit.values())),
            "vs_strongest_fixed": context_fit_vs_fixed,
            "vs_predictor": context_fit_vs_predictor,
            "selected_route_counterfactual": context_fit_route_quality,
            "headline_screen": context_fit_headline,
            "full_goal_gate": _full_goal_gate(
                context_fit_vs_fixed, context_fit_vs_predictor,
                context_fit_route_quality),
        },
        "original_goal_gate_on_live_tempo": original_gate,
        "live_screen_correctness_pass": True,
        "controller_performance_gate_pass": original_gate["all_pass"],
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "unchanged_pd_data_plane": True,
        "transport": "LMCacheConnectorV1:UCX",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "strongest_fixed": strongest_fixed,
        "tempo_gate_pass": original_gate["all_pass"],
        "cross_fit_phase_headline_pass": cross_fit_headline["all_pass"],
        "cross_fit_context_headline_pass": context_fit_headline["all_pass"],
        "cross_fit_context_full_goal_pass": payload[
            "cross_fit_phase_geometry_router"]["full_goal_gate"]["all_pass"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
