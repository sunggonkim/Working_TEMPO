#!/usr/bin/env python3
"""Replay C4 paired counterfactuals through the calibrated endpoint controller."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as analyzer
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as screen
from eval.sota_4node import build_tempo_pd_c4_calibrated_profiles as profiles
from tempo.pd_contention_workload import CacheState
from tempo.pd_elastic_controller_v443 import CacheResidency
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_controller import (
    EndpointFeedbackController,
    EndpointRequest,
    EndpointRoute,
    EndpointWork,
)
from tempo.pd_endpoint_profile import load_endpoint_service_profile


SCHEMA = "tempo-pd-c4-calibrated-controller-replay-v1"
MIN_MEAN_GAIN_VS_STRONGEST_FIXED = 0.03
MIN_MEAN_GAIN_VS_PREDICTOR = 0.02
MAX_P99_REGRESSION = 0.05
MIN_PAIRED_WIN_FRACTION = 0.55

_STATE_TO_RESIDENCY = {
    CacheState.MISS: CacheResidency.MISS,
    CacheState.P_ONLY: CacheResidency.P_ONLY,
    CacheState.D_ONLY: CacheResidency.D_ONLY,
    CacheState.BOTH: CacheResidency.BOTH,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_fingerprint(value: Mapping[str, object]) -> str:
    """Return the canonical fingerprint of a replay artifact."""

    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be lowercase SHA-256",
    )
    return value


def _load_bound_object(
    path: Path, expected_sha256: str, *, name: str,
) -> dict[str, object]:
    path = path.resolve()
    expected_sha256 = _canonical_sha(expected_sha256, name=f"{name} SHA-256")
    _require(path.is_file(), f"{name} is missing")
    _require(_sha256(path) == expected_sha256, f"{name} digest differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {name}") from exc
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _nearest_rank(values: list[float], fraction: float) -> float:
    _require(bool(values), "latency summary is empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summary(values: list[float], *, deadline_ms: float) -> dict[str, object]:
    _require(bool(values), "latency summary is empty")
    return {
        "count": len(values),
        "mean_e2e_ms": statistics.mean(values),
        "median_e2e_ms": statistics.median(values),
        "p99_e2e_ms": _nearest_rank(values, 0.99),
        "goodput_fraction": (
            sum(value <= deadline_ms for value in values) / len(values)),
    }


def _gain(candidate: float, baseline: float) -> float:
    _require(baseline > 0.0, "baseline must be positive")
    return (baseline - candidate) / baseline


def _validate_inputs(
    *, analysis_path: Path, analysis_sha256: str,
    manifest_path: Path, manifest_sha256: str,
    elastic_path: Path, elastic_sha256: str,
    endpoint_path: Path, endpoint_sha256: str,
    receipt_path: Path, receipt_sha256: str,
):
    analysis = _load_bound_object(
        analysis_path, analysis_sha256, name="C4 analysis")
    _require(
        analysis.get("schema") == analyzer.SCHEMA
        and analysis.get("fingerprint_sha256")
        == analyzer._analysis_fingerprint(analysis)
        and analysis.get("authorizes_profile_fit") is True
        and analysis.get("performance_claim_allowed") is False,
        "C4 replay analysis contract differs",
    )
    source = analysis.get("source_node_result")
    _require(isinstance(source, dict) and set(source) == {"path", "sha256"},
             "C4 replay analysis source binding differs")
    _require(
        analyzer.analyze(
            Path(str(source["path"])),
            expected_result_sha256=source["sha256"],
        ) == analysis,
        "C4 replay analysis does not reproduce",
    )

    manifest = _load_bound_object(
        manifest_path, manifest_sha256, name="adaptive workload manifest")
    _require(
        manifest.get("schema") == profiles.LIVE_MANIFEST_SCHEMA
        and manifest.get("fingerprint_sha256")
        == screen.manifest_fingerprint(manifest)
        and manifest.get("calibration_analysis") == {
            "path": str(analysis_path.resolve()),
            "sha256": analysis_sha256,
            "fingerprint_sha256": analysis["fingerprint_sha256"],
        }
        and manifest.get("profile_fit_formula") == profiles.FORMULA_ID
        and manifest.get("performance_claim_allowed") is False,
        "adaptive workload replay manifest differs",
    )

    receipt = _load_bound_object(
        receipt_path, receipt_sha256, name="C4 profile receipt")
    _require(
        receipt.get("schema") == profiles.SCHEMA
        and receipt.get("fingerprint_sha256")
        == profiles._receipt_fingerprint(receipt)
        and receipt.get("formula_id") == profiles.FORMULA_ID
        and receipt.get("source_analysis") == {
            "path": str(analysis_path.resolve()),
            "sha256": analysis_sha256,
            "fingerprint_sha256": analysis["fingerprint_sha256"],
        }
        and receipt.get("workload_manifest") == {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
        }
        and receipt.get("performance_claim_allowed") is False,
        "C4 profile receipt binding differs",
    )

    _canonical_sha(elastic_sha256, name="Elastic profile SHA-256")
    _canonical_sha(endpoint_sha256, name="endpoint profile SHA-256")
    _require(_sha256(elastic_path.resolve()) == elastic_sha256,
             "calibrated Elastic profile digest differs")
    _require(_sha256(endpoint_path.resolve()) == endpoint_sha256,
             "calibrated endpoint profile digest differs")
    elastic = load_elastic_profile(elastic_path.resolve())
    endpoint = load_endpoint_service_profile(endpoint_path.resolve())
    _require(
        receipt["elastic_profile"]["fingerprint_sha256"]
        == elastic.fingerprint_sha256
        and receipt["endpoint_profile"]["fingerprint_sha256"]
        == endpoint.fingerprint_sha256
        and endpoint.elastic_profile_fingerprint_sha256
        == elastic.fingerprint_sha256
        and endpoint.workload_manifest_sha256 == manifest_sha256
        and len(elastic.rows) == len(endpoint.rows) == 6,
        "C4 calibrated profile/replay binding differs",
    )
    return analysis, manifest, elastic, endpoint, receipt


def _predictor_route(sample: Mapping[str, object], elastic_row) -> EndpointRoute:
    state = CacheState(sample["cache_state"])
    if state in {CacheState.D_ONLY, CacheState.BOTH}:
        return EndpointRoute.LOCAL
    return (
        EndpointRoute.LOCAL
        if elastic_row.local_upper_bound_ms <= elastic_row.remote_upper_bound_ms
        else EndpointRoute.REMOTE
    )


def _endpoint_request(
    *, request_id: str, sample: Mapping[str, object], elastic_row,
    service_row, remaining_deadline_ms: float,
) -> EndpointRequest:
    state = CacheState(sample["cache_state"])
    return EndpointRequest(
        request_id=request_id,
        local_e2e_prior_ms=elastic_row.local_upper_bound_ms,
        remote_e2e_prior_ms=elastic_row.remote_upper_bound_ms,
        local_ttft_prior_ms=service_row.local_ttft_prior_ms,
        remote_ttft_prior_ms=service_row.remote_ttft_prior_ms,
        uncertainty_ms=elastic_row.uncertainty_ms,
        e2e_deadline_ms=remaining_deadline_ms,
        work=EndpointWork(
            local_token_ms=service_row.local_token_ms,
            remote_prefill_token_ms=service_row.remote_prefill_token_ms,
            remote_kv_bytes=elastic_row.remote_kv_bytes,
            remote_semantic_ops=1,
        ),
        local_allowed=True,
        remote_allowed=state not in {CacheState.D_ONLY, CacheState.BOTH},
    )


def _replay_replicate(
    samples: list[Mapping[str, object]], *, elastic, endpoint,
    deadline_ms: float, replicate: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    ordered = sorted(samples, key=lambda row: (
        float(row["arrival_offset_ms"]), int(row["ordinal"]), row["pair_key"]))
    controller = EndpointFeedbackController(endpoint.controller)
    arrivals = [
        (max(1, round(float(row["arrival_offset_ms"]) * 1_000_000)), row)
        for row in ordered
    ]
    arrival_index = 0
    pending: list[tuple[int, Mapping[str, object]]] = []
    completions: list[tuple[int, int, str, float]] = []
    completion_order = 0
    records: list[dict[str, object]] = []
    attempts: dict[str, int] = {}
    now_ns = 1

    def release_at(timestamp_ns: int) -> None:
        nonlocal now_ns
        now_ns = timestamp_ns
        while completions and completions[0][0] <= timestamp_ns:
            completed_ns, _order, request_id, observed_ttft_ms = heapq.heappop(
                completions)
            controller.observe_first_response(
                request_id,
                observed_ttft_ms=observed_ttft_ms,
                now_ns=completed_ns,
            )

    def drain() -> None:
        nonlocal completion_order
        while pending:
            arrival_ns, sample = pending[0]
            request_id = f"replay-r{replicate}-{sample['pair_key']}"
            waited_ms = (now_ns - arrival_ns) / 1_000_000.0
            remaining = deadline_ms - waited_ms
            _require(remaining > 0.0,
                     f"offline replay request expired in queue: {request_id}")
            prompt_tokens = int(sample["prompt_tokens"])
            output_tokens = int(sample["output_tokens"])
            state = CacheState(sample["cache_state"])
            residency = _STATE_TO_RESIDENCY[state]
            elastic_row = elastic.exact_row(prompt_tokens, output_tokens)
            _require(elastic_row is not None,
                     "offline replay lacks an Elastic profile row")
            service = endpoint.exact_row(prompt_tokens, output_tokens, residency)
            request = _endpoint_request(
                request_id=request_id,
                sample=sample,
                elastic_row=elastic_row,
                service_row=service,
                remaining_deadline_ms=remaining,
            )
            decision = controller.submit(request, now_ns=now_ns)
            attempts[request_id] = attempts.get(request_id, 0) + 1
            if decision.route is EndpointRoute.QUEUE:
                return
            pending.pop(0)
            chosen = sample[
                "local" if decision.route is EndpointRoute.LOCAL else "remote"]
            observed_ttft = float(chosen["ttft_ms"])
            completion_order += 1
            heapq.heappush(completions, (
                now_ns + max(1, math.ceil(observed_ttft * 1_000_000)),
                completion_order,
                request_id,
                observed_ttft,
            ))
            predictor_route = _predictor_route(sample, elastic_row)
            predictor_value = sample[
                "local" if predictor_route is EndpointRoute.LOCAL else "remote"]
            records.append({
                "request_id": request_id,
                "pair_key": sample["pair_key"],
                "replicate": replicate,
                "phase": sample["phase"],
                "arrival_offset_ms": sample["arrival_offset_ms"],
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "cache_state": state.value,
                "route": decision.route.value,
                "reason": decision.reason,
                "probe": decision.probe,
                "local_multiplier": decision.local_multiplier,
                "remote_multiplier": decision.remote_multiplier,
                "queue_attempts": attempts[request_id] - 1,
                "queue_wait_ms": waited_ms,
                "tempo_e2e_ms": waited_ms + float(chosen["e2e_ms"]),
                "fixed_local_e2e_ms": float(sample["local"]["e2e_ms"]),
                "fixed_remote_e2e_ms": float(sample["remote"]["e2e_ms"]),
                "predictor_route": predictor_route.value,
                "predictor_e2e_ms": float(predictor_value["e2e_ms"]),
                "oracle_e2e_ms": min(
                    float(sample["local"]["e2e_ms"]),
                    float(sample["remote"]["e2e_ms"]),
                ),
            })

    while arrival_index < len(arrivals) or completions or pending:
        next_arrival = (
            arrivals[arrival_index][0]
            if arrival_index < len(arrivals) else None)
        next_completion = completions[0][0] if completions else None
        candidates = [value for value in (next_arrival, next_completion)
                      if value is not None]
        _require(bool(candidates), "offline replay queue deadlocked")
        event_ns = min(candidates)
        release_at(event_ns)
        while (
            arrival_index < len(arrivals)
            and arrivals[arrival_index][0] <= event_ns
        ):
            pending.append(arrivals[arrival_index])
            arrival_index += 1
        before = len(pending)
        drain()
        if pending and not completions and arrival_index >= len(arrivals):
            _require(len(pending) < before,
                     "offline replay has no completion capable of draining queue")

    final_ns = max(now_ns, max((item[0] for item in completions), default=now_ns))
    while completions:
        release_at(completions[0][0])
    snapshot = controller.snapshot(now_ns=final_ns)
    _require(all(value == 0 for value in snapshot["resources"].values())
             and snapshot["inflight"] == 0,
             "offline replay leaked endpoint admission resources")
    _require(len(records) == len(samples),
             "offline replay did not admit every paired request")
    return records, snapshot


def _phase_summaries(
    records: list[Mapping[str, object]], *, deadline_ms: float,
) -> list[dict[str, object]]:
    phases = [phase.value for phase in analyzer.manifest_builder.PHASES]
    result = []
    for phase in phases:
        rows = [row for row in records if row["phase"] == phase]
        _require(bool(rows), f"offline replay phase is empty: {phase}")
        result.append({
            "phase": phase,
            "requests": len(rows),
            "route_counts": {
                EndpointRoute.LOCAL.value: sum(
                    row["route"] == EndpointRoute.LOCAL.value for row in rows),
                EndpointRoute.REMOTE.value: sum(
                    row["route"] == EndpointRoute.REMOTE.value for row in rows),
            },
            "tempo": _summary(
                [float(row["tempo_e2e_ms"]) for row in rows],
                deadline_ms=deadline_ms),
            "fixed_local": _summary(
                [float(row["fixed_local_e2e_ms"]) for row in rows],
                deadline_ms=deadline_ms),
            "fixed_remote": _summary(
                [float(row["fixed_remote_e2e_ms"]) for row in rows],
                deadline_ms=deadline_ms),
            "predictor": _summary(
                [float(row["predictor_e2e_ms"]) for row in rows],
                deadline_ms=deadline_ms),
        })
    return result


def replay(
    *, analysis_path: Path, analysis_sha256: str,
    manifest_path: Path, manifest_sha256: str,
    elastic_path: Path, elastic_sha256: str,
    endpoint_path: Path, endpoint_sha256: str,
    receipt_path: Path, receipt_sha256: str,
) -> dict[str, object]:
    paths = [analysis_path, manifest_path, elastic_path, endpoint_path, receipt_path]
    paths = [path.resolve() for path in paths]
    analysis_path, manifest_path, elastic_path, endpoint_path, receipt_path = paths
    analysis, manifest, elastic, endpoint, receipt = _validate_inputs(
        analysis_path=analysis_path,
        analysis_sha256=analysis_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        elastic_path=elastic_path,
        elastic_sha256=elastic_sha256,
        endpoint_path=endpoint_path,
        endpoint_sha256=endpoint_sha256,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
    )
    del receipt
    samples = analysis.get("foreground_paired_samples")
    _require(isinstance(samples, list) and bool(samples),
             "offline replay paired samples are missing")
    by_replicate = {
        replicate: [row for row in samples if row["replicate"] == replicate]
        for replicate in (0, 1)
    }
    _require(all(by_replicate.values())
             and sum(map(len, by_replicate.values())) == len(samples),
             "offline replay replicate inventory differs")
    deadline_ms = float(manifest["measurement"]["e2e_slo_ms"])
    records = []
    final_snapshots = []
    for replicate in (0, 1):
        replicate_records, snapshot = _replay_replicate(
            by_replicate[replicate],
            elastic=elastic,
            endpoint=endpoint,
            deadline_ms=deadline_ms,
            replicate=replicate,
        )
        records.extend(replicate_records)
        final_snapshots.append({"replicate": replicate, "controller": snapshot})

    tempo = _summary(
        [float(row["tempo_e2e_ms"]) for row in records],
        deadline_ms=deadline_ms)
    fixed_local = _summary(
        [float(row["fixed_local_e2e_ms"]) for row in records],
        deadline_ms=deadline_ms)
    fixed_remote = _summary(
        [float(row["fixed_remote_e2e_ms"]) for row in records],
        deadline_ms=deadline_ms)
    predictor = _summary(
        [float(row["predictor_e2e_ms"]) for row in records],
        deadline_ms=deadline_ms)
    oracle = _summary(
        [float(row["oracle_e2e_ms"]) for row in records],
        deadline_ms=deadline_ms)
    strongest_name, strongest = min(
        (("fixed_local", fixed_local), ("fixed_remote", fixed_remote)),
        key=lambda item: item[1]["mean_e2e_ms"],
    )
    mean_gain_fixed = _gain(
        float(tempo["mean_e2e_ms"]), float(strongest["mean_e2e_ms"]))
    mean_gain_predictor = _gain(
        float(tempo["mean_e2e_ms"]), float(predictor["mean_e2e_ms"]))
    p99_regression = -_gain(
        float(tempo["p99_e2e_ms"]), float(strongest["p99_e2e_ms"]))
    strongest_field = strongest_name + "_e2e_ms"
    paired_win_fraction = sum(
        float(row["tempo_e2e_ms"]) < float(row[strongest_field])
        for row in records
    ) / len(records)
    route_counts = {
        EndpointRoute.LOCAL.value: sum(
            row["route"] == EndpointRoute.LOCAL.value for row in records),
        EndpointRoute.REMOTE.value: sum(
            row["route"] == EndpointRoute.REMOTE.value for row in records),
    }
    queue_fraction = sum(int(row["queue_attempts"]) > 0 for row in records) / len(records)
    gates = {
        "all_requests_replayed": len(records) == len(samples),
        "all_resources_released": all(
            all(value == 0 for value in item["controller"]["resources"].values())
            for item in final_snapshots),
        "both_routes_exercised": all(value > 0 for value in route_counts.values()),
        "mean_gain_vs_strongest_fixed_at_least_3pct": (
            mean_gain_fixed >= MIN_MEAN_GAIN_VS_STRONGEST_FIXED),
        "mean_gain_vs_predictor_at_least_2pct": (
            mean_gain_predictor >= MIN_MEAN_GAIN_VS_PREDICTOR),
        "goodput_not_below_strongest_fixed": (
            tempo["goodput_fraction"] >= strongest["goodput_fraction"]),
        "p99_regression_at_most_5pct": p99_regression <= MAX_P99_REGRESSION,
        "paired_win_fraction_at_least_55pct": (
            paired_win_fraction >= MIN_PAIRED_WIN_FRACTION),
    }
    live_authorized = all(gates.values())
    output: dict[str, object] = {
        "schema": SCHEMA,
        "analysis": {"path": str(analysis_path), "sha256": analysis_sha256},
        "workload_manifest": {
            "path": str(manifest_path), "sha256": manifest_sha256},
        "elastic_profile": {
            "path": str(elastic_path),
            "sha256": elastic_sha256,
            "fingerprint_sha256": elastic.fingerprint_sha256,
        },
        "endpoint_profile": {
            "path": str(endpoint_path),
            "sha256": endpoint_sha256,
            "fingerprint_sha256": endpoint.fingerprint_sha256,
        },
        "profile_receipt": {
            "path": str(receipt_path), "sha256": receipt_sha256},
        "paired_requests": len(records),
        "route_counts": route_counts,
        "queue_retry_fraction": queue_fraction,
        "pooled": {
            "tempo": tempo,
            "fixed_local": fixed_local,
            "fixed_remote": fixed_remote,
            "strongest_fixed_name_calibration_only": strongest_name,
            "predictor": predictor,
            "per_request_oracle": oracle,
            "mean_gain_vs_strongest_fixed": mean_gain_fixed,
            "mean_gain_vs_predictor": mean_gain_predictor,
            "p99_regression_vs_strongest_fixed": p99_regression,
            "paired_win_fraction_vs_strongest_fixed": paired_win_fraction,
        },
        "phase_summaries": _phase_summaries(records, deadline_ms=deadline_ms),
        "controller_final": final_snapshots,
        "decisions": records,
        "screen_gates": gates,
        "live_adaptive_screen_authorized": live_authorized,
        "calibration_only": True,
        "strongest_fixed_selection_authoritative": False,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "independent_validation_required": True,
    }
    output["fingerprint_sha256"] = replay_fingerprint(output)
    return output


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis", "manifest", "elastic", "endpoint", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    _require(not args.output.exists(), "refusing to overwrite C4 offline replay")
    value = replay(
        analysis_path=args.analysis,
        analysis_sha256=args.analysis_sha256,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        elastic_path=args.elastic,
        elastic_sha256=args.elastic_sha256,
        endpoint_path=args.endpoint,
        endpoint_sha256=args.endpoint_sha256,
        receipt_path=args.receipt,
        receipt_sha256=args.receipt_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "live_adaptive_screen_authorized": value[
            "live_adaptive_screen_authorized"],
        "fingerprint_sha256": value["fingerprint_sha256"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
