#!/usr/bin/env python3
"""Fail-closed analysis of fixed-local, LMCache-remote, and TEMPO-PD runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from eval.sota_4node.run_tempo_pd_stream_metrics_v1 import SCHEMA as RAW_SCHEMA


ANALYSIS_SCHEMA = "tempo-pd-performance-analysis-1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing explicit raw artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _number(value: Any, field: str) -> float:
    _require(type(value) in (int, float), f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result) and result >= 0, f"{field} must be nonnegative")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    _require(bool(values), "percentile requires samples")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _distribution(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "distribution requires samples")
    return {
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _parse_run(
    label: str,
    raw: Mapping[str, Any],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    e2e_slo_ms: float | None,
) -> dict[str, Any]:
    _require(raw.get("schema") == RAW_SCHEMA, f"{label}: raw schema mismatch")
    _require(raw.get("validation", {}).get("performance_claim_allowed") is True,
             f"{label}: raw validation failed")
    run = raw.get("run")
    model = raw.get("model")
    workload = raw.get("workload")
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(all(isinstance(value, dict) for value in (run, model, workload)),
             f"{label}: metadata missing")
    _require(isinstance(requests, list) and requests, f"{label}: requests missing")
    _require(isinstance(decisions, list), f"{label}: decisions missing")
    by_decision = {row.get("request_id"): row for row in decisions}
    _require(len(by_decision) == len(decisions), f"{label}: duplicate decisions")
    mode = run.get("mode")
    metrics: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    contracts: dict[str, tuple[Any, ...]] = {}
    reasons: dict[str, int] = {}
    routes: dict[str, int] = {}
    for index, record in enumerate(requests):
        request_id = record.get("request_id")
        _require(isinstance(request_id, str) and request_id not in outputs,
                 f"{label}: invalid request ID")
        _require(record.get("valid") is True, f"{label}:{request_id}: invalid stream")
        decision = by_decision.get(request_id)
        router = record.get("router")
        _require(isinstance(decision, dict) and isinstance(router, dict),
                 f"{label}:{request_id}: route evidence missing")
        _require(router.get("route") == decision.get("route")
                 and router.get("reason") == decision.get("reason")
                 and router.get("workload_fingerprint") == decision.get("workload_fingerprint"),
                 f"{label}:{request_id}: header/decision mismatch")
        _require(decision.get("phase") == "complete" and decision.get("error") is None,
                 f"{label}:{request_id}: route did not complete")
        arrivals = record.get("token_arrival_offsets_ns")
        dispatch = record.get("dispatch_offset_ns")
        _require(isinstance(arrivals, list) and len(arrivals) >= 2
                 and all(type(value) is int for value in arrivals)
                 and type(dispatch) is int,
                 f"{label}:{request_id}: timing contract invalid")
        itl = [(right - left) / 1_000_000.0
               for left, right in zip(arrivals, arrivals[1:])]
        ttft = (arrivals[0] - dispatch) / 1_000_000.0
        e2e = (arrivals[-1] - dispatch) / 1_000_000.0
        tpot = (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1_000_000.0
        slo_pass = (
            ttft <= ttft_slo_ms
            and tpot <= tpot_slo_ms
            and (e2e_slo_ms is None or e2e <= e2e_slo_ms)
        )
        metrics.append({
            "request_id": request_id,
            "dispatch_offset_ns": dispatch,
            "last_token_offset_ns": arrivals[-1],
            "completion_tokens": len(arrivals),
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "itl_ms": itl,
            "e2e_ms": e2e,
            "slo_pass": slo_pass,
            "route": decision["route"],
            "reason": decision["reason"],
            "potential_kv_bytes": decision["potential_kv_bytes"],
            "workload_fingerprint": decision["workload_fingerprint"],
        })
        outputs[request_id] = record["output_text_sha256"]
        contracts[request_id] = (
            record["prompt_sha256"], record["requested_max_tokens"],
            record["scheduled_dispatch_offset_ns"],
        )
        routes[decision["route"]] = routes.get(decision["route"], 0) + 1
        reasons[decision["reason"]] = reasons.get(decision["reason"], 0) + 1
    _require(set(outputs) == set(by_decision), f"{label}: decision coverage mismatch")
    if mode == "fixed_local":
        _require(set(routes) == {"decoder_local_recompute_or_cache"},
                 f"{label}: fixed-local route mismatch")
    if mode == "lmcache_always_remote":
        _require(set(routes) == {"remote_prefill_live_kv"},
                 f"{label}: fixed-remote route mismatch")
    first = min(row["dispatch_offset_ns"] for row in metrics)
    last = max(row["last_token_offset_ns"] for row in metrics)
    window_s = (last - first) / 1_000_000_000.0
    _require(window_s > 0, f"{label}: nonpositive measurement window")
    passed = [row for row in metrics if row["slo_pass"]]
    tokens = sum(row["completion_tokens"] for row in metrics)
    passed_tokens = sum(row["completion_tokens"] for row in passed)
    return {
        "label": label,
        "mode": mode,
        "model_config_sha256": model.get("config_sha256"),
        "workload_sha256": workload.get("sha256"),
        "request_count": len(metrics),
        "routes": routes,
        "reasons": reasons,
        "performance": {
            "measurement_window_s": window_s,
            "request_throughput_per_s": len(metrics) / window_s,
            "output_token_throughput_per_s": tokens / window_s,
            "ttft_ms": _distribution([row["ttft_ms"] for row in metrics]),
            "tpot_ms": _distribution([row["tpot_ms"] for row in metrics]),
            "itl_ms": _distribution([value for row in metrics for value in row["itl_ms"]]),
            "e2e_ms": _distribution([row["e2e_ms"] for row in metrics]),
            "slo_goodput": {
                "successful_requests": len(passed),
                "success_fraction": len(passed) / len(metrics),
                "request_goodput_per_s": len(passed) / window_s,
                "output_token_goodput_per_s": passed_tokens / window_s,
            },
        },
        "request_metrics": metrics,
        "_outputs": outputs,
        "_contracts": contracts,
    }


def _paired(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    left = {row["request_id"]: row for row in candidate["request_metrics"]}
    right = {row["request_id"]: row for row in baseline["request_metrics"]}
    _require(set(left) == set(right), "paired request IDs differ")
    rows = []
    for request_id in sorted(left):
        rows.append({
            "request_id": request_id,
            "route": left[request_id]["route"],
            "ttft_delta_ms": left[request_id]["ttft_ms"] - right[request_id]["ttft_ms"],
            "tpot_delta_ms": left[request_id]["tpot_ms"] - right[request_id]["tpot_ms"],
            "e2e_delta_ms": left[request_id]["e2e_ms"] - right[request_id]["e2e_ms"],
        })
    e2e = [row["e2e_delta_ms"] for row in rows]
    return {
        "pairs": rows,
        "e2e_win_count": sum(value < 0 for value in e2e),
        "e2e_delta_median_ms": statistics.median(e2e),
        "request_goodput_delta_per_s": (
            candidate["performance"]["slo_goodput"]["request_goodput_per_s"]
            - baseline["performance"]["slo_goodput"]["request_goodput_per_s"]
        ),
        "output_token_goodput_delta_per_s": (
            candidate["performance"]["slo_goodput"]["output_token_goodput_per_s"]
            - baseline["performance"]["slo_goodput"]["output_token_goodput_per_s"]
        ),
    }


def analyze(
    runs: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    e2e_slo_ms: float | None,
) -> dict[str, Any]:
    _require(len(runs) == 3, "exactly three runs are required")
    parsed = [_parse_run(
        label, raw,
        ttft_slo_ms=_number(ttft_slo_ms, "ttft_slo_ms"),
        tpot_slo_ms=_number(tpot_slo_ms, "tpot_slo_ms"),
        e2e_slo_ms=None if e2e_slo_ms is None else _number(e2e_slo_ms, "e2e_slo_ms"),
    ) for label, raw in runs]
    by_mode = {row["mode"]: row for row in parsed}
    _require(set(by_mode) == {"fixed_local", "lmcache_always_remote", "tempo_auto"},
             "runs must contain the three exact modes")
    first = parsed[0]
    same_model = all(row["model_config_sha256"] == first["model_config_sha256"]
                     for row in parsed)
    same_workload = all(row["workload_sha256"] == first["workload_sha256"]
                        and row["_contracts"] == first["_contracts"] for row in parsed)
    same_outputs = all(row["_outputs"] == first["_outputs"] for row in parsed)
    correctness = same_model and same_workload and same_outputs
    public = [{key: value for key, value in row.items() if not key.startswith("_")}
              for row in parsed]
    tempo = by_mode["tempo_auto"]
    remote = by_mode["lmcache_always_remote"]
    local = by_mode["fixed_local"]
    tempo_vs_remote = _paired(tempo, remote) if correctness else None
    tempo_vs_local = _paired(tempo, local) if correctness else None
    return {
        "schema": ANALYSIS_SCHEMA,
        "evidence": "actual_vllm_pd_router_performance_comparison",
        "slo": {
            "ttft_ms": ttft_slo_ms,
            "tpot_ms": tpot_slo_ms,
            "e2e_ms": e2e_slo_ms,
        },
        "runs": public,
        "correctness": {
            "same_model": same_model,
            "same_workload_and_schedule": same_workload,
            "exact_output_text_equivalence": same_outputs,
            "correctness_met": correctness,
        },
        "comparisons": {
            "tempo_vs_official_lmcache_remote": tempo_vs_remote,
            "tempo_vs_fixed_local": tempo_vs_local,
        },
        "route_evidence": {
            "tempo_routes": tempo["routes"],
            "remote_branch_observed": "remote_prefill_live_kv" in tempo["routes"],
            "local_branch_observed": "decoder_local_recompute_or_cache" in tempo["routes"],
        },
        "comparison_claim_allowed": correctness,
        "claim_boundary": (
            "This measures admission plus official LMCache remote P/D; it does not claim "
            "a faster KV transport. Promotion requires independent replication."
        ),
    }


def _run_arg(value: str) -> tuple[str, Path]:
    label, separator, raw = value.partition("=")
    if not (separator and label and raw):
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    return label, Path(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_run_arg, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    parser.add_argument("--e2e-slo-ms", type=float)
    args = parser.parse_args()
    report = analyze(
        [(label, _load(path)) for label, path in args.run],
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        e2e_slo_ms=args.e2e_slo_ms,
    )
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    return 0 if report["comparison_claim_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
