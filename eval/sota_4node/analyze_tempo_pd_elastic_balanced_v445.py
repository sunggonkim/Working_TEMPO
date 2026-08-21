#!/usr/bin/env python3
"""Analyze one-live-epoch four-arm Elastic-PD crossover evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


ARMS = ("local", "remote", "predictor", "tempo")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metrics(row):
    arrivals = row["token_arrival_offsets_ns"]
    dispatch = row["dispatch_offset_ns"]
    intervals = [(right - left) / 1_000_000
                 for left, right in zip(arrivals, arrivals[1:])]
    return {
        "ttft_ms": (arrivals[0] - dispatch) / 1_000_000,
        "e2e_ms": (row["stream_end_offset_ns"] - dispatch) / 1_000_000,
        "tpot_ms": statistics.median(intervals) if intervals else 0.0,
        "tpot_max_ms": max(intervals) if intervals else 0.0,
    }


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))]


def analyze(stage_root: Path):
    public = json.loads((stage_root / "raw.json").read_text())
    orchestration = public.get("elastic_balanced_orchestration")
    _require(isinstance(orchestration, dict), "elastic orchestration missing")
    _require(orchestration.get("one_live_server_epoch") is True,
             "one live server epoch required")
    artifacts = orchestration.get("artifacts")
    _require(isinstance(artifacts, dict) and len(artifacts) == 8,
             "exactly eight measured arm artifacts required")
    by_arm = defaultdict(list)
    fingerprints = set()
    profile_ids = set()
    rows_by_key = {}
    route_counts = defaultdict(int)
    for key, raw_path in artifacts.items():
        artifact = json.loads(Path(raw_path).read_text())
        contract = artifact.get("elastic_balanced_contract")
        _require(isinstance(contract, dict), f"{key}: contract missing")
        arm = contract.get("arm")
        replicate = contract.get("replicate")
        _require(arm in ARMS and replicate in (0, 1), f"{key}: arm/replicate")
        _require(artifact.get("validation", {}).get("performance_claim_allowed") is True,
                 f"{key}: invalid artifact")
        fingerprints.add(contract.get("base_semantic_sha256"))
        requests = artifact.get("requests")
        decisions = artifact.get("router_decisions")
        _require(isinstance(requests, list) and isinstance(decisions, list),
                 f"{key}: request/decision rows missing")
        decision_map = {row["request_id"]: row for row in decisions}
        _require(len(decision_map) == len(decisions) == len(requests),
                 f"{key}: decision identity mismatch")
        for item, request in enumerate(requests):
            _require(request.get("valid") is True, f"{key}: invalid stream")
            decision = decision_map.get(request["request_id"])
            _require(decision is not None and decision.get("phase") == "complete",
                     f"{key}: incomplete route lifecycle")
            expected_arm = {
                "local": "always_local", "remote": "official_lmcache_remote",
                "predictor": "predictor", "tempo": "tempo",
            }[arm]
            _require(decision.get("arm") == expected_arm, f"{key}: arm mismatch")
            _require(decision.get("attempt", 0) >= 1, f"{key}: invalid attempt")
            route_counts[(arm, decision.get("route"))] += 1
            profile_ids.add((decision.get("profile_id"),
                             decision.get("profile_fingerprint_sha256")))
            metric = _metrics(request)
            item_key = (replicate, item)
            _require((arm, *item_key) not in rows_by_key, "duplicate paired row")
            rows_by_key[(arm, *item_key)] = {
                **metric,
                "output_sha256": request["output_text_sha256"],
                "prompt_tokens": request["usage"]["prompt_tokens"] - (
                    1 if decision.get("route") == "official_lmcache_remote_prefill" else 0),
                "output_tokens": request["requested_max_tokens"],
                "attempt": decision["attempt"],
                "route": decision["route"],
            }
            by_arm[arm].append(metric)
    _require(len(fingerprints) == 1 and len(profile_ids) == 1,
             "workload/profile identity changed")
    _require(all(len(by_arm[arm]) == 2 * len(by_arm["local"]) // 2
                 for arm in ARMS), "arm sample counts differ")

    pair_rows = []
    for replicate, item in sorted({key[1:] for key in rows_by_key}):
        values = {arm: rows_by_key[(arm, replicate, item)] for arm in ARMS}
        geometries = {(value["prompt_tokens"], value["output_tokens"])
                      for value in values.values()}
        hashes = {value["output_sha256"] for value in values.values()}
        _require(len(geometries) == 1 and len(hashes) == 1,
                 "paired geometry/output mismatch")
        pair_rows.append({
            "replicate": replicate, "item": item,
            **{
                f"{arm}_minus_remote_{metric}": values[arm][metric] - values["remote"][metric]
                for arm in ("local", "predictor", "tempo")
                for metric in ("ttft_ms", "e2e_ms", "tpot_ms", "tpot_max_ms")
            },
        })

    def summary(arm):
        values = by_arm[arm]
        return {
            metric: {
                "median": statistics.median(row[metric] for row in values),
                "p99_nearest_rank": _percentile([row[metric] for row in values], 0.99),
                "max": max(row[metric] for row in values),
            }
            for metric in ("ttft_ms", "e2e_ms", "tpot_ms", "tpot_max_ms")
        }

    tempo_e2e = [row["tempo_minus_remote_e2e_ms"] for row in pair_rows]
    tempo_tpot_max = [row["tempo_minus_remote_tpot_max_ms"] for row in pair_rows]
    arm_summary = {arm: summary(arm) for arm in ARMS}
    gates = {
        "all_streams_routes_outputs_exact": True,
        "tempo_e2e_median_beats_official_lmcache": statistics.median(tempo_e2e) < 0,
        "tempo_e2e_win_fraction_ge_60pct": sum(value < 0 for value in tempo_e2e) / len(tempo_e2e) >= 0.6,
        "tempo_tpot_max_p99_le_110pct_lmcache": (
            arm_summary["tempo"]["tpot_max_ms"]["p99_nearest_rank"]
            <= 1.10 * arm_summary["remote"]["tpot_max_ms"]["p99_nearest_rank"]),
        "tempo_worst_e2e_regression_le_100ms": max(tempo_e2e) <= 100.0,
    }
    return {
        "schema": "tempo-elastic-pd-balanced-analysis-445",
        "measurement_valid": True,
        "same_allocation_non_independent_screen": True,
        "profile": next(iter(profile_ids)),
        "arm_summary": arm_summary,
        "paired_rows": pair_rows,
        "route_counts": {f"{arm}:{route}": count
                         for (arm, route), count in sorted(route_counts.items())},
        "candidate_gates": gates,
        "candidate_passes": all(gates.values()),
        "verdict": "continue_elastic_pd" if all(gates.values()) else "revise_elastic_pd",
        "claim_boundary": (
            "actual vLLM Qwen2.5-7B TP4 P/D, two replicas, one live server epoch, "
            "official LMCacheConnectorV1 remote path, screen-only prior profile; "
            "not independent replication and no Mooncake comparison"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite")
    result = analyze(args.stage_root.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "gates": result["candidate_gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
