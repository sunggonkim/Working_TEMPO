#!/usr/bin/env python3
"""Combine the frozen bursty and steady microburst25 validations."""

import argparse
import json
from pathlib import Path
import statistics


SCHEMA = "tempo-pd-microburst25-suite-analysis-348"
INPUT_SCHEMA = "tempo-pd-online-regime-microburst25-analysis-343"
POLICY = "tempo-pd-online-regime-microburst25-credit5-342"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict:
    value = json.loads(path.resolve().read_text())
    _require(isinstance(value, dict), f"{path}: expected object")
    return value


def analyze(bursty_path: Path, steady_path: Path) -> dict:
    inputs = {
        "bursty": _load(bursty_path),
        "steady": _load(steady_path),
    }
    allocations = set()
    pooled_e2e = []
    pooled_tpot = []
    for workload, report in inputs.items():
        _require(report.get("schema") == INPUT_SCHEMA, f"{workload}: schema")
        _require(report.get("policy") == POLICY, f"{workload}: policy")
        _require(report.get("workload_class") == workload, f"{workload}: class")
        _require(report.get("passes") is True, f"{workload}: did not pass")
        _require(all(report.get("gates", {}).values()), f"{workload}: gate")
        pairs = report.get("pairs")
        _require(isinstance(pairs, list) and len(pairs) == 24,
                 f"{workload}: pairs")
        _require([row.get("item") for row in pairs] == list(range(24)),
                 f"{workload}: item identity")
        allocations.add(report.get("allocation_id"))
        pooled_e2e.extend(float(row["e2e_delta_ms"]) for row in pairs)
        pooled_tpot.extend(float(row["tpot_delta_ms"]) for row in pairs)

    _require(len(allocations) == 1 and None not in allocations,
             "inputs must come from one explicit allocation")
    burst = inputs["bursty"]
    steady = inputs["steady"]
    _require(burst["microburst_active_request_count"] == 20,
             "bursty credit activation")
    _require(len(burst["capped_requests"]) == 2, "bursty cap geometry")
    _require(sorted(row["item"] for row in burst["capped_requests"]) == [14, 15],
             "bursty capped identities")
    _require(burst["route_counts"] == {
        "tempo_local": 22, "tempo_remote": 2, "lmcache_remote": 24,
    }, "bursty routing")
    _require(steady["microburst_active_request_count"] == 0,
             "steady credit deactivation")
    _require(steady["capped_requests"] == [], "steady must not cap")
    _require(steady["route_counts"] == {
        "tempo_local": 24, "tempo_remote": 0, "lmcache_remote": 24,
    }, "steady routing")

    gates = {
        "both_workloads_pass_individually": True,
        "bursty_credit_activates_and_caps_exactly_two": True,
        "steady_credit_deactivates": True,
        "pooled_e2e_win_fraction_ge_90pct": sum(x < 0 for x in pooled_e2e) >= 44,
        "pooled_e2e_median_improves": statistics.median(pooled_e2e) < 0,
        "pooled_tpot_win_fraction_ge_95pct": sum(x < 0 for x in pooled_tpot) >= 46,
        "pooled_tpot_median_improves": statistics.median(pooled_tpot) < 0,
    }
    passes = all(gates.values())
    return {
        "schema": SCHEMA,
        "allocation_id": allocations.pop(),
        "policy": POLICY,
        "evidence_scope": (
            "one 4-node allocation; two predeclared workload shapes; "
            "actual vLLM P/D with the official LMCacheConnectorV1 baseline"
        ),
        "inputs": {
            "bursty": str(bursty_path.resolve()),
            "steady": str(steady_path.resolve()),
        },
        "workloads": {
            name: {
                "route_counts": report["route_counts"],
                "pair_local_median_gap_ns": report["pair_local_median_gap_ns"],
                "microburst_active_request_count": report[
                    "microburst_active_request_count"
                ],
                "capped_requests": report["capped_requests"],
                "summary": report["summary"],
            }
            for name, report in inputs.items()
        },
        "pooled_summary": {
            "paired_requests": 48,
            "e2e_win_count": sum(x < 0 for x in pooled_e2e),
            "e2e_delta_median_ms": statistics.median(pooled_e2e),
            "tpot_win_count": sum(x < 0 for x in pooled_tpot),
            "tpot_delta_median_ms": statistics.median(pooled_tpot),
        },
        "gates": gates,
        "passes": passes,
        "verdict": (
            "freeze_microburst25_credit5_controller"
            if passes else "revise_microburst25_controller"
        ),
        "claim_boundary": {
            "lmcache": "same-request actual-vLLM P/D baseline in this suite",
            "mooncake": (
                "not measured: Mooncake runtime is absent and no safe same-harness "
                "explicit-topology adapter is installed"
            ),
            "sota": "not claimed without an independent allocation and Mooncake parity",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bursty", type=Path, required=True)
    parser.add_argument("--steady", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    result = analyze(args.bursty, args.steady)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"],
                      "pooled_summary": result["pooled_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
