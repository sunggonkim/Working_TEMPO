#!/usr/bin/env python3
"""Analyze credit-admission against local and an explicit failed remote arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--failed-remote-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local = base._parse_run("local", _load(args.local), ttft_slo_ms=3000,
                            tpot_slo_ms=250, e2e_slo_ms=12000)
    candidate = base._parse_run("candidate", _load(args.candidate), ttft_slo_ms=3000,
                                tpot_slo_ms=250, e2e_slo_ms=12000)
    correctness = (
        local["model_config_sha256"] == candidate["model_config_sha256"]
        and local["workload_sha256"] == candidate["workload_sha256"]
        and local["_contracts"] == candidate["_contracts"]
        and local["_outputs"] == candidate["_outputs"]
    )
    logs = sorted((args.failed_remote_root / "crossover_remote").glob("node-*-vllm.log"))
    allocation_failures = sum(
        path.read_text(encoding="utf-8", errors="replace").count("Receiver allocation failed")
        for path in logs
    )
    failed_remote_raw_absent = not (
        args.failed_remote_root / "crossover_remote/raw.json"
    ).exists()
    paired = base._paired(candidate, local) if correctness else None
    perf = candidate["performance"]
    reference = local["performance"]
    routes = candidate["routes"]
    gates = {
        "same_model_workload_schedule_outputs": correctness,
        "mixed_local_and_remote_routes_observed": (
            routes.get("remote_prefill_live_kv", 0) > 0
            and routes.get("decoder_local_recompute_or_cache", 0) > 0
        ),
        "official_remote_oversubscription_failure_observed": (
            allocation_failures > 0 and failed_remote_raw_absent
        ),
        "candidate_all_requests_slo_valid": (
            perf["slo_goodput"]["success_fraction"] == 1.0
        ),
        "candidate_goodput_at_least_local": (
            perf["slo_goodput"]["request_goodput_per_s"]
            >= reference["slo_goodput"]["request_goodput_per_s"]
        ),
        "candidate_e2e_p50_within_5_percent_of_local": (
            perf["e2e_ms"]["p50"] <= reference["e2e_ms"]["p50"] * 1.05
        ),
        "candidate_tpot_p99_within_10_percent_of_local": (
            perf["tpot_ms"]["p99"] <= reference["tpot_ms"]["p99"] * 1.10
        ),
    }

    def public(row: dict) -> dict:
        return {key: value for key, value in row.items() if not key.startswith("_")}

    result = {
        "schema": "tempo-pd-credit-admission-analysis-13",
        "local": public(local),
        "tempo_credit_admission": public(candidate),
        "paired_candidate_minus_local": paired,
        "failed_official_remote": {
            "root": str(args.failed_remote_root.resolve()),
            "receiver_allocation_failure_count": allocation_failures,
            "raw_result_absent_after_bounded_abort": failed_remote_raw_absent,
        },
        "gates": gates,
        "passes_credit_admission_gate": all(gates.values()),
        "verdict": "credit_admission_promising" if all(gates.values()) else "revise_credit_admission",
        "claim_boundary": (
            "Same-allocation overload screen: official LMCache remote failed to complete; "
            "candidate latency is compared to exact fixed-local, not an unavailable remote raw."
        ),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
