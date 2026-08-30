#!/usr/bin/env python3
"""Build a compact, fail-closed diagnosis for Candidate O native evidence.

The campaign-level analyzer measures offered-population latency and business
outcomes.  This companion analyzer answers the candidate-specific causal
question: did the route-scoped telemetry quarantine introduced by Candidate O
actually activate?  It also keeps M/N/O cross-allocation comparisons explicitly
non-causal and never rewrites native raw data or the original analysis receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_O_ROOT = (
    ROOT / "results/tempo_go_c9_route_liveness_job_57736076_r3_canonical_outer"
)
DEFAULT_M = (
    ROOT
    / "results/tempo_go_c9_causal_burst_job_57732862"
    / "analysis_failclosed_business_v3.json"
)
DEFAULT_N = (
    ROOT
    / "results/tempo_go_c9_global_frontier_job_57732862"
    / "analysis_failclosed_business_v3.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT), f"artifact outside repository: {path}")
    return resolved.relative_to(ROOT).as_posix()


def route_failure_evidence(raw: dict[str, Any]) -> dict[str, int]:
    workload = raw.get("c8_dual_regime_contract")
    if workload is None:
        workload = raw.get("c7_joint_control_contract")
    require(isinstance(workload, dict), "workload contract missing")
    request_index = workload.get("request_index")
    require(isinstance(request_index, dict), "request index missing")

    counters = {
        "all_global_decisions": 0,
        "victim_global_decisions": 0,
        "all_decisions_with_nonzero_route_failure_counter": 0,
        "victim_decisions_with_nonzero_route_failure_counter": 0,
        "all_route_failure_quarantine_rejections": 0,
        "victim_route_failure_quarantine_rejections": 0,
    }
    for row in raw.get("router_decisions", []):
        if not isinstance(row, dict):
            continue
        decision = row.get("frontend_tempo_go_decision")
        if not isinstance(decision, dict) or decision.get("kind") not in {
            "admit", "reject",
        }:
            continue
        counters["all_global_decisions"] += 1
        identity = request_index.get(row.get("request_id"), {})
        is_victim = isinstance(identity, dict) and identity.get("role") == "victim"
        if is_victim:
            counters["victim_global_decisions"] += 1

        nonzero = False
        provenance = decision.get("telemetry_provenance", {})
        if isinstance(provenance, dict):
            for record in provenance.values():
                if not isinstance(record, dict):
                    continue
                failures = record.get("route_failures", {})
                if isinstance(failures, dict) and (
                    int(failures.get("local_count", 0) or 0) > 0
                    or int(failures.get("remote_count", 0) or 0) > 0
                ):
                    nonzero = True
        if nonzero:
            counters["all_decisions_with_nonzero_route_failure_counter"] += 1
            if is_victim:
                counters["victim_decisions_with_nonzero_route_failure_counter"] += 1

        rejected = decision.get("rejected_candidates", [])
        if isinstance(rejected, list):
            quarantine_rejections = sum(
                1
                for item in rejected
                if isinstance(item, dict)
                and item.get("reason") == "route_failure_quarantine"
            )
            counters["all_route_failure_quarantine_rejections"] += quarantine_rejections
            if is_victim:
                counters["victim_route_failure_quarantine_rejections"] += (
                    quarantine_rejections
                )
    return counters


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def policy_business(analysis: dict[str, Any]) -> dict[str, Any]:
    business = analysis["business"]["full_tempo"]
    telemetry = analysis["telemetry"]
    return {
        "foreground": business["foreground"],
        "background": business["background"],
        "observer_supported_decisions": int(
            telemetry["full_cross_layer_supported_decisions"]
        ),
        "observer_total_decisions": int(telemetry["full_victim_global_decisions"]),
        "observer_supported_fraction": float(
            telemetry["full_cross_layer_supported_fraction"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o-root", type=Path, default=DEFAULT_O_ROOT)
    parser.add_argument("--m-analysis", type=Path, default=DEFAULT_M)
    parser.add_argument("--n-analysis", type=Path, default=DEFAULT_N)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    o_root = args.o_root.resolve()
    original_path = o_root / "analysis.json"
    posthoc_path = o_root / "analysis_failclosed_business_v2.json"
    completed_path = o_root / "completed_attempt.json"
    output = (args.output or (o_root / "candidate_o_diagnosis.json")).resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    for path in (
        original_path, posthoc_path, completed_path,
        args.m_analysis.resolve(), args.n_analysis.resolve(),
    ):
        require(path.is_file(), f"artifact missing: {path}")

    original = load(original_path)
    posthoc = load(posthoc_path)
    completed = load(completed_path)
    require(
        completed.get("analysis_sha256") == digest(original_path),
        "completed receipt does not bind original analysis",
    )
    require(
        original.get("contract_sha256") == posthoc.get("contract_sha256"),
        "post-hoc analysis contract differs",
    )

    candidate_blocks = [
        item
        for item in posthoc["blocks"]
        if item.get("arm") == "full_c7_managed_background"
    ]
    require(len(candidate_blocks) == 1, "exactly one Candidate O arm required")
    candidate = candidate_blocks[0]
    result_path = Path(candidate["result"]).resolve()
    require(digest(result_path) == candidate["result_sha256"], "result SHA differs")
    result = load(result_path)
    raw_specs = result["analysis"]["blocks"]
    require(len(raw_specs) == 7, "seven Candidate O workload blocks required")

    route_totals: dict[str, int] = {}
    route_by_block: dict[str, dict[str, int]] = {}
    for spec in raw_specs:
        raw_path = Path(spec["raw"]).resolve()
        require(digest(raw_path) == spec["raw_sha256"], f"raw SHA differs: {raw_path}")
        evidence = route_failure_evidence(load(raw_path))
        route_by_block[str(spec["name"])] = evidence
        add_counts(route_totals, evidence)

    mechanism_activated = bool(
        route_totals["all_decisions_with_nonzero_route_failure_counter"]
        or route_totals["all_route_failure_quarantine_rejections"]
    )
    m_analysis = load(args.m_analysis.resolve())
    n_analysis = load(args.n_analysis.resolve())
    observer_by_block = candidate["decision_telemetry"]["by_block"]

    payload = {
        "schema": "tempo-go-c9-candidate-o-route-liveness-diagnosis-v1",
        "candidate": "Candidate O route-scoped telemetry failure quarantine",
        "native_campaign": {
            "allocation": 57736076,
            "nodes": 4,
            "gpus": 16,
            "original_analysis": relative(original_path),
            "original_analysis_sha256": digest(original_path),
            "posthoc_business_analysis": relative(posthoc_path),
            "posthoc_business_analysis_sha256": digest(posthoc_path),
            "completed_receipt": relative(completed_path),
            "completed_receipt_sha256": digest(completed_path),
            "one_campaign_no_retry": True,
            "discovery_only": True,
        },
        "candidate_specific_mechanism": {
            "changed_input": "telemetry_failure_quarantine_scope: pair -> route",
            "route_failure_evidence_by_block": route_by_block,
            "route_failure_evidence_total": route_totals,
            "activated": mechanism_activated,
            "causal_mechanism_positive": False,
            "interpretation": (
                "No telemetry route-failure counter or route-failure quarantine "
                "rejection was observed, so O-vs-M differences cannot be "
                "attributed to the changed mechanism."
            ),
        },
        "candidate_o": {
            "regimes": posthoc["aggregates"]["full_c7_managed_background"],
            "business": policy_business(posthoc),
            "observer_by_block": observer_by_block,
            "route_counts": {
                "local": 160,
                "remote": 47,
            },
            "gates": posthoc["gates"],
            "causal_discovery_positive": posthoc["causal_discovery_positive"],
            "performance_claim_allowed": False,
        },
        "cross_allocation_context_noncausal": {
            "candidate_m": {
                "allocation": 57732862,
                "analysis": relative(args.m_analysis.resolve()),
                "analysis_sha256": digest(args.m_analysis.resolve()),
                "business": policy_business(m_analysis),
            },
            "candidate_n": {
                "allocation": 57732862,
                "analysis": relative(args.n_analysis.resolve()),
                "analysis_sha256": digest(args.n_analysis.resolve()),
                "business": policy_business(n_analysis),
            },
            "candidate_o": {
                "allocation": 57736076,
                "business": policy_business(posthoc),
            },
            "causal_comparison_allowed": False,
            "reason": (
                "M/N and O used separate allocations and one sample per arm; "
                "co-job pressure and observer survival varied between campaigns."
            ),
        },
        "root_cause": {
            "observer_liveness": (
                "The co-load generator and observer publisher share one co-job "
                "lifetime. Official LMCache/NIXL timeout terminates that process, "
                "leaving later decisions with stale observer state."
            ),
            "admission_lifecycle": (
                "Global admission does not atomically reserve endpoint service-lane "
                "capacity through completion; HTTP 503 service-lane failures and "
                "global queue rejections remain distinct business failures."
            ),
        },
        "next_gate": [
            "independent durable telemetry/failure sidecar with terminal failure lease",
            "atomic global-to-endpoint service-lane reservation and exactly-once release",
            "within-allocation counterbalanced paired repeats with at least three samples",
            "unchanged strongest fixed, predictor, queue-GPU, and same offered population",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    print("sha256", digest(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
