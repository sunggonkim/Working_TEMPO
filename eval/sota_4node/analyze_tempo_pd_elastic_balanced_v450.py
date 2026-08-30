#!/usr/bin/env python3
"""Strengthen Elastic-PD evidence with phase-correct credit validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_elastic_balanced_v445 as prior


def _credit_evidence(stage_root: Path) -> dict:
    public = json.loads((stage_root / "raw.json").read_text())
    artifacts = public["elastic_balanced_orchestration"]["artifacts"]
    rows = []
    for key, raw_path in sorted(artifacts.items()):
        artifact = json.loads(Path(raw_path).read_text())
        contract = artifact.get("elastic_balanced_contract", {})
        if contract.get("arm") != "tempo":
            continue
        decisions = artifact.get("router_decisions")
        prior._require(isinstance(decisions, list), f"{key}: decisions missing")
        prior._require(len(decisions) == len(artifact.get("requests", [])),
                       f"{key}: request/decision count mismatch")
        for row in decisions:
            prior._require(row.get("arm") == "tempo", f"{key}: non-TEMPO row")
            prior._require(row.get("phase") == "complete", f"{key}: incomplete row")
            prior._require(row.get("error") is None, f"{key}: route error")
            prior._require(row.get("route") != "bounded_ingress_queue",
                           f"{key}: terminal queued route")
            prior._require(
                row.get("admission_credit_scope") == "prefill_or_remote_handoff",
                f"{key}: credit scope mismatch",
            )
            prior._require(
                row.get("admission_credit_release_event") == "first_response_chunk",
                f"{key}: credit release event mismatch",
            )
            started = row.get("started_ns")
            released = row.get("admission_credit_released_ns")
            response = row.get("response_started_ns")
            finished = row.get("finished_ns")
            prior._require(
                all(type(value) is int for value in (started, released, response, finished)),
                f"{key}: lifecycle timestamps missing",
            )
            prior._require(started <= released == response < finished,
                           f"{key}: credit lifecycle ordering")
            prior._require(type(row.get("attempt")) is int and row["attempt"] >= 1,
                           f"{key}: invalid attempt")
            rows.append(row)
    prior._require(len(rows) == 48, "exactly 48 TEMPO decisions required")
    return {
        "tempo_decisions": len(rows),
        "first_response_credit_releases": len(rows),
        "terminal_queue_routes": 0,
        "route_errors": 0,
        "retried_decisions": sum(row["attempt"] > 1 for row in rows),
        "max_attempt": max(row["attempt"] for row in rows),
        "release_precedes_stream_finish_all": True,
    }


def analyze(stage_root: Path) -> dict:
    result = prior.analyze(stage_root)
    evidence = _credit_evidence(stage_root)
    result["schema"] = "tempo-elastic-pd-balanced-analysis-450"
    result["credit_lifecycle"] = evidence
    result["candidate_gates"][
        "tempo_credit_scope_and_first_response_release_exact"
    ] = True
    result["candidate_passes"] = all(result["candidate_gates"].values())
    result["verdict"] = (
        "continue_elastic_pd" if result["candidate_passes"]
        else "revise_elastic_pd"
    )
    result["claim_boundary"] = (
        "actual vLLM Qwen2.5-7B TP4 P/D, two replicas, one live server epoch, "
        "official LMCacheConnectorV1 remote path, weighted local/remote admission "
        "credits released at the first response chunk; cache residency was "
        "conservatively classified as confirmed miss; same-allocation screen, "
        "not independent replication and no Mooncake comparison"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior._require(not args.output.exists(), "refusing to overwrite")
    result = analyze(args.stage_root.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": result["verdict"],
        "gates": result["candidate_gates"],
        "credit_lifecycle": result["credit_lifecycle"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
