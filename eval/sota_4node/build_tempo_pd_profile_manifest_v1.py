#!/usr/bin/env python3
"""Build a frozen screen-only TEMPO-PD profile from explicit calibration runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from eval.sota_4node.run_tempo_pd_stream_metrics_v1 import SCHEMA as RAW_SCHEMA
from tempo.pd_admission import (
    PDCalibrationProfile, PDEvidenceLevel, PDPolicyConfig, PDWorkloadClass,
)
from tempo.pd_policy_manifest import PDPolicyManifest, write_manifest


REPORT_SCHEMA = "tempo-pd-profile-build-report-1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing explicit artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(raw: Mapping[str, Any], expected_mode: str) -> dict[str, dict[str, Any]]:
    _require(raw.get("schema") == RAW_SCHEMA, "raw schema mismatch")
    _require(raw.get("run", {}).get("mode") == expected_mode, "raw mode mismatch")
    _require(raw.get("validation", {}).get("performance_claim_allowed") is True,
             "calibration raw artifact is invalid")
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(requests, list) and isinstance(decisions, list),
             "raw requests/decisions missing")
    by_decision = {row.get("request_id"): row for row in decisions}
    _require(len(by_decision) == len(decisions), "duplicate decision request IDs")
    result: dict[str, dict[str, Any]] = {}
    for record in requests:
        request_id = record.get("request_id")
        _require(isinstance(request_id, str) and request_id not in result,
                 "invalid or duplicate request ID")
        decision = by_decision.get(request_id)
        _require(isinstance(decision, dict), "request has no router decision")
        arrivals = record.get("token_arrival_offsets_ns")
        dispatch = record.get("dispatch_offset_ns")
        _require(isinstance(arrivals, list) and arrivals and type(dispatch) is int,
                 "request timing is incomplete")
        _require(record.get("valid") is True and decision.get("phase") == "complete",
                 "request or decision is invalid")
        result[request_id] = {
            "workload_fingerprint": decision.get("workload_fingerprint"),
            "workload": decision.get("workload"),
            "route": decision.get("route"),
            "e2e_ms": (arrivals[-1] - dispatch) / 1_000_000.0,
            "output_text_sha256": record.get("output_text_sha256"),
        }
    _require(set(result) == set(by_decision), "request/decision identities differ")
    return result


def build_manifest(
    local_raw: Mapping[str, Any],
    remote_raw: Mapping[str, Any],
    *,
    classifier_version: str,
    policy_epoch: int,
    minimum_samples_per_route: int,
    remote_advantage_margin_ms: float,
) -> tuple[PDPolicyManifest, dict[str, Any]]:
    local = _rows(local_raw, "fixed_local")
    remote = _rows(remote_raw, "lmcache_always_remote")
    _require(set(local) == set(remote), "calibration request identities differ")
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for request_id in sorted(local):
        left, right = local[request_id], remote[request_id]
        _require(left["workload_fingerprint"] == right["workload_fingerprint"],
                 f"{request_id}: workload fingerprint differs")
        _require(left["workload"] == right["workload"],
                 f"{request_id}: workload class differs")
        _require(left["output_text_sha256"] == right["output_text_sha256"],
                 f"{request_id}: local/remote output mismatch")
        _require(left["route"] == "decoder_local_recompute_or_cache",
                 f"{request_id}: local route mismatch")
        _require(right["route"] == "remote_prefill_live_kv",
                 f"{request_id}: remote route mismatch")
        grouped.setdefault(str(left["workload_fingerprint"]), []).append((left, right))

    profiles: list[PDCalibrationProfile] = []
    groups: list[dict[str, Any]] = []
    for fingerprint in sorted(grouped):
        pairs = grouped[fingerprint]
        _require(len(pairs) >= minimum_samples_per_route,
                 f"{fingerprint}: insufficient calibration samples")
        workload_raw = pairs[0][0]["workload"]
        _require(isinstance(workload_raw, dict), "workload must be an object")
        expected_prefix = classifier_version + ":"
        _require(all(str(workload_raw[name]).startswith(expected_prefix) for name in (
            "prompt_bucket", "output_bucket", "decoder_load_bucket", "kv_bytes_bucket"
        )), "workload bucket classifier version mismatch")
        local_values = [pair[0]["e2e_ms"] for pair in pairs]
        remote_values = [pair[1]["e2e_ms"] for pair in pairs]
        profile = PDCalibrationProfile(
            workload=PDWorkloadClass(**workload_raw),
            evidence_level=PDEvidenceLevel.SCREEN,
            local_samples=len(local_values),
            remote_samples=len(remote_values),
            local_latency_p50_ms=statistics.median(local_values),
            remote_latency_p50_ms=statistics.median(remote_values),
            local_latency_lower_bound_ms=min(local_values),
            remote_latency_upper_bound_ms=max(remote_values),
            outputs_equivalent=True,
            remote_transfer_failures=0,
            valid_from_epoch=policy_epoch,
            valid_through_epoch=policy_epoch,
        )
        profiles.append(profile)
        groups.append({
            "workload_fingerprint": fingerprint,
            "profile_id": profile.profile_id,
            "samples_per_route": len(pairs),
            "local_e2e_ms": local_values,
            "remote_e2e_ms": remote_values,
            "remote_advantage_lower_bound_ms": profile.remote_advantage_lower_bound_ms,
        })
    manifest = PDPolicyManifest(
        classifier_version=classifier_version,
        policy_epoch=policy_epoch,
        deployment_scope="screen_only",
        config=PDPolicyConfig(
            remote_advantage_margin_ms=remote_advantage_margin_ms,
            minimum_samples_per_route=minimum_samples_per_route,
            require_replicated_evidence=False,
        ),
        profiles=tuple(profiles),
    )
    return manifest, {"groups": groups}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--classifier-version", required=True)
    parser.add_argument("--policy-epoch", type=int, required=True)
    parser.add_argument("--minimum-samples-per-route", type=int, default=3)
    parser.add_argument("--remote-advantage-margin-ms", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest, details = build_manifest(
        _load(args.local), _load(args.remote),
        classifier_version=args.classifier_version,
        policy_epoch=args.policy_epoch,
        minimum_samples_per_route=args.minimum_samples_per_route,
        remote_advantage_margin_ms=args.remote_advantage_margin_ms,
    )
    write_manifest(args.output, manifest)
    _require(not args.report.exists(), f"refusing to overwrite report: {args.report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": REPORT_SCHEMA,
        "manifest_id": manifest.manifest_id,
        "manifest_path": str(args.output.resolve()),
        "sources": {
            "local": {"path": str(args.local.resolve()), "sha256": _sha(args.local)},
            "remote": {"path": str(args.remote.resolve()), "sha256": _sha(args.remote)},
        },
        **details,
        "claim_boundary": "screen-only calibration; validation and replication remain separate",
    }
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
