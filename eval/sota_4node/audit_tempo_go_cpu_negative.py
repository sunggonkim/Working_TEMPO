#!/usr/bin/env python3
"""Audit a reproducible CPU-control-plane negative conclusion.

The audit is deliberately narrower than a native performance claim.  It proves
that two structurally different, contract-bound global candidates were replayed
on the same held-out trace, against the same fixed arms, and both failed the
pre-registered primary median gate.  It also records which native claims remain
unproven.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tempo.pd_global_profile import load_global_profile


ARMS = ("always_local", "official_always_remote", "predictor_only", "queue_gpu_only", "tempo_go")
FIXED_ARMS = ARMS[:4]


def _sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contract_binding(replay: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    contract = _load(contract_path)
    _require(replay.get("run_contract") == str(contract_path.resolve()), "replay contract path differs")
    _require(replay.get("run_contract_sha256") == _file_sha(contract_path), "replay contract SHA differs")
    _require(replay.get("run_contract_fingerprint_sha256") == contract.get("fingerprint_sha256"), "replay contract fingerprint differs")
    gates = replay.get("replay_gates")
    _require(isinstance(gates, dict) and gates.get("frozen_run_contract_valid") is True, "frozen run contract was not verified")
    return contract


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_replay(replay_path: Path, contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = _load(replay_path)
    _require(replay.get("schema") == "tempo-go-global-five-arm-replay-v2", f"{replay_path} schema is not v2")
    _require(replay.get("performance_claim_allowed") is False, "replay permits a performance claim")
    _require(replay.get("native_gpu_run_allowed") is False, "CPU replay permits native GPU use")
    gates = replay.get("replay_gates")
    _require(isinstance(gates, dict), "replay gates are missing")
    for key in (
        "all_arms_have_same_request_count",
        "all_arms_have_same_trace_sha",
        "all_arms_no_phase_policy_input",
        "all_arms_no_physical_switch_input",
        "all_arms_terminal_and_leak_free",
        "manifest_valid",
        "frozen_run_contract_valid",
    ):
        _require(gates.get(key) is True, f"replay gate failed: {key}")
    arms = replay.get("arms")
    _require(isinstance(arms, dict) and set(arms) == set(ARMS), "replay arm inventory differs")
    contract = _contract_binding(replay, contract_path)
    global_binding = contract["artifacts"]["global_profile"]
    _require(Path(str(global_binding["path"])).resolve() == Path(str(replay["global_profile"])).resolve(), "contract/global profile path differs")
    _require(global_binding["fingerprint_sha256"] == replay["global_profile_sha256"], "contract/global profile fingerprint differs")
    _require(contract["artifacts"]["manifest"]["sha256"] == replay["manifest_sha256"], "contract/manifest SHA differs")
    _require(contract["artifacts"]["workload"]["sha256"] == replay["workload_sha256"], "contract/workload SHA differs")
    return replay, contract


def _fixed_fingerprints(replay: dict[str, Any]) -> dict[str, str]:
    return {arm: _sha(replay["arms"][arm]) for arm in FIXED_ARMS}


def _candidate_record(replay: dict[str, Any], candidate: str) -> dict[str, Any]:
    arms = replay["arms"]
    fixed_p50 = {
        arm: float(arms[arm]["e2e_ms"]["p50"])
        for arm in FIXED_ARMS
    }
    strongest_fixed = min(fixed_p50, key=fixed_p50.get)
    candidate_p50 = float(arms["tempo_go"]["e2e_ms"]["p50"])
    predictor_p50 = fixed_p50["predictor_only"]
    strongest_fixed_gate_limit = fixed_p50[strongest_fixed] * 0.90
    predictor_gate_limit = predictor_p50 * 0.95
    return {
        "candidate": candidate,
        "profile_id": replay["global_profile"],
        "request_count": arms["tempo_go"]["request_count"],
        "tempo_completed": arms["tempo_go"]["completed"],
        "tempo_rejected": arms["tempo_go"]["rejected"],
        "tempo_failed": arms["tempo_go"]["failed"],
        "tempo_e2e_p50_ms": candidate_p50,
        "tempo_e2e_p99_ms": float(arms["tempo_go"]["e2e_ms"]["p99"]),
        "strongest_fixed_arm": strongest_fixed,
        "strongest_fixed_p50_ms": fixed_p50[strongest_fixed],
        "strongest_fixed_gate_limit_ms": strongest_fixed_gate_limit,
        "predictor_p50_ms": predictor_p50,
        "predictor_gate_limit_ms": predictor_gate_limit,
        "primary_median_gate_pass": (
            candidate_p50 <= strongest_fixed_gate_limit
            and candidate_p50 <= predictor_gate_limit
        ),
        "primary_median_gate_failure_reasons": [
            reason for reason, passed in (
                ("not_10_percent_below_strongest_fixed", candidate_p50 <= strongest_fixed_gate_limit),
                ("not_5_percent_below_predictor", candidate_p50 <= predictor_gate_limit),
            ) if not passed
        ],
        "fixed_arm_fingerprints": _fixed_fingerprints(replay),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-a-replay", type=Path, required=True)
    parser.add_argument("--candidate-a-profile", type=Path, required=True)
    parser.add_argument("--candidate-a-contract", type=Path, required=True)
    parser.add_argument("--candidate-a-name", required=True)
    parser.add_argument("--candidate-b-replay", type=Path, required=True)
    parser.add_argument("--candidate-b-profile", type=Path, required=True)
    parser.add_argument("--candidate-b-contract", type=Path, required=True)
    parser.add_argument("--candidate-b-name", required=True)
    parser.add_argument("--failure-replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replay_a, contract_a = _validate_replay(args.candidate_a_replay.resolve(), args.candidate_a_contract.resolve())
    replay_b, contract_b = _validate_replay(args.candidate_b_replay.resolve(), args.candidate_b_contract.resolve())
    profile_a = load_global_profile(args.candidate_a_profile.resolve())
    profile_b = load_global_profile(args.candidate_b_profile.resolve())
    _require(profile_a.fingerprint_sha256 == replay_a["global_profile_sha256"], "candidate A profile fingerprint differs")
    _require(profile_b.fingerprint_sha256 == replay_b["global_profile_sha256"], "candidate B profile fingerprint differs")

    shared = (
        "manifest_sha256",
        "workload_sha256",
        "baseline_global_profile_sha256",
        "elastic_profile_sha256",
        "endpoint_profile_sha256",
    )
    shared_identity = {}
    for key in shared:
        _require(replay_a.get(key) == replay_b.get(key), f"candidate replays differ in {key}")
        shared_identity[key] = replay_a.get(key)
    fingerprints_a = _fixed_fingerprints(replay_a)
    fingerprints_b = _fixed_fingerprints(replay_b)
    _require(fingerprints_a == fingerprints_b, "fixed-arm receipts differ across candidate replays")

    tenant_reservations = [item.queue_reservation_slots for item in profile_a.tenants]
    telemetry_mode_a = profile_a.controller.get("telemetry_failure_quarantine_mode", "disabled")
    telemetry_mode_b = profile_b.controller.get("telemetry_failure_quarantine_mode", "disabled")
    structural_diff = {
        "candidate_a_queue_reservation_slots": tenant_reservations,
        "candidate_a_telemetry_failure_mode": telemetry_mode_a,
        "candidate_b_queue_reservation_slots": [item.queue_reservation_slots for item in profile_b.tenants],
        "candidate_b_telemetry_failure_mode": telemetry_mode_b,
        "different_mechanism": (
            tenant_reservations != [item.queue_reservation_slots for item in profile_b.tenants]
            and telemetry_mode_a != telemetry_mode_b
        ),
    }
    _require(structural_diff["different_mechanism"], "candidate mechanisms are not structurally different")

    result: dict[str, Any] = {
        "schema": "tempo-go-cpu-negative-audit-v1",
        "scope": "contract-bound CPU control-plane replay; not native performance evidence",
        "performance_claim_allowed": False,
        "native_performance_negative_proven": False,
        "shared_identity": shared_identity,
        "candidate_a": {
            "name": args.candidate_a_name,
            "profile_fingerprint_sha256": profile_a.fingerprint_sha256,
            "contract_fingerprint_sha256": contract_a["fingerprint_sha256"],
            "evidence": _candidate_record(replay_a, args.candidate_a_name),
        },
        "candidate_b": {
            "name": args.candidate_b_name,
            "profile_fingerprint_sha256": profile_b.fingerprint_sha256,
            "contract_fingerprint_sha256": contract_b["fingerprint_sha256"],
            "evidence": _candidate_record(replay_b, args.candidate_b_name),
        },
        "structural_difference": structural_diff,
        "same_fixed_arm_receipts": True,
        "reproducible_primary_median_negative": True,
        "completion_status": "CPU_negative_only_native_validation_unproven",
    }
    if args.failure_replay is not None:
        failure = _load(args.failure_replay.resolve())
        _require(failure.get("replay_gates", {}).get("frozen_run_contract_valid") is True, "failure replay contract was not verified")
        _require(failure["arms"]["tempo_go"].get("telemetry_failure_injection", {}).get("triggered") is True, "failure replay did not trigger telemetry failure")
        result["candidate_b_failure_evidence"] = {
            "replay": str(args.failure_replay.resolve()),
            "tempo_completed": failure["arms"]["tempo_go"]["completed"],
            "tempo_rejected": failure["arms"]["tempo_go"]["rejected"],
            "tempo_failed": failure["arms"]["tempo_go"]["failed"],
            "tempo_e2e_p99_ms": failure["arms"]["tempo_go"]["e2e_ms"]["p99"],
            "telemetry_failure_injection": failure["arms"]["tempo_go"]["telemetry_failure_injection"],
            "utility_gate_pass": False,
        }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
