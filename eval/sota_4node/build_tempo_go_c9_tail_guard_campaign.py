#!/usr/bin/env python3
"""Create a source-bound C9 campaign with a stronger endpoint-utilization price.

This is an experiment artifact, not a replacement for the frozen C9
discovery contract.  It changes one controller knob only so the existing
ABBA workload and native co-job remain identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


ROOT = Path(__file__).resolve().parents[2]
PARENT_BASE = ROOT / "eval/sota_4node/tempo_go_c9_business_lane_base_contract_v14.json"
PARENT_C9 = ROOT / "eval/sota_4node/tempo_go_c9_business_lane_followup_contract_v11.json"
SOURCE_PROFILE = ROOT / "results/tempo_go_c9_dual_route_business_lane_profile_v12/real_tempo_go_c9_dual_route_business_lane_profile_v12.json"
OUT_DIR = ROOT / "results/tempo_go_c9_tail_guard_campaign_v1"
PROFILE = OUT_DIR / "real_tempo_go_c9_tail_guard_profile_v1.json"
BASE = ROOT / "eval/sota_4node/tempo_go_c9_tail_guard_base_contract_v1.json"
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_tail_guard_contract_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    if any(path.exists() for path in (PROFILE, BASE, CONTRACT)):
        raise RuntimeError("tail-guard campaign artifact already exists")
    profile = json.loads(SOURCE_PROFILE.read_text(encoding="utf-8"))
    # Only this policy parameter changes.  It raises the score cost of an
    # already observed endpoint queue/utilization imbalance, while retaining
    # the same cross-layer signals, business lane, capacities, and workload.
    profile["controller"]["utilization_penalty_ms"] = 500
    profile["profile_id"] = "real_tempo_go_c9_tail_guard_profile_v1"
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(PROFILE, profile)
    loaded = load_global_profile(PROFILE)

    base = json.loads(PARENT_BASE.read_text(encoding="utf-8"))
    base["candidate"]["id"] = "tempo-go-c9-tail-guard-utilization-price-v1"
    base["purpose"] = (
        "C9 discovery of a stronger endpoint-utilization shadow price against "
        "the measured normal-control D0 tail while preserving the same "
        "receiver-incast and native transport workload."
    )
    base["joint_control"]["global_profile"] = {
        "path": str(PROFILE.relative_to(ROOT)),
        "sha256": sha256(PROFILE),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["source_inventory"] = dict(sorted(base["source_inventory"].items()))
    write_new(BASE, base)

    contract = json.loads(PARENT_C9.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "ABBA C9 discovery of a utilization-price tail guard under the same "
        "actual NCCL plus official LMCache receiver-incast burst."
    )
    contract["system_under_test"]["base_contract"] = str(BASE.relative_to(ROOT))
    contract["system_under_test"]["base_contract_sha256"] = sha256(BASE)
    contract["claim_boundary"] = {
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "discovery_only": True,
        "reason": "same-allocation parameter sensitivity after C9 tail diagnosis",
    }
    contract["mechanism"] = {
        "schema": "tempo-go-c9-tail-guard-mechanism-v1",
        "changed_parameter": "utilization_penalty_ms",
        "baseline_value": 100,
        "candidate_value": 500,
        "oracle_input": False,
        "workload_changed": False,
        "transport_changed": False,
    }
    write_new(CONTRACT, contract)
    print(CONTRACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
