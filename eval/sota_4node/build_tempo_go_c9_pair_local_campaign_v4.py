#!/usr/bin/env python3
"""Build the observer-gated C9 pair-local campaign without overwriting v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


ROOT = Path(__file__).resolve().parents[2]
PARENT_BASE = ROOT / "eval/sota_4node/tempo_go_c9_business_lane_base_contract_v14.json"
PARENT_C9 = ROOT / "eval/sota_4node/tempo_go_c9_business_lane_followup_contract_v11.json"
SOURCE_PROFILE = ROOT / "results/tempo_go_c9_dual_route_business_lane_profile_v12/real_tempo_go_c9_dual_route_business_lane_profile_v12.json"
OUT_DIR = ROOT / "results/tempo_go_c9_pair_local_campaign_v4"
PROFILE = OUT_DIR / "real_tempo_go_c9_pair_local_profile_v4.json"
BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v4.json"
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v4.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def refresh_inventory(inventory: dict[str, str]) -> dict[str, str]:
    refreshed: dict[str, str] = {}
    for relative in inventory:
        source = (ROOT / relative).resolve()
        if not source.is_file() or ROOT not in source.parents:
            raise RuntimeError(f"source inventory entry is missing: {relative}")
        refreshed[relative] = digest(source)
    return dict(sorted(refreshed.items()))


def main() -> int:
    if any(path.exists() for path in (PROFILE, BASE, CONTRACT)):
        raise RuntimeError("pair-local v4 artifact already exists")

    profile = json.loads(SOURCE_PROFILE.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_pair_local_profile_v4"
    profile["controller"]["cross_layer_local_receiver_price_ms"] = 0.10
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(PROFILE, profile)
    loaded = load_global_profile(PROFILE)

    base = json.loads(PARENT_BASE.read_text(encoding="utf-8"))
    base["candidate"]["id"] = "tempo-go-c9-pair-local-receiver-price-v3-observer-gated"
    base["purpose"] = (
        "Observer-gated C9 discovery of pair-local receiver externality "
        "pricing under the same actual NCCL plus official LMCache receiver-incast burst."
    )
    base["joint_control"]["global_profile"] = {
        "path": str(PROFILE.relative_to(ROOT)),
        "sha256": digest(PROFILE),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["source_inventory"] = refresh_inventory(base["source_inventory"])
    write_new(BASE, base)

    contract = json.loads(PARENT_C9.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "Observer-gated ABBA C9 discovery of pair-local receiver-aware global "
        "routing under the same actual NCCL plus official LMCache receiver-incast burst."
    )
    contract["system_under_test"]["base_contract"] = str(BASE.relative_to(ROOT))
    contract["system_under_test"]["base_contract_sha256"] = digest(BASE)
    contract["system_under_test"]["observer_scope"] = (
        "pair-scoped P0-D0 co-job observer; pair 1 remains not_collected; "
        "victim release waits for the first active snapshot"
    )
    contract["claim_boundary"] = {
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "discovery_only": True,
        "reason": "same-allocation observer-gated pair-local telemetry/controller discovery",
    }
    contract["mechanism"] = {
        "schema": "tempo-go-c9-pair-local-receiver-mechanism-v2",
        "pair_local_observer_path": "nccl_observer_pair-{pair}.json",
        "observer_gate": "active_correct_sequence_at_least_one_before_victim_release",
        "local_receiver_price_ms_per_observed_ms": 0.10,
        "cojob_pair": 0,
        "unobserved_pair_policy": "not_collected_no_synthetic_pressure",
        "oracle_input": False,
        "workload_changed": False,
        "transport_changed": False,
    }
    contract["source_inventory"] = refresh_inventory(contract["source_inventory"])
    write_new(CONTRACT, contract)
    print(CONTRACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
