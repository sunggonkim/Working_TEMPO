#!/usr/bin/env python3
"""Freeze C9 v10 with pressure-aware business pair spill enabled."""

from __future__ import annotations

import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile
from eval.sota_4node.build_tempo_go_c9_pair_local_campaign_v7 import (
    digest, refresh_inventory, write_new,
)


ROOT = Path(__file__).resolve().parents[2]
V9_PROFILE = ROOT / "results/tempo_go_c9_pair_local_campaign_v9/real_tempo_go_c9_pair_local_profile_v9.json"
V9_BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v9.json"
V9_CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v9.json"
OUT_DIR = ROOT / "results/tempo_go_c9_pair_local_campaign_v10"
PROFILE = OUT_DIR / "real_tempo_go_c9_pair_local_profile_v10.json"
BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v10.json"
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v10.json"


def main() -> int:
    if any(path.exists() for path in (PROFILE, BASE, CONTRACT)):
        raise RuntimeError("pair-local v10 artifact already exists")

    profile = json.loads(V9_PROFILE.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_pair_local_profile_v10"
    profile["controller"]["business_clean_pair_pressure_fraction"] = 0.5
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(PROFILE, profile)
    loaded = load_global_profile(PROFILE)

    base = json.loads(V9_BASE.read_text(encoding="utf-8"))
    base["candidate"]["id"] = "tempo-go-c9-pair-local-pressure-spill-v1"
    base["purpose"] = (
        "Pressure-aware business pair spill after v9 showed that unconditional "
        "clean-pair packing concentrated protected MISS traffic on one decoder."
    )
    base["joint_control"]["global_profile"] = {
        "path": str(PROFILE.relative_to(ROOT)),
        "sha256": digest(PROFILE),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["joint_control"]["business_pair_spill"] = {
        "schema": "tempo-go-business-pair-spill-v1",
        "pressure_fraction": 0.5,
        "policy_input": "observed_candidate_pair_utilization",
        "phase_label_policy_input": False,
        "future_arrival_policy_input": False,
        "purpose": (
            "Keep higher-priority isolation below current pressure, then permit "
            "the complete global live score to spread work across pairs."
        ),
    }
    base["source_inventory"] = refresh_inventory(base["source_inventory"])
    write_new(BASE, base)

    contract = json.loads(V9_CONTRACT.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "Observer-live ABBA C9 validation with pressure-aware business pair "
        "spill under bounded resident NCCL plus official LMCache/NIXL incast."
    )
    contract["system_under_test"]["base_contract"] = str(BASE.relative_to(ROOT))
    contract["system_under_test"]["base_contract_sha256"] = digest(BASE)
    contract["mechanism"]["schema"] = (
        "tempo-go-c9-pressure-aware-business-spill-mechanism-v1")
    contract["mechanism"]["business_pair_spill"] = {
        "pressure_fraction": 0.5,
        "same_offered_population": True,
        "phase_label_policy_input": False,
        "future_arrival_policy_input": False,
    }
    contract["claim_boundary"]["reason"] = (
        "same-allocation bounded-burst observer and lifecycle validation; "
        "performance claim remains closed until the fresh ABBA passes every gate"
    )
    contract["source_inventory"] = refresh_inventory(contract["source_inventory"])
    write_new(CONTRACT, contract)
    print(CONTRACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
