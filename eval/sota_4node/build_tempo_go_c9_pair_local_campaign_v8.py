#!/usr/bin/env python3
"""Freeze C9 v8 with a producer lease longer than victim finalization."""

from __future__ import annotations

import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile
from eval.sota_4node.build_tempo_go_c9_pair_local_campaign_v7 import (
    digest, refresh_inventory, write_new,
)


ROOT = Path(__file__).resolve().parents[2]
V7_PROFILE = ROOT / "results/tempo_go_c9_pair_local_campaign_v7/real_tempo_go_c9_pair_local_profile_v7.json"
V7_BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v7.json"
V7_CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v7.json"
OUT_DIR = ROOT / "results/tempo_go_c9_pair_local_campaign_v8"
PROFILE = OUT_DIR / "real_tempo_go_c9_pair_local_profile_v8.json"
BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v8.json"
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v8.json"


def main() -> int:
    if any(path.exists() for path in (PROFILE, BASE, CONTRACT)):
        raise RuntimeError("pair-local v8 artifact already exists")

    profile = json.loads(V7_PROFILE.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_pair_local_profile_v8"
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(PROFILE, profile)
    loaded = load_global_profile(PROFILE)

    base = json.loads(V7_BASE.read_text(encoding="utf-8"))
    base["candidate"]["id"] = "tempo-go-c9-pair-local-observer-live-v3"
    base["purpose"] = (
        "Observer-live C9 validation with a producer lease extending beyond "
        "victim finalization under sustained native incast."
    )
    base["joint_control"]["global_profile"] = {
        "path": str(PROFILE.relative_to(ROOT)),
        "sha256": digest(PROFILE),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["source_inventory"] = refresh_inventory(base["source_inventory"])
    write_new(BASE, base)

    contract = json.loads(V7_CONTRACT.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "Observer-live ABBA C9 validation with a producer lease that remains "
        "active until the victim inference lifecycle requests shutdown."
    )
    contract["system_under_test"]["base_contract"] = str(BASE.relative_to(ROOT))
    contract["system_under_test"]["base_contract_sha256"] = digest(BASE)
    contract["claim_boundary"]["reason"] = (
        "same-allocation observer-live pair-local telemetry/controller validation"
    )
    contract["mechanism"]["schema"] = "tempo-go-c9-pair-local-observer-live-mechanism-v3"
    contract["mechanism"]["bootstrap_policy"] = (
        "cojob_bootstrap_then_active_snapshot_until_victim_finalization"
    )
    contract["burst"]["minimum_active_duration_s"] = 1500
    contract["burst"]["interpretation"] = (
        "Each block is a bounded NCCL collective plus 128 MiB aggregate "
        "receiver-incast KV burst; producer lease exceeds victim finalization "
        "and is stopped only by the C9 launcher after inference completion."
    )
    contract["source_inventory"] = refresh_inventory(contract["source_inventory"])
    write_new(CONTRACT, contract)
    print(CONTRACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
