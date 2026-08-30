#!/usr/bin/env python3
"""Freeze C9 v9 with bounded resident cross-layer burst pressure."""

from __future__ import annotations

import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile
from eval.sota_4node.build_tempo_go_c9_pair_local_campaign_v7 import (
    digest, refresh_inventory, write_new,
)


ROOT = Path(__file__).resolve().parents[2]
V8_PROFILE = ROOT / "results/tempo_go_c9_pair_local_campaign_v8/real_tempo_go_c9_pair_local_profile_v8.json"
V8_BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v8.json"
V8_CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v8.json"
OUT_DIR = ROOT / "results/tempo_go_c9_pair_local_campaign_v9"
PROFILE = OUT_DIR / "real_tempo_go_c9_pair_local_profile_v9.json"
BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v9.json"
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v9.json"


def main() -> int:
    if any(path.exists() for path in (PROFILE, BASE, CONTRACT)):
        raise RuntimeError("pair-local v9 artifact already exists")

    profile = json.loads(V8_PROFILE.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_pair_local_profile_v9"
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(PROFILE, profile)
    loaded = load_global_profile(PROFILE)

    base = json.loads(V8_BASE.read_text(encoding="utf-8"))
    base["candidate"]["id"] = "tempo-go-c9-pair-local-bounded-burst-v1"
    base["purpose"] = (
        "Observer-live C9 validation with bounded resident native incast "
        "pressure so endpoint/cache failure is measured rather than used as "
        "the only terminal outcome."
    )
    base["joint_control"]["global_profile"] = {
        "path": str(PROFILE.relative_to(ROOT)),
        "sha256": digest(PROFILE),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["source_inventory"] = refresh_inventory(base["source_inventory"])
    write_new(BASE, base)

    contract = json.loads(V8_CONTRACT.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "Observer-live ABBA C9 validation under a bounded resident NCCL plus "
        "official LMCache/NIXL incast train."
    )
    contract["system_under_test"]["base_contract"] = str(BASE.relative_to(ROOT))
    contract["system_under_test"]["base_contract_sha256"] = digest(BASE)
    contract["claim_boundary"]["reason"] = (
        "same-allocation bounded-burst observer and lifecycle validation; "
        "performance claim remains closed until every arm is complete"
    )
    contract["mechanism"]["schema"] = "tempo-go-c9-pair-local-bounded-burst-mechanism-v1"
    contract["mechanism"]["bootstrap_policy"] = (
        "cojob_bootstrap_then_active_snapshot_with_bounded_resident_burst"
    )
    contract["burst"].update({
        "requests_per_source": 1,
        "kv_mib_per_request": 8,
        "token_iters": 256,
        "foreground_mib": 16,
        "minimum_active_duration_s": 600,
        "maximum_blocks": 2048,
        "block_delay_s": 0.25,
        "observer_max_age_ms": 60000,
        "interpretation": (
            "Each block is a bounded NCCL collective plus 32 MiB aggregate "
            "receiver-incast KV burst. The fixed-size resident buffer and "
            "moderate block gap keep the observer live while avoiding an "
            "unbounded co-job allocation train; victim LMCache pressure and "
            "business admission remain unchanged."
        ),
    })
    contract["source_inventory"] = refresh_inventory(contract["source_inventory"])
    write_new(CONTRACT, contract)
    print(CONTRACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
