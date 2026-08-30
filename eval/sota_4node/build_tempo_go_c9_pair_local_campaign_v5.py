#!/usr/bin/env python3
"""Version the observer-gated C9 campaign after the bootstrap-gate fix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


ROOT = Path(__file__).resolve().parents[2]
V4_PROFILE = ROOT / "results/tempo_go_c9_pair_local_campaign_v4/real_tempo_go_c9_pair_local_profile_v4.json"
V4_BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v4.json"
V4_CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v4.json"
OUT_DIR = ROOT / "results/tempo_go_c9_pair_local_campaign_v5"
PROFILE = OUT_DIR / "real_tempo_go_c9_pair_local_profile_v5.json"
BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v5.json"
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v5.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def refresh_inventory(inventory: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in inventory:
        source = (ROOT / relative).resolve()
        if not source.is_file() or ROOT not in source.parents:
            raise RuntimeError(f"source inventory entry is missing: {relative}")
        result[relative] = digest(source)
    return dict(sorted(result.items()))


def main() -> int:
    if any(path.exists() for path in (PROFILE, BASE, CONTRACT)):
        raise RuntimeError("pair-local v5 artifact already exists")

    profile = json.loads(V4_PROFILE.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_pair_local_profile_v5"
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(PROFILE, profile)
    loaded = load_global_profile(PROFILE)

    base = json.loads(V4_BASE.read_text(encoding="utf-8"))
    base["candidate"]["id"] = "tempo-go-c9-pair-local-receiver-price-v4-bootstrap-gated"
    base["purpose"] = (
        "Bootstrap-gated C9 discovery of pair-local receiver externality "
        "pricing under the same actual NCCL plus official LMCache receiver-incast burst."
    )
    base["joint_control"]["global_profile"] = {
        "path": str(PROFILE.relative_to(ROOT)),
        "sha256": digest(PROFILE),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["source_inventory"] = refresh_inventory(base["source_inventory"])
    write_new(BASE, base)

    contract = json.loads(V4_CONTRACT.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "Bootstrap-gated ABBA C9 discovery of pair-local receiver-aware global "
        "routing under the same actual NCCL plus official LMCache receiver-incast burst."
    )
    contract["system_under_test"]["base_contract"] = str(BASE.relative_to(ROOT))
    contract["system_under_test"]["base_contract_sha256"] = digest(BASE)
    contract["claim_boundary"]["reason"] = (
        "same-allocation bootstrap-gated pair-local telemetry/controller discovery"
    )
    contract["mechanism"]["schema"] = "tempo-go-c9-pair-local-receiver-mechanism-v3"
    contract["mechanism"]["bootstrap_policy"] = (
        "cojob_runs_before_victim_start_file; victim_release_waits_for_active_snapshot"
    )
    contract["source_inventory"] = refresh_inventory(contract["source_inventory"])
    write_new(CONTRACT, contract)
    print(CONTRACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
