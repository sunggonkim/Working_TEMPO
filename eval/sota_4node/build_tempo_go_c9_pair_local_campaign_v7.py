#!/usr/bin/env python3
"""Freeze a lower-intensity C9 observer-live boundary campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


ROOT = Path(__file__).resolve().parents[2]
V6_PROFILE = ROOT / "results/tempo_go_c9_pair_local_campaign_v6/real_tempo_go_c9_pair_local_profile_v6.json"
V6_BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v6.json"
V6_CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v6.json"
OUT_DIR = ROOT / "results/tempo_go_c9_pair_local_campaign_v7"
PROFILE = OUT_DIR / "real_tempo_go_c9_pair_local_profile_v7.json"
BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v7.json"
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v7.json"


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
        raise RuntimeError("pair-local v7 artifact already exists")

    profile = json.loads(V6_PROFILE.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_pair_local_profile_v7"
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(PROFILE, profile)
    loaded = load_global_profile(PROFILE)

    base = json.loads(V6_BASE.read_text(encoding="utf-8"))
    base["candidate"]["id"] = "tempo-go-c9-pair-local-observer-live-v2"
    base["purpose"] = (
        "Lower-intensity observer-live C9 validation of pair-local receiver-aware "
        "global routing under a sustained native incast workload."
    )
    base["joint_control"]["global_profile"] = {
        "path": str(PROFILE.relative_to(ROOT)),
        "sha256": digest(PROFILE),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["source_inventory"] = refresh_inventory(base["source_inventory"])
    write_new(BASE, base)

    contract = json.loads(V6_CONTRACT.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "Lower-intensity observer-live ABBA C9 validation where the native incast "
        "producer remains active while every victim block completes."
    )
    contract["system_under_test"]["base_contract"] = str(BASE.relative_to(ROOT))
    contract["system_under_test"]["base_contract_sha256"] = digest(BASE)
    contract["claim_boundary"]["reason"] = (
        "same-allocation observer-live pair-local telemetry/controller boundary validation"
    )
    contract["mechanism"]["schema"] = "tempo-go-c9-pair-local-observer-live-mechanism-v2"
    contract["burst"].update({
        "interpretation": (
            "Each block is a bounded NCCL collective plus 128 MiB aggregate "
            "receiver-incast KV burst; the producer remains active through "
            "the victim window without stale-snapshot grace."
        ),
        "requests_per_source": 1,
        "kv_mib_per_request": 32,
        "token_iters": 512,
        "block_delay_s": 0.1,
        "minimum_active_duration_s": 900,
        "maximum_blocks": 2048,
        "observer_max_age_ms": 60000,
    })
    contract["source_inventory"] = refresh_inventory(contract["source_inventory"])
    write_new(CONTRACT, contract)
    print(CONTRACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
