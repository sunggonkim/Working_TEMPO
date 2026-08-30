#!/usr/bin/env python3
"""Freeze a lower-peak, long-lived observer C9 validation campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


ROOT = Path(__file__).resolve().parents[2]
V5_PROFILE = ROOT / "results/tempo_go_c9_pair_local_campaign_v5/real_tempo_go_c9_pair_local_profile_v5.json"
V5_BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v5.json"
V5_CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v5.json"
OUT_DIR = ROOT / "results/tempo_go_c9_pair_local_campaign_v6"
PROFILE = OUT_DIR / "real_tempo_go_c9_pair_local_profile_v6.json"
BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v6.json"
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v6.json"


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
        raise RuntimeError("pair-local v6 artifact already exists")

    profile = json.loads(V5_PROFILE.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_pair_local_profile_v6"
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(PROFILE, profile)
    loaded = load_global_profile(PROFILE)

    base = json.loads(V5_BASE.read_text(encoding="utf-8"))
    base["candidate"]["id"] = "tempo-go-c9-pair-local-observer-live-v1"
    base["purpose"] = (
        "Observer-live C9 validation of pair-local receiver-aware global "
        "routing under a bounded but sustained native incast workload."
    )
    base["joint_control"]["global_profile"] = {
        "path": str(PROFILE.relative_to(ROOT)),
        "sha256": digest(PROFILE),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["source_inventory"] = refresh_inventory(base["source_inventory"])
    write_new(BASE, base)

    contract = json.loads(V5_CONTRACT.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "Observer-live ABBA C9 validation of pair-local receiver-aware global "
        "routing with a bounded sustained incast that remains valid through "
        "the victim measurement window."
    )
    contract["system_under_test"]["base_contract"] = str(BASE.relative_to(ROOT))
    contract["system_under_test"]["base_contract_sha256"] = digest(BASE)
    contract["claim_boundary"]["reason"] = (
        "same-allocation observer-live pair-local telemetry/controller validation"
    )
    contract["mechanism"]["schema"] = "tempo-go-c9-pair-local-observer-live-mechanism-v1"
    contract["mechanism"]["bootstrap_policy"] = (
        "cojob_bootstrap_then_sustained_active_snapshot_through_victim_window"
    )
    contract["burst"].update({
        "interpretation": (
            "Each block is a bounded NCCL collective plus 512 MiB aggregate "
            "receiver-incast KV burst; the producer remains active through "
            "the victim window without stale-snapshot grace."
        ),
        "requests_per_source": 2,
        "kv_mib_per_request": 64,
        "token_iters": 2048,
        "block_delay_s": 0.1,
        "minimum_active_duration_s": 900,
        "maximum_blocks": 1024,
        "observer_max_age_ms": 60000,
    })
    contract["source_inventory"] = refresh_inventory(contract["source_inventory"])
    write_new(CONTRACT, contract)
    print(CONTRACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
