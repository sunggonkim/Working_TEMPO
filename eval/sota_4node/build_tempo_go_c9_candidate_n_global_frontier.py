#!/usr/bin/env python3
"""Build Candidate N: M pressure spill plus pair-local receiver pricing.

This is a source-bound discovery candidate.  It changes one controller
factor relative to Candidate M: a LOCAL route is charged for supported,
pair-scoped LMCache receiver tail.  The existing M mesh, shared-fabric,
decoder-business, and pressure-spill controls remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


ROOT = Path(__file__).resolve().parents[2]
PARENT_DIR = ROOT / "results/tempo_go_c9_candidate_m_pressure_spill_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/tempo_go_c9_candidate_n_global_frontier_v1",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    profile_path = output_dir / "real_tempo_go_c9_candidate_n_global_frontier_profile_v1.json"
    base_path = output_dir / "tempo_go_c8_candidate_n_global_frontier_contract.json"
    population_path = output_dir / "tempo_go_c9_candidate_n_global_frontier_population_contract.json"

    parent_profile = PARENT_DIR / "real_tempo_go_c9_candidate_m_pressure_spill_profile_v1.json"
    parent_base = PARENT_DIR / "tempo_go_c8_candidate_m_pressure_spill_contract.json"
    parent_population = PARENT_DIR / "tempo_go_c9_candidate_m_pressure_spill_population_contract.json"
    for path in (parent_profile, parent_base, parent_population):
        if not path.is_file():
            raise FileNotFoundError(path)

    profile = json.loads(parent_profile.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_candidate_n_global_frontier_profile_v1"
    profile["controller"]["cross_layer_local_receiver_price_ms"] = 0.10
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(profile_path, profile)
    loaded = load_global_profile(profile_path)

    base = json.loads(parent_base.read_text(encoding="utf-8"))
    base["candidate"] = {
        "base_contract": base_path.relative_to(ROOT).as_posix(),
        "id": "tempo-go-c9-candidate-n-global-frontier-v1",
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "purpose": (
            "compose observed receiver-tail pricing with M pressure spill, "
            "mesh credits, shared-fabric budgets, decoder admission, and "
            "business service accounting"
        ),
    }
    base["purpose"] = (
        "Candidate N global-frontier discovery: use supported pair-local "
        "LMCache receiver tail as an externality price for LOCAL routes while "
        "retaining Candidate M's cross-layer controls."
    )
    base["joint_control"]["global_profile"] = {
        "path": profile_path.relative_to(ROOT).as_posix(),
        "sha256": digest(profile_path),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["joint_control"]["business_pair_spill"]["schema"] = (
        "tempo-go-business-pair-spill-plus-receiver-price-v1"
    )
    base["joint_control"]["receiver_externality_pricing"] = {
        "enabled": True,
        "route": "local",
        "price_ms_per_observed_ms": 0.10,
        "signal": "lmcache_transfer_p99_ms",
        "scope": "pair",
        "supported_only": True,
        "missing_observation_is_not_congestion": True,
    }
    base["source_inventory"] = dict(sorted(base["source_inventory"].items()))
    write_new(base_path, base)

    population = json.loads(parent_population.read_text(encoding="utf-8"))
    population["candidate"] = {
        "base_contract": base_path.relative_to(ROOT).as_posix(),
        "id": "tempo-go-c9-candidate-n-global-frontier-v1",
        "mechanism": "pressure_spill_plus_pair_local_receiver_externality",
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "pressure_fraction": 0.5,
    }
    population["purpose"] = (
        "C9 same-population discovery of Candidate N against fixed local, "
        "fixed remote, predictor, queue-GPU, and Candidate M's predecessor "
        "contract under the same native NCCL plus LMCache burst."
    )
    population["system_under_test"]["base_contract"] = base_path.relative_to(ROOT).as_posix()
    population["system_under_test"]["base_contract_sha256"] = digest(base_path)
    population["system_under_test"]["source_policy"] = (
        "current_source_bound_candidate_n_global_frontier_v1"
    )
    population["provenance"]["base_contract"] = base_path.relative_to(ROOT).as_posix()
    population["provenance"]["base_contract_sha256"] = digest(base_path)
    population["provenance"]["candidate_parent"] = (
        parent_population.relative_to(ROOT).as_posix()
    )
    population["execution"]["order"] = [
        dict(item, name=(
            "06_candidate_n"
            if item.get("arm") == "full_c7_managed_background"
            else item["name"]))
        for item in population["execution"]["order"]
    ]
    write_new(population_path, population)
    print(profile_path)
    print(base_path)
    print(population_path)
    print("profile_fingerprint_sha256", loaded.fingerprint_sha256)
    print("population_contract_sha256", digest(population_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
