#!/usr/bin/env python3
"""Freeze Candidate M: pressure-triggered global pair spill.

Candidate M changes one causal mechanism from the v13 population: a higher
priority request normally prefers a clean pair, but it may spill to a packed
pair once the clean pair's observed service pressure reaches the frozen
fraction.  This is an observed service-envelope rule, not a workload-phase or
future-arrival hint.  All transport, workload, tenant, and comparator
identities remain inherited from the v13 C9 contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROFILE = ROOT / (
    "results/tempo_go_c9_dual_route_business_lane_profile_v12/"
    "real_tempo_go_c9_dual_route_business_lane_profile_v12.json"
)
SOURCE_BASE = ROOT / (
    "results/tempo_go_c9_dual_route_business_lane_v13/"
    "tempo_go_c8_dual_route_business_lane_contract.json"
)
SOURCE_C9 = ROOT / (
    "results/tempo_go_c9_dual_route_business_lane_v13/"
    "tempo_go_c9_dual_route_business_lane_population_contract.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/tempo_go_c9_candidate_m_pressure_spill_v1",
    )
    parser.add_argument(
        "--pressure-fraction", type=float, default=0.5,
        help="observed clean-pair pressure at which packed-pair spill is allowed",
    )
    args = parser.parse_args()
    if not 0.0 < args.pressure_fraction <= 1.0:
        raise ValueError("pressure fraction must be in (0, 1]")
    for path in (SOURCE_PROFILE, SOURCE_BASE, SOURCE_C9):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    profile_path = output_dir / "real_tempo_go_c9_candidate_m_pressure_spill_profile_v1.json"
    base_path = output_dir / "tempo_go_c8_candidate_m_pressure_spill_contract.json"
    c9_path = output_dir / "tempo_go_c9_candidate_m_pressure_spill_population_contract.json"
    if any(path.exists() for path in (profile_path, base_path, c9_path)):
        raise FileExistsError("Candidate M output already exists")

    profile = json.loads(SOURCE_PROFILE.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_candidate_m_pressure_spill_profile_v1"
    controller = dict(profile["controller"])
    controller["business_clean_pair_pressure_fraction"] = args.pressure_fraction
    profile["controller"] = controller
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(profile_path, profile)
    loaded = load_global_profile(profile_path)

    base = json.loads(SOURCE_BASE.read_text(encoding="utf-8"))
    base["candidate"] = {
        "id": "tempo-go-c9-candidate-m-pressure-spill-v1",
        "base_contract": SOURCE_BASE.relative_to(ROOT).as_posix(),
        "purpose": (
            "allow observed service-pressure spill across a packed business "
            "pair while preserving global route, endpoint, fabric, and "
            "tenant admission guards"
        ),
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    base["purpose"] = (
        "C8 native base for Candidate M pressure-triggered business-pair spill; "
        "all v13 workload and transport identities are inherited."
    )
    base["joint_control"]["global_profile"] = {
        "path": profile_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(profile_path),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["joint_control"]["business_pair_spill"] = {
        "schema": "tempo-go-business-pair-spill-v1",
        "pressure_fraction": args.pressure_fraction,
        "policy_input": "observed_candidate_pair_utilization",
        "phase_label_policy_input": False,
        "future_arrival_policy_input": False,
        "purpose": (
            "prefer a clean pair only below its observed pressure envelope; "
            "then let the complete global score use a packed spill pair"
        ),
    }
    write_new(base_path, base)

    contract = json.loads(SOURCE_C9.read_text(encoding="utf-8"))
    # The v13 source contract calls the full arm ``06_candidate_l`` because it
    # inherited Candidate L's display name.  Candidate M must carry its own
    # immutable artifact label; the policy arm itself remains the same full
    # cross-layer controller.
    for item in contract.get("execution", {}).get("order", []):
        if item.get("arm") == "full_c7_managed_background":
            item["name"] = "06_candidate_m_pressure_spill"
    contract["purpose"] = (
        "Candidate M C9 population: compare v13 fixed/predictor/queue-GPU "
        "baselines with pressure-triggered global pair spill under the same "
        "NCCL plus official LMCache/NIXL receiver-incast burst."
    )
    contract["candidate"] = {
        "id": "tempo-go-c9-candidate-m-pressure-spill-v1",
        "base_contract": base_path.relative_to(ROOT).as_posix(),
        "pressure_fraction": args.pressure_fraction,
        "mechanism": "observed_service_pressure_triggered_pair_spill",
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    contract["system_under_test"]["base_contract"] = (
        base_path.relative_to(ROOT).as_posix())
    contract["system_under_test"]["base_contract_sha256"] = sha256(base_path)
    contract["system_under_test"]["source_policy"] = (
        "current_source_bound_candidate_m_pressure_spill_v1")
    contract["provenance"]["base_contract"] = (
        base_path.relative_to(ROOT).as_posix())
    contract["provenance"]["base_contract_sha256"] = sha256(base_path)
    contract["provenance"]["schema"] = (
        "tempo-go-c9-candidate-m-current-source-binding-v1")
    contract["provenance"]["business_pair_spill"] = {
        "schema": "tempo-go-business-pair-spill-v1",
        "pressure_fraction": args.pressure_fraction,
        "source_of_pressure": "live_pair_telemetry_and_global_owned_work",
        "phase_label_policy_input": False,
        "future_arrival_policy_input": False,
        "cache_affinity_override": False,
    }
    contract["claim_boundary"] = {
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "discovery_only": True,
        "reason": (
            "same-population causal discovery of pressure-triggered pair spill; "
            "fresh allocation and every offered-population gate are required "
            "before any paper claim"
        ),
    }
    contract["source_inventory"] = {
        relative: sha256(ROOT / relative)
        for relative in contract["provenance"]["source_inventory"]
    }
    write_new(c9_path, contract)
    print(profile_path)
    print(base_path)
    print(c9_path)
    print("profile_sha256", sha256(profile_path))
    print("profile_fingerprint", loaded.fingerprint_sha256)
    print("base_sha256", sha256(base_path))
    print("c9_sha256", sha256(c9_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
