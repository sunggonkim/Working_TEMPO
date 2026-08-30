#!/usr/bin/env python3
"""Build Candidate O: route-scoped failure isolation on Candidate M.

Candidate M recovered the remote-favorable block, but its native receipt also
showed that a cumulative endpoint failure delta could quarantine both routes
of one decoder pair.  The sibling local path then disappeared from every
later decision even though service-lane capacity failures are not transport
failures.  Candidate O keeps M's pressure spill, business admission, mesh and
shared-fabric controls, while narrowing *telemetry-derived* quarantine to the
failed route.  Explicit request failures remain fail-closed and probe-gated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "results/tempo_go_c9_candidate_m_pressure_spill_v1"
PARENT_PROFILE = PARENT / "real_tempo_go_c9_candidate_m_pressure_spill_profile_v1.json"
PARENT_BASE = PARENT / "tempo_go_c8_candidate_m_pressure_spill_contract.json"
PARENT_POPULATION = PARENT / "tempo_go_c9_candidate_m_pressure_spill_population_contract.json"


def digest(path: Path) -> str:
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
        "--output-dir",
        type=Path,
        default=ROOT / "results/tempo_go_c9_candidate_o_route_liveness_v1",
    )
    args = parser.parse_args()
    for path in (PARENT_PROFILE, PARENT_BASE, PARENT_POPULATION):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    profile_path = output_dir / "real_tempo_go_c9_candidate_o_route_liveness_profile_v1.json"
    base_path = output_dir / "tempo_go_c8_candidate_o_route_liveness_contract.json"
    population_path = output_dir / "tempo_go_c9_candidate_o_route_liveness_population_contract.json"
    if any(path.exists() for path in (profile_path, base_path, population_path)):
        raise FileExistsError("Candidate O output already exists")

    profile = json.loads(PARENT_PROFILE.read_text(encoding="utf-8"))
    profile["profile_id"] = "real_tempo_go_c9_candidate_o_route_liveness_profile_v1"
    controller = dict(profile["controller"])
    if controller.get("telemetry_failure_quarantine_mode") != "deny_until_probe":
        raise ValueError("Candidate M telemetry quarantine mode differs")
    if controller.get("telemetry_failure_quarantine_scope") != "pair":
        raise ValueError("Candidate M telemetry quarantine scope differs")
    controller["telemetry_failure_quarantine_scope"] = "route"
    profile["controller"] = controller
    profile["fingerprint_sha256"] = global_profile_fingerprint(profile)
    write_new(profile_path, profile)
    loaded = load_global_profile(profile_path)
    config = loaded.orchestrator_config()
    if (
        config.telemetry_failure_quarantine_mode != "deny_until_probe"
        or config.telemetry_failure_quarantine_scope != "route"
        or config.route_failure_quarantine_mode != "deny_until_probe"
        or config.business_clean_pair_pressure_fraction != 0.5
    ):
        raise RuntimeError("Candidate O liveness policy did not round-trip")

    base = json.loads(PARENT_BASE.read_text(encoding="utf-8"))
    base["candidate"] = {
        "id": "tempo-go-c9-candidate-o-route-liveness-v1",
        "base_contract": base_path.relative_to(ROOT).as_posix(),
        "parent": PARENT_BASE.relative_to(ROOT).as_posix(),
        "purpose": (
            "retain Candidate M's joint business/fabric policy while isolating "
            "cumulative endpoint failures to the failed route so a healthy "
            "sibling route and surviving P/D pair remain work-conserving"
        ),
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    base["purpose"] = (
        "C8 native base for Candidate O route-scoped failure isolation; all "
        "Candidate M workload, transport, admission, and actuation identities "
        "are inherited."
    )
    base["joint_control"]["global_profile"] = {
        "path": profile_path.relative_to(ROOT).as_posix(),
        "sha256": digest(profile_path),
        "fingerprint_sha256": loaded.fingerprint_sha256,
    }
    base["joint_control"]["failure_isolation"] = {
        "schema": "tempo-go-route-scoped-failure-isolation-v1",
        "explicit_request_failure": "failed_route_deny_until_probe",
        "telemetry_failure_delta": "failed_route_deny_until_probe",
        "service_lane_capacity_failure_is_transport_failure": False,
        "healthy_sibling_route_remains_candidate": True,
        "future_arrival_policy_input": False,
        "phase_label_policy_input": False,
    }
    write_new(base_path, base)

    population = json.loads(PARENT_POPULATION.read_text(encoding="utf-8"))
    for item in population["execution"]["order"]:
        if item.get("arm") == "full_c7_managed_background":
            item["name"] = "06_candidate_o_route_liveness"
    population["candidate"] = {
        "id": "tempo-go-c9-candidate-o-route-liveness-v1",
        "base_contract": base_path.relative_to(ROOT).as_posix(),
        "parent": PARENT_POPULATION.relative_to(ROOT).as_posix(),
        "mechanism": "route_scoped_failure_isolation_plus_pressure_spill",
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    population["purpose"] = (
        "Candidate O C9 same-population discovery against fixed local/remote, "
        "predictor, and queue-GPU under the same native NCCL plus official "
        "LMCache/NIXL receiver-incast burst."
    )
    population["system_under_test"]["base_contract"] = base_path.relative_to(ROOT).as_posix()
    population["system_under_test"]["base_contract_sha256"] = digest(base_path)
    population["system_under_test"]["source_policy"] = (
        "current_source_bound_candidate_o_route_liveness_v1"
    )
    population["provenance"]["base_contract"] = base_path.relative_to(ROOT).as_posix()
    population["provenance"]["base_contract_sha256"] = digest(base_path)
    population["provenance"]["candidate_parent"] = PARENT_POPULATION.relative_to(ROOT).as_posix()
    population["provenance"]["schema"] = (
        "tempo-go-c9-candidate-o-current-source-binding-v1"
    )
    population["provenance"]["failure_isolation"] = {
        "schema": "tempo-go-route-scoped-failure-isolation-v1",
        "parent_scope": "pair",
        "candidate_scope": "route",
        "explicit_route_failure_probe_required": True,
        "same_request_retry": False,
    }
    population["claim_boundary"] = {
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "discovery_only": True,
        "reason": (
            "same-population causal discovery of route-scoped failure "
            "isolation; every native offered-population gate must pass before "
            "a performance claim"
        ),
    }
    current_source_inventory = {
        relative: digest(ROOT / relative)
        for relative in population["provenance"]["source_inventory"]
    }
    population["provenance"]["source_inventory"] = current_source_inventory
    population["source_inventory"] = current_source_inventory
    write_new(population_path, population)
    print(profile_path)
    print(base_path)
    print(population_path)
    print("profile_fingerprint_sha256", loaded.fingerprint_sha256)
    print("population_contract_sha256", digest(population_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
