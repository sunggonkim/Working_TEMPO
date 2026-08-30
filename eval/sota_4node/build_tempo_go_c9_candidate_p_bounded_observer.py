#!/usr/bin/env python3
"""Freeze Candidate P: Candidate O policy under the v10 bounded observer load.

Candidate O's 2 GiB-per-pair incast train repeatedly reached the official
LMCache/NIXL 60 s timeout.  The co-load process was also the observer producer,
so later policy decisions lost fresh NCCL/LMCache evidence.  Earlier v9/v10
receipts established a bounded resident shape that kept the observer alive
without removing actual NCCL/Slingshot and LMCache/NIXL contention.

This builder changes no controller threshold or route policy.  It binds the
current Candidate O base contract to that preregistered bounded load, both
physical P/D pairs, the current analyzer, and a strict 100% victim-observer
coverage gate.  It is a discovery/reproduction contract, not an independent
validation claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT_DIR = ROOT / "results/tempo_go_c9_candidate_o_route_liveness_v1"
PARENT_POPULATION = (
    PARENT_DIR / "tempo_go_c9_candidate_o_route_liveness_population_contract.json"
)
PARENT_BASE = PARENT_DIR / "tempo_go_c8_candidate_o_route_liveness_contract.json"


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
        default=ROOT / "results/tempo_go_c9_candidate_p_bounded_observer_v1",
    )
    args = parser.parse_args()
    for path in (PARENT_POPULATION, PARENT_BASE):
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output_dir.resolve()
    contract_path = output / "tempo_go_c9_candidate_p_bounded_observer_contract.json"
    if contract_path.exists():
        raise FileExistsError(contract_path)

    contract = json.loads(PARENT_POPULATION.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "Current-source seven-arm reproduction of Candidate O under the "
        "bounded-resident dual-pair NCCL plus official LMCache/NIXL incast "
        "shape that kept the observer live in the historical v9/v10 campaign."
    )
    contract["candidate"] = {
        "id": "tempo-go-c9-candidate-p-bounded-observer-v1",
        "base_contract": PARENT_BASE.relative_to(ROOT).as_posix(),
        "parent": PARENT_POPULATION.relative_to(ROOT).as_posix(),
        "policy_delta_from_candidate_o": "none",
        "experimental_delta": "bounded_resident_dual_pair_observer_lifecycle",
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    contract["burst"] = {
        "block_delay_s": 0.25,
        "cojob_pair_count": 2,
        "cojob_time_limit": "00:29:00",
        "cojob_timeout_s": 1740,
        "foreground_mib": 16,
        "interpretation": (
            "Each physical pair runs a bounded 4-source x 8 MiB receiver-incast "
            "block plus native NCCL collective. The resident buffers are reused "
            "and blocks are spaced by 250 ms, preserving contention while "
            "avoiding the 2 GiB timeout-shaped observer-lifetime confound."
        ),
        "kv_mib_per_request": 8,
        "maximum_blocks": 2048,
        "minimum_active_duration_s": 600,
        "minimum_blocks": 1,
        "nixl_transfer_timeout_s": 60,
        "observer_max_age_ms": 60000,
        "process_group_timeout_s": 120,
        "requests_per_source": 1,
        "start_delay_s": 0,
        "token_iters": 256,
        "traffic_pattern": "incast_4to1",
        "transport": {
            "lmcache": "official NixlChannel with UCX",
            "nccl": "AWS Libfabric over Slingshot/CXI",
            "socket_fallback_forbidden": True,
        },
    }
    for item in contract["execution"]["order"]:
        if item.get("arm") == "full_c7_managed_background":
            item["name"] = "06_candidate_p_bounded_observer"
    contract["execution"]["minimum_outer_time_s"] = 9000
    contract["execution"]["one_campaign_no_retry"] = True
    contract["execution"]["paired_indices"] = []
    contract["gates"]["minimum_full_supported_observer_fraction"] = 1.0
    contract["gates"]["require_all_full_decisions_observer_supported"] = True
    contract["system_under_test"]["base_contract"] = (
        PARENT_BASE.relative_to(ROOT).as_posix()
    )
    contract["system_under_test"]["base_contract_sha256"] = digest(PARENT_BASE)
    contract["system_under_test"]["source_policy"] = (
        "current_source_bound_candidate_p_bounded_observer_v1"
    )
    contract["system_under_test"]["observer_scope"] = (
        "both physical P/D pairs; each pair runs one native 8-rank NCCL and "
        "official LMCache/NIXL-UCX bounded receiver-incast producer"
    )
    contract["claim_boundary"] = {
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "discovery_only": True,
        "reason": (
            "current-source reproduction of the historical v10 load boundary; "
            "all offered-population, terminal, observer, and performance gates "
            "must pass before promotion to a held-out validation"
        ),
    }
    contract["provenance"]["schema"] = (
        "tempo-go-c9-candidate-p-bounded-observer-current-source-v1"
    )
    contract["provenance"]["candidate_parent"] = (
        PARENT_POPULATION.relative_to(ROOT).as_posix()
    )
    contract["provenance"]["bounded_resident_observer"] = {
        "schema": "tempo-go-bounded-resident-observer-lifecycle-v1",
        "historical_reference_contract": (
            "eval/sota_4node/tempo_go_c9_pair_local_contract_v10.json"
        ),
        "historical_reference_contract_sha256": digest(
            ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v10.json"
        ),
        "policy_delta_from_candidate_o": False,
        "same_offered_population": True,
        "both_physical_pairs_observed": True,
        "stale_max_age_relaxed": False,
    }

    source_paths = set(contract["provenance"]["source_inventory"])
    source_paths.update({
        "eval/sota_4node/run_lmcache_nixl_contention_2node.py",
        "eval/sota_4node/run_lmcache_nixl_contention_2node_in_allocation.sh",
        "tempo/cross_layer_observer.py",
    })
    source_inventory = {
        relative: digest(ROOT / relative)
        for relative in sorted(source_paths)
    }
    contract["provenance"]["source_inventory"] = source_inventory
    contract["source_inventory"] = source_inventory
    write_new(contract_path, contract)
    print(contract_path)
    print("contract_sha256", digest(contract_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
