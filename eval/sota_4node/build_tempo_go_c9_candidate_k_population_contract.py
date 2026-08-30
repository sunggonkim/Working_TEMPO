#!/usr/bin/env python3
"""Build the Candidate K C9 contract with fixed/predictor comparators."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / (
    "results/tempo_go_c9_candidate_k_protected_lane_v1/"
    "tempo_go_c9_candidate_k_contract.json"
)
OUTPUT = ROOT / (
    "results/tempo_go_c9_candidate_k_protected_lane_v2/"
    "tempo_go_c9_candidate_k_population_contract.json"
)
DEFAULT_BASE = ROOT / (
    "results/tempo_go_c9_candidate_k_protected_lane_v1/"
    "tempo_go_c8_candidate_k_contract.json"
)

SOURCES = (
    "eval/sota_4node/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh",
    "eval/sota_4node/run_lmcache_nixl_contention_2node_in_allocation.sh",
    "eval/sota_4node/c9_gate_node_entry.sh",
    "eval/sota_4node/vllm_lmcache_tempo_go_c9_gate_node.py",
    "eval/sota_4node/analyze_tempo_go_c9_causal_burst_discovery.py",
    "eval/sota_4node/run_tempo_go_c8_dual_regime_client.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--base-contract", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    output = args.output.resolve()
    base_contract = args.base_contract.resolve()
    if not TEMPLATE.is_file():
        raise FileNotFoundError(TEMPLATE)
    if output.exists():
        raise FileExistsError(output)
    contract = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    order = (
        ("00_fixed_local_d0", "fixed_local_d0"),
        ("01_fixed_local_d1", "fixed_local_d1"),
        ("02_fixed_remote_p0d1", "fixed_remote_p0d1"),
        ("03_fixed_remote_p1d0", "fixed_remote_p1d0"),
        ("04_predictor", "predictor"),
        ("05_queue_gpu", "queue_gpu"),
        ("06_candidate_k", "full_c7_managed_background"),
    )
    contract["purpose"] = (
        "Candidate K population C9: compare fixed local/remote routes, the "
        "simple request predictor, queue-GPU baseline, and protected-lane "
        "TEMPO under one identical NCCL plus official LMCache/NIXL burst."
    )
    contract["burst"]["interpretation"] = (
        "Each block is a high-peak NCCL collective plus 2 GiB aggregate "
        "receiver-incast KV burst, followed by a 100 ms gap; the train is "
        "exogenous and identical in all population arms."
    )
    contract["burst"]["cojob_pair_count"] = 2
    contract["system_under_test"]["observer_scope"] = (
        "both physical P/D pairs, each with a real 8-rank NCCL and four-source "
        "NIXL receiver-incast co-job"
    )
    contract["execution"]["order"] = [
        {"name": name, "arm": arm, "port_slot": 2420 + index * 40}
        for index, (name, arm) in enumerate(order)
    ]
    contract["execution"]["paired_indices"] = []
    contract["gates"].update({
        "minimum_full_supported_observer_fraction": 0.5,
        "minimum_stressed_p99_reduction_fraction": 0.15,
        "minimum_stressed_slo_good_ratio": 1.0,
        "maximum_normal_p50_regression_fraction": 0.03,
        "require_full_cross_layer_actuation": True,
        "require_baseline_cross_layer_blind": False,
    })
    contract["claim_boundary"]["reason"] = (
        "The same native burst is compared across fixed, predictor, queue-GPU, "
        "and Candidate K arms; this remains discovery until all population "
        "gates pass on a fresh allocation."
    )
    system = contract["system_under_test"]
    if ROOT not in base_contract.parents:
        raise ValueError("base contract must be inside repository")
    if not base_contract.is_file():
        raise FileNotFoundError(base_contract)
    system["base_contract"] = base_contract.relative_to(ROOT).as_posix()
    contract["provenance"]["base_contract"] = (
        base_contract.relative_to(ROOT).as_posix())
    base = base_contract
    system["base_contract_sha256"] = sha256(base)
    contract["provenance"]["base_contract_sha256"] = sha256(base)
    base_value = json.loads(base.read_text(encoding="utf-8"))
    contract["provenance"]["base_source_inventory_count"] = len(
        base_value.get("source_inventory", {}))
    contract["provenance"]["source_inventory"] = {
        relative: sha256(ROOT / relative) for relative in SOURCES
    }
    contract["provenance"]["c9_source_inventory_count"] = len(SOURCES)
    contract["provenance"]["population_comparison"] = {
        "fixed_arms": [arm for _name, arm in order if arm.startswith("fixed_")],
        "predictor_arm": "predictor",
        "queue_gpu_arm": "queue_gpu",
        "tempo_arm": "full_c7_managed_background",
        "same_offered_population_required": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print("contract_sha256", sha256(output))
    print("source_inventory_count", len(SOURCES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
