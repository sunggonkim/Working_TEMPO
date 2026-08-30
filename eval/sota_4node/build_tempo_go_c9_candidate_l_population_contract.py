#!/usr/bin/env python3
"""Freeze the Candidate L C9 population contract.

Candidate L keeps Candidate K's native workload and comparator population, but
binds a new C8 base contract whose TEMPO profile uses the v2 protected-lane
reserve semantics.  K artifacts are immutable and are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / (
    "results/tempo_go_c9_candidate_k_protected_lane_v9/"
    "tempo_go_c9_candidate_k_population_contract.json"
)
DEFAULT_BASE = ROOT / (
    "results/tempo_go_c9_candidate_l_protected_reserve_v1/"
    "tempo_go_c8_candidate_l_contract.json"
)
DEFAULT_OUTPUT = ROOT / (
    "results/tempo_go_c9_candidate_l_protected_reserve_v1/"
    "tempo_go_c9_candidate_l_population_contract.json"
)

SOURCES = (
    "eval/sota_4node/require_perlmutter_4node_4h_interactive.sh",
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
    parser.add_argument("--base-contract", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not TEMPLATE.is_file():
        raise FileNotFoundError(TEMPLATE)
    base = args.base_contract.resolve()
    output = args.output.resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    if output.exists():
        raise FileExistsError(output)

    contract = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    contract["purpose"] = (
        "Candidate L population C9: compare fixed local/remote routes, the "
        "simple request predictor, queue-GPU baseline, and protected-reserve "
        "TEMPO under one identical NCCL plus official LMCache/NIXL burst."
    )
    contract["system_under_test"].update({
        "base_contract": base.relative_to(ROOT).as_posix(),
        "base_contract_sha256": sha256(base),
        "source_policy": "current_source_bound_candidate_l_v1",
    })
    contract["provenance"].update({
        "base_contract": base.relative_to(ROOT).as_posix(),
        "base_contract_sha256": sha256(base),
        "schema": "tempo-go-c9-candidate-l-current-source-binding-v1",
        "source_inventory": {
            relative: sha256(ROOT / relative) for relative in SOURCES
        },
        "c9_source_inventory_count": len(SOURCES),
    })
    contract["claim_boundary"]["reason"] = (
        "The same native burst is compared across fixed, predictor, queue-GPU, "
        "and Candidate L arms; this remains discovery until all population "
        "gates pass on a fresh allocation."
    )
    contract["execution"]["order"] = [
        dict(item, name=(
            "06_candidate_l"
            if item.get("arm") == "full_c7_managed_background"
            else item["name"]
        ))
        for item in contract["execution"]["order"]
    ]
    # The seven arms run sequentially while each arm owns all four nodes.
    # Record the lower bound used by the launcher to reject a short outer step
    # before starting any GPU child. This is an execution-integrity guard.
    contract["execution"]["minimum_outer_time_s"] = 9000
    contract["execution"]["minimum_outer_time_reason"] = (
        "seven-arm sequential native population requires a 150-minute outer "
        "step; fail closed before GPU launch when the parent budget is shorter"
    )
    contract["candidate"] = {
        "id": "tempo-go-c9-candidate-l-protected-reserve-v1",
        "base_contract": base.relative_to(ROOT).as_posix(),
        "purpose": (
            "separate protected service reserve/floor from protected request "
            "concurrency while retaining physical endpoint, fabric, and "
            "deadline guards"
        ),
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
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
