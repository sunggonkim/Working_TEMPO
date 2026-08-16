#!/usr/bin/env python3
"""Non-submitting NVLink/P2P attribution plan.

Checkpoint D2H evidence must not be relabeled as NVLink evidence.  This
module freezes a separate one-node P2P experiment contract; it contains no
Slurm submission or CUDA execution.  A future live result must add observed
topology/path records and monotonic NVLink TX/RX byte counters before any
causal claim is accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tempo.domain_evidence import CounterSupport, PathStatus
from tempo.resource_domain import ResourceDomain, domain_contract


SCHEMA = "tempo-rd-nvlink-p2p-plan-1"
# Perlmutter's four-GPU A100 node exposes an NVLink path between each GPU
# pair. Keep both directions because source/destination roles are asymmetric;
# live evidence must still prove each path and counter rather than trusting
# this inventory.
P2P_PAIRS = [[source, destination] for source in range(4) for destination in range(4) if source != destination]
TRANSFER_SIZES = [4 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024]


def build_nvlink_p2p_plan() -> Dict[str, Any]:
    """Return the frozen design-only P2P matrix."""

    nvlink = domain_contract(ResourceDomain.NVLINK_P2P)
    gpu = domain_contract(ResourceDomain.GPU_LOCAL)
    plan = {
        "schema_version": SCHEMA,
        "stage": "g1_nvlink_p2p",
        "evidence_state": "design_only",
        "nodes": 1,
        "world_size": 4,
        "pairs": P2P_PAIRS,
        "pair_contract": {
            "ordered_pairs": True,
            "all_distinct_gpu_pairs": True,
            "required_path_record_per_pair": True,
            "required_counter_record_per_pair": True,
        },
        "transfer_sizes_bytes": TRANSFER_SIZES,
        "warmup_iterations": 5,
        "measured_iterations": 20,
        "foreground": {
            "operation": "synchronous_fsdp_collective",
            "tail_metric": "complete_group_slowest_rank_p99_ns",
            "skew_metric": "complete_group_arrival_skew_p99_ns",
        },
        "required_domains": [ResourceDomain.GPU_LOCAL.value, ResourceDomain.NVLINK_P2P.value],
        "path_contract": {
            "gpu_local": {
                "path_evidence": gpu.path_evidence,
                "counter_family": gpu.counter_family,
                "path_status": "not_traversed",
                "counter_support": "not_collected",
            },
            "nvlink_p2p": {
                "path_evidence": nvlink.path_evidence,
                "counter_family": nvlink.counter_family,
                "path_status": "not_traversed",
                "counter_support": "not_collected",
            },
        },
        "promotion_requires": {
            "path_status": PathStatus.OBSERVED.value,
            "counter_support": CounterSupport.SUPPORTED.value,
            "counter_samples": "monotonic_timestamp_cumulative_bytes_and_busy_ns",
            "intervention": "p2p_enabled_vs_matched_open",
            "pairwise_path_and_counter_coverage": "all_ordered_distinct_pairs",
        },
        "required_artifacts": [
            "topology_matrix.txt",
            "cuda_p2p_path_records.json",
            "domain_counters/gpu_local.json",
            "domain_counters/nvlink_p2p.json",
            "foreground_tail_records.json",
            "matched_open_tail_records.json",
        ],
        "slurm_submitted": False,
        "inference_adapter": "not_implemented_in_p2p_plan",
    }
    validate_nvlink_p2p_plan(plan)
    return plan


def validate_nvlink_p2p_plan(plan: Dict[str, Any]) -> None:
    expected = {
        "schema_version", "stage", "evidence_state", "nodes", "world_size",
        "pairs", "pair_contract", "transfer_sizes_bytes", "warmup_iterations", "measured_iterations",
        "foreground", "required_domains", "path_contract", "required_artifacts",
        "promotion_requires", "slurm_submitted", "inference_adapter",
    }
    if type(plan) is not dict or set(plan) != expected:
        raise ValueError("NVLink P2P plan keys are not exact")
    if plan["schema_version"] != SCHEMA or plan["stage"] != "g1_nvlink_p2p":
        raise ValueError("unsupported NVLink P2P plan")
    if plan["evidence_state"] != "design_only" or plan["slurm_submitted"] is not False:
        raise ValueError("NVLink P2P plan must remain non-submitting design_only")
    if plan["nodes"] != 1 or plan["world_size"] != 4:
        raise ValueError("NVLink P2P plan must be one node/four ranks")
    if plan["pairs"] != P2P_PAIRS:
        raise ValueError("NVLink P2P pair matrix is not frozen")
    if plan["pair_contract"] != {
        "ordered_pairs": True,
        "all_distinct_gpu_pairs": True,
        "required_path_record_per_pair": True,
        "required_counter_record_per_pair": True,
    }:
        raise ValueError("NVLink P2P pair coverage contract is not exact")
    if plan["transfer_sizes_bytes"] != TRANSFER_SIZES:
        raise ValueError("NVLink P2P transfer sizes are not frozen")
    if type(plan["warmup_iterations"]) is not int or plan["warmup_iterations"] <= 0:
        raise ValueError("warmup_iterations must be positive")
    if type(plan["measured_iterations"]) is not int or plan["measured_iterations"] < 10:
        raise ValueError("measured_iterations must be at least ten")
    foreground = plan["foreground"]
    if type(foreground) is not dict or set(foreground) != {"operation", "tail_metric", "skew_metric"}:
        raise ValueError("foreground metric contract is not exact")
    if foreground["operation"] != "synchronous_fsdp_collective":
        raise ValueError("foreground operation is not the registered collective")
    if plan["required_domains"] != ["gpu_local", "nvlink_p2p"]:
        raise ValueError("NVLink P2P required domain set is not exact")
    contract = plan["path_contract"]
    if type(contract) is not dict or set(contract) != {"gpu_local", "nvlink_p2p"}:
        raise ValueError("NVLink P2P path contract is not exact")
    for name in ("gpu_local", "nvlink_p2p"):
        domain = ResourceDomain(name)
        expected_contract = domain_contract(domain)
        record = contract[name]
        if type(record) is not dict or set(record) != {
            "path_evidence", "counter_family", "path_status", "counter_support"
        }:
            raise ValueError("NVLink P2P path record is not exact")
        if record != {
            "path_evidence": expected_contract.path_evidence,
            "counter_family": expected_contract.counter_family,
            "path_status": "not_traversed",
            "counter_support": "not_collected",
        }:
            raise ValueError("NVLink P2P path/counter labels are not exact")
    if plan["promotion_requires"] != {
        "path_status": PathStatus.OBSERVED.value,
        "counter_support": CounterSupport.SUPPORTED.value,
        "counter_samples": "monotonic_timestamp_cumulative_bytes_and_busy_ns",
        "intervention": "p2p_enabled_vs_matched_open",
        "pairwise_path_and_counter_coverage": "all_ordered_distinct_pairs",
    }:
        raise ValueError("NVLink P2P promotion contract is not exact")
    artifacts = plan["required_artifacts"]
    if type(artifacts) is not list or any(type(item) is not str for item in artifacts) or len(set(artifacts)) != len(artifacts):
        raise ValueError("NVLink P2P artifact list is invalid")
    if plan["inference_adapter"] != "not_implemented_in_p2p_plan":
        raise ValueError("NVLink P2P inference marker changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(build_nvlink_p2p_plan(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
