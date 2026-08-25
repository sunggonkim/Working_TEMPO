#!/usr/bin/env python3
"""Freeze the one-shot, held-out C8 independent-validation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASE_SCHEMA = "tempo-go-c8-dual-regime-contract-v1"
INDEPENDENT_SCHEMA = "tempo-go-c8-independent-validation-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--parent-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-seed", type=int, required=True)
    parser.add_argument("--preregistered-at-utc", required=True)
    parser.add_argument("--forbid-job-id", action="append", required=True)
    parser.add_argument(
        "--prior-failure-receipt", type=Path, action="append", default=[])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    parent_path = args.parent_contract.resolve()
    parent_analysis_path = args.parent_analysis.resolve()
    output = args.output.resolve()
    _require(parent_path.is_file(), "parent C8 contract is missing")
    _require(parent_analysis_path.is_file(), "parent C8 analysis is missing")
    _require(not output.exists(), "refusing to overwrite held-out contract")
    _require(args.request_seed >= 0, "request seed must be nonnegative")
    _require(len(set(args.forbid_job_id)) == len(args.forbid_job_id)
             and all(str(value).isdigit() for value in args.forbid_job_id),
             "forbidden discovery job IDs are invalid")
    prior_failures = []
    for receipt_arg in args.prior_failure_receipt:
        receipt_path = receipt_arg.resolve()
        _require(
            receipt_path.is_file() and repo_root in receipt_path.parents,
            "prior independent failure receipt is outside the repository",
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _require(
            receipt.get("schema") == "tempo-go-c8-independent-failure-v1"
            and receipt.get("terminal") is False
            and receipt.get("retry_allowed") is False,
            "prior independent failure receipt is invalid",
        )
        prior_failures.append({
            "path": receipt_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256(receipt_path),
            "slurm_job_id": str(receipt["slurm_job_id"]),
            "failed_arm": str(receipt["failed_arm"]),
            "contract_sha256": str(receipt["contract_sha256"]),
            "performance_result": False,
        })

    raw = json.loads(parent_path.read_text(encoding="utf-8"))
    _require(raw.get("schema") == BASE_SCHEMA, "parent C8 schema differs")
    parent_sha = _sha256(parent_path)
    parent_analysis = json.loads(
        parent_analysis_path.read_text(encoding="utf-8"))
    _require(
        parent_analysis.get("contract_sha256") == parent_sha
        and parent_analysis.get("c8_dual_regime_discovery_positive") is True
        and parent_analysis.get("performance_claim_allowed") is True,
        "parent C8 campaign is not a positive frozen discovery",
    )
    inventory = dict(raw.get("source_inventory", {}))
    _require(bool(inventory), "parent C8 source inventory is missing")
    for relative, expected in inventory.items():
        source = (repo_root / relative).resolve()
        _require(source.is_file() and _sha256(source) == expected,
                 f"parent C8 source drift detected: {relative}")

    section = raw["joint_control"]
    discovery_arms = [dict(row) for row in section["arms"]]
    discovery_blocks = [dict(row) for row in section["blocks"]]
    section["arms"] = list(reversed(discovery_arms))
    section["blocks"] = list(reversed(discovery_blocks))
    arm_order = [str(row["name"]) for row in section["arms"]]
    block_order = [str(row["name"]) for row in section["blocks"]]
    _require(
        arm_order[0] == section["headline_full_arm"]
        and block_order[1] == "05_p_only_dual_decoder_hot",
        "held-out counterbalance does not move TEMPO/remote early",
    )

    required_supported_signals = [
        "cassini_rx_pause_fraction_max",
        "cassini_tx_pause_fraction_max",
        "cassini_host_posted_cycles_per_packet_max",
        "cassini_tx_packets_per_s",
        "cassini_rx_packets_per_s",
        "cassini_oxe_channel_active_fraction_max",
        "cassini_oxe_channel_active_fraction_mean",
        "cassini_ecn_fraction_max",
        "cassini_retries",
        "cassini_timeouts",
        "lmcache_remote_semantic_ops_inflight",
        "lmcache_remote_kv_bytes_inflight",
    ]
    explicit_status_signals = [
        "nccl_collective_p99_ms",
        "nccl_arrival_spread_ms",
        "lmcache_transfer_p99_ms",
    ]
    port_slot_base = 2140
    port_slot_stride = 40
    maximum_port_slot = port_slot_base + port_slot_stride * (
        len(section["arms"]) - 1)
    _require(30_000 + maximum_port_slot < 32_768,
             "held-out endpoint probe port exceeds the user port range")
    raw["independent_validation"] = {
        "schema": INDEPENDENT_SCHEMA,
        "preregistered_at_utc": args.preregistered_at_utc,
        "preregistered_before_fresh_allocation": True,
        "fresh_allocation_required": True,
        "one_shot_no_retry": True,
        "runtime_port_schedule": {
            "port_slot_base": port_slot_base,
            "port_slot_stride_per_arm": port_slot_stride,
            "maximum_port_slot": maximum_port_slot,
            "endpoint_probe_port_base": 30000,
            "maximum_endpoint_probe_port": 30_000 + maximum_port_slot,
            "exclusive_upper_bound": 32768,
        },
        "forbidden_discovery_job_ids": [
            str(value) for value in args.forbid_job_id
        ],
        "prior_failed_attempts": prior_failures,
        "parent_discovery": {
            "contract_path": parent_path.relative_to(repo_root).as_posix(),
            "contract_sha256": parent_sha,
            "analysis_path": parent_analysis_path.relative_to(repo_root).as_posix(),
            "analysis_sha256": _sha256(parent_analysis_path),
        },
        "request_seed": args.request_seed,
        "controller_receives_workload_seed": False,
        "controller_receives_future_arrivals": False,
        "arrival_jitter": {
            "algorithm": "sha256_centered_subspacing_v1",
            "maximum_spacing_fraction": 0.25,
            "same_count_rate_duration_in_every_arm": True,
            "controller_input": False,
        },
        "p_only_prompt_namespace": {
            "algorithm": "c8_marker_offset_v1",
            "base_marker": 240000,
            "marker_offset": 8192,
            "controller_input": False,
        },
        "counterbalance": {
            "discovery_arm_order": [str(row["name"]) for row in discovery_arms],
            "validation_arm_order": arm_order,
            "discovery_block_order": [
                str(row["name"]) for row in discovery_blocks
            ],
            "validation_block_order": block_order,
            "full_temporal_position_reversed": True,
            "remote_regime_temporal_position_reversed": True,
        },
        "remote_favorable_block": "05_p_only_dual_decoder_hot",
        "gates": {
            "background": {
                "minimum_c7_background_completion_fraction": 0.80,
                "minimum_c7_block_tenant_completion_fraction": 0.70,
                "minimum_c7_tenant_jain_fairness": 0.99,
                "maximum_c7_service_lane_failure_fraction": 0.01,
                "minimum_c8_background_completion_fraction": 0.99,
                "maximum_c8_background_noncomplete": 13,
            },
            "telemetry": {
                "required_supported_signals": required_supported_signals,
                "minimum_supported_signal_fraction": 0.90,
                "required_explicit_status_signals": explicit_status_signals,
                "minimum_complete_batch_fraction": 1.0,
                "maximum_collection_p50_ms": 50.0,
                "maximum_collection_p99_ms": 250.0,
                "maximum_admission_wait_p50_ms": 50.0,
                "maximum_admission_wait_p99_ms": 250.0,
                "minimum_source_virtual_service_binding_fraction": 0.50,
                "unavailable_is_never_zero_pressure": True,
            },
        },
        "claim_rule": (
            "Only the frozen independent analyzer may set both performance "
            "and independent_validation_claim_allowed after every gate passes"
        ),
    }
    raw["claim_boundary"] = {
        "controller_performance_claim_allowed": True,
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "purpose": (
            "Preregistered held-out execution; no claim is authorized before "
            "the fresh-allocation analyzer passes"
        ),
    }
    candidate = dict(raw["candidate"])
    candidate.update({
        "validation_id": "tempo-go-c8-c9-heldout-independent-v3",
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    })
    raw["candidate"] = candidate
    raw["purpose"] = (
        "One-shot fresh-allocation C9 validation with reverse temporal order, "
        "held-out arrivals/prompts, background fairness, and telemetry gates"
    )

    independent_sources = (
        "eval/sota_4node/build_tempo_go_c8_independent_validation_contract.py",
        "eval/sota_4node/run_tempo_go_c8_independent_validation_client.py",
        "eval/sota_4node/vllm_lmcache_tempo_go_c8_independent_validation_node.py",
        "eval/sota_4node/c8_independent_validation_node_entry.sh",
        "eval/sota_4node/run_tempo_go_c8_independent_validation_in_allocation.sh",
        "eval/sota_4node/analyze_tempo_go_c8_independent_validation.py",
    )
    for relative in independent_sources:
        source = repo_root / relative
        _require(source.is_file(), f"independent C8 source is missing: {relative}")
        inventory[relative] = _sha256(source)
    raw["source_inventory"] = dict(sorted(inventory.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print("contract_sha256", _sha256(output))
    print("source_inventory_count", len(inventory))
    print("validation_arm_order", ",".join(arm_order))
    print("validation_block_order", ",".join(block_order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
