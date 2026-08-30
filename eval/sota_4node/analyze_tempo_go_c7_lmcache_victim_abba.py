#!/usr/bin/env python3
"""Qualify official LMCache completion as a victim of sustained real NCCL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


SCHEMA = "tempo-go-c7-lmcache-victim-abba-analysis-v1"
CONTRACT_SCHEMA = "tempo-go-c7-lmcache-victim-abba-contract-v1"
EXECUTION_SCHEMA = "tempo-go-c7-lmcache-victim-execution-v1"
RESULT_SCHEMA = "tempo-lmcache-nixl-contention-2node-1"
TRANSPORT_SCHEMA = "tempo-go-native-transport-receipt-v1"
MIB = 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_positive(value: object, name: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0,
        f"{name} must be finite and positive",
    )
    return float(value)


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "percentile requires at least one sample")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _ratio(numerator: float, denominator: float) -> float:
    _require(denominator > 0.0, "ratio denominator must be positive")
    return numerator / denominator


def _arm(
    root: Path,
    spec: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    name = spec["name"]
    arm_root = root / name
    result_path = arm_root / "result.json"
    transport_path = arm_root / "native_transport_receipt.json"
    _require(result_path.is_file(), f"missing native result: {result_path}")
    _require(transport_path.is_file(), f"missing transport receipt: {transport_path}")
    result = _load(result_path)
    transport = _load(transport_path)

    _require(result.get("schema_version") == RESULT_SCHEMA, f"{name} result schema differs")
    _require(result.get("evidence_state") == "live_official_component", f"{name} is not live")
    _require(result.get("world_size") == 8 and result.get("nodes") == 2, f"{name} topology differs")
    _require(result.get("pair_count") == 4, f"{name} pair count differs")
    _require(result.get("overall_correctness_met") is True, f"{name} correctness failed")
    _require(result.get("baseline", {}).get("proxy") is False, f"{name} proxy is forbidden")
    _require(result.get("baseline", {}).get("backend") == "NIXL UCX", f"{name} LMCache backend differs")
    _require(
        transport.get("schema") == TRANSPORT_SCHEMA
        and transport.get("production_transport_verified") is True,
        f"{name} production transport is not verified",
    )
    _require(
        transport.get("transport", {}).get("nccl_net") == "AWS Libfabric",
        f"{name} NCCL did not use NERSC AWS Libfabric",
    )

    section = contract["lmcache_victim_abba"]
    params = section["parameters"]
    config = result.get("config")
    _require(isinstance(config, dict), f"{name} native config is missing")
    expected = {
        "requests": params["requests"],
        "kv_bytes": params["kv_mib"] * MIB,
        "token_iters": spec["token_iters"],
        "foreground_bytes": params["foreground_mib"] * MIB,
        "block_delay_s": params["block_delay_s"],
        "minimum_active_duration_s": section["minimum_active_duration_s"],
        "maximum_blocks": params["maximum_blocks"],
        "process_group_timeout_s": params["process_group_timeout_s"],
        "nixl_transfer_timeout_s": params["nixl_transfer_timeout_s"],
        "background_mode": "nixl_ucx",
        "traffic_pattern": section["topology"]["traffic_pattern"],
    }
    for key, value in expected.items():
        _require(config.get(key) == value, f"{name} config differs: {key}")
    _require(
        params["minimum_blocks"] <= config.get("blocks", 0) <= params["maximum_blocks"],
        f"{name} block count is outside the frozen bounds",
    )

    active = result.get("active_loop")
    _require(isinstance(active, dict), f"{name} active-loop receipt is missing")
    _require(active.get("horizon_met") is True, f"{name} did not reach its service horizon")
    active_ms = _finite_positive(active.get("rank_min_elapsed_ms"), f"{name} active elapsed")
    _require(
        active_ms >= 1000.0 * section["minimum_active_duration_s"],
        f"{name} active horizon is shorter than frozen",
    )

    expected_bytes = 4 * params["requests"] * params["kv_mib"] * MIB
    blocks = result.get("blocks")
    _require(isinstance(blocks, list) and len(blocks) == config["blocks"], f"{name} block list differs")
    completions: list[float] = []
    for index, block in enumerate(blocks):
        _require(block.get("block_index") == index, f"{name} block index differs")
        _require(block.get("correctness_met") is True, f"{name} block correctness failed")
        _require(
            block.get("expected_background_bytes") == expected_bytes
            and block.get("source_completed_bytes") == expected_bytes
            and block.get("receiver_verified_bytes") == expected_bytes
            and block.get("full_bytes_completed") is True
            and block.get("full_bytes_verified") is True,
            f"{name} LMCache bytes did not complete and verify",
        )
        completions.append(
            _finite_positive(block.get("background_completion_ms"), f"{name} completion")
        )

    rank_diagnostics = result.get("rank_diagnostics")
    _require(
        isinstance(rank_diagnostics, list) and len(rank_diagnostics) == 8,
        f"{name} rank diagnostics differ",
    )
    rank_hosts = [item.get("hostname") for item in rank_diagnostics]
    _require(
        len(set(rank_hosts[:4])) == 1
        and len(set(rank_hosts[4:])) == 1
        and rank_hosts[0] != rank_hosts[4],
        f"{name} node-major host topology differs",
    )
    fixed_nodelist = transport.get("transport", {}).get("fixed_nodelist")
    if fixed_nodelist is not None:
        _require(fixed_nodelist == [rank_hosts[0], rank_hosts[4]], f"{name} transport node pair differs")
    for rank, diagnostic in enumerate(rank_diagnostics):
        rank_blocks = diagnostic.get("blocks")
        _require(isinstance(rank_blocks, list) and len(rank_blocks) == config["blocks"], f"{name} rank block receipts differ")
        if rank < 4:
            for block in rank_blocks:
                _require(
                    block.get("attempted_objects") == params["requests"]
                    and block.get("returned_objects") == params["requests"]
                    and block.get("started") is True
                    and block.get("finished") is True
                    and block.get("worker_alive_after_join") is False
                    and block.get("error") is None,
                    f"{name} source transfer lifecycle differs",
                )

    summary = result.get("summary")
    _require(isinstance(summary, dict), f"{name} summary is missing")
    p50 = statistics.median(completions)
    p99 = _percentile(completions, 0.99)
    _require(math.isclose(float(summary.get("background_completion_p50_ms")), p50, rel_tol=1e-9), f"{name} transfer p50 summary differs")
    _require(math.isclose(float(summary.get("background_completion_p99_ms")), p99, rel_tol=1e-9), f"{name} transfer p99 summary differs")
    return {
        "name": name,
        "nccl_load": spec["nccl_load"],
        "nccl_token_iters_per_block": spec["token_iters"],
        "nccl_collective_calls": spec["token_iters"] * config["blocks"],
        "nccl_collective_payload_bytes_per_rank": (
            spec["token_iters"] * config["blocks"] * config["foreground_bytes"]
        ),
        "host_pair": [rank_hosts[0], rank_hosts[4]],
        "blocks": config["blocks"],
        "lmcache_bytes_per_block": expected_bytes,
        "lmcache_total_verified_bytes": expected_bytes * config["blocks"],
        "rank_min_active_elapsed_ms": active_ms,
        "lmcache_completion_samples_ms": completions,
        "lmcache_completion_p50_ms": p50,
        "lmcache_completion_p99_ms": p99,
        "nccl_collective_completion_p50_ms": _finite_positive(
            summary.get("global_token_tail_p50_ms"), f"{name} NCCL p50"
        ),
        "nccl_collective_completion_p99_ms": _finite_positive(
            summary.get("global_token_tail_p99_ms"), f"{name} NCCL p99"
        ),
        "result": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "transport_receipt": str(transport_path.resolve()),
        "transport_receipt_sha256": _sha256(transport_path),
    }


def analyze(root: Path, contract_path: Path) -> dict[str, Any]:
    root = root.resolve()
    contract_path = contract_path.resolve()
    contract = _load(contract_path)
    _require(contract.get("schema") == CONTRACT_SCHEMA, "C7 qualification contract differs")
    specs = contract.get("lmcache_victim_abba", {}).get("arms")
    _require(isinstance(specs, list) and len(specs) == 4, "C7 ABBA arm contract differs")
    _require(
        [item.get("nccl_load") for item in specs] == ["control", "hot", "hot", "control"],
        "C7 ABBA order differs",
    )

    receipt_path = root / "execution_receipt.json"
    _require(receipt_path.is_file(), "C7 execution receipt is missing")
    receipt = _load(receipt_path)
    _require(receipt.get("schema") == EXECUTION_SCHEMA, "C7 execution receipt schema differs")
    _require(receipt.get("contract_sha256") == _sha256(contract_path), "C7 contract binding differs")
    _require(receipt.get("batch_submission") is False, "batch execution is outside C7 scope")
    _require(
        receipt.get("privileged_or_container_configuration") is False,
        "privileged/container execution is outside C7 scope",
    )

    arms = [_arm(root, spec, contract) for spec in specs]
    same_host_pair = all(item["host_pair"] == arms[0]["host_pair"] for item in arms[1:])
    same_blocks = len({item["blocks"] for item in arms}) == 1
    same_victim_bytes = len({item["lmcache_total_verified_bytes"] for item in arms}) == 1
    _require(same_host_pair, "ABBA arms did not use the same physical node pair")
    _require(same_blocks, "ABBA arms did not execute the same LMCache transfer count")
    _require(same_victim_bytes, "ABBA arms did not verify the same LMCache bytes")

    slo_ms = float(contract["lmcache_victim_abba"]["qualification_gates"]["transfer_slo_ms"])
    controls = [arms[0], arms[3]]
    loaded = [arms[1], arms[2]]
    paired_effects = []
    for control, hot in zip(controls, loaded, strict=True):
        control_slo = sum(value <= slo_ms for value in control["lmcache_completion_samples_ms"]) / control["blocks"]
        hot_slo = sum(value <= slo_ms for value in hot["lmcache_completion_samples_ms"]) / hot["blocks"]
        paired_effects.append({
            "control": control["name"],
            "hot": hot["name"],
            "p50_degradation_fraction": _ratio(hot["lmcache_completion_p50_ms"], control["lmcache_completion_p50_ms"]) - 1.0,
            "p50_ratio": _ratio(hot["lmcache_completion_p50_ms"], control["lmcache_completion_p50_ms"]),
            "p99_ratio": _ratio(hot["lmcache_completion_p99_ms"], control["lmcache_completion_p99_ms"]),
            "control_slo_attainment": control_slo,
            "hot_slo_attainment": hot_slo,
            "slo_attainment_drop_percentage_points": 100.0 * (control_slo - hot_slo),
        })

    p50_degradation = statistics.median(item["p50_degradation_fraction"] for item in paired_effects)
    p99_ratio = statistics.median(item["p99_ratio"] for item in paired_effects)
    slo_drop_pp = statistics.median(item["slo_attainment_drop_percentage_points"] for item in paired_effects)
    thresholds = contract["lmcache_victim_abba"]["qualification_gates"]
    gates = {
        "all_native_correct_and_transport_verified": True,
        "same_official_lmcache_victim_population_abba": same_blocks and same_victim_bytes,
        "same_physical_node_pair_abba": same_host_pair,
        "every_arm_at_least_30s": all(item["rank_min_active_elapsed_ms"] >= 30_000.0 for item in arms),
        "victim_p50_degradation_at_least_25pct": p50_degradation >= float(thresholds["transfer_p50_degradation_fraction"]),
        "victim_p99_at_least_2x": p99_ratio >= float(thresholds["transfer_p99_ratio"]),
        "victim_slo_drop_at_least_20pp": slo_drop_pp >= float(thresholds["transfer_slo_attainment_drop_percentage_points"]),
    }
    base_valid = all(
        gates[key]
        for key in (
            "all_native_correct_and_transport_verified",
            "same_official_lmcache_victim_population_abba",
            "same_physical_node_pair_abba",
            "every_arm_at_least_30s",
        )
    )
    material = any(
        gates[key]
        for key in (
            "victim_p50_degradation_at_least_25pct",
            "victim_p99_at_least_2x",
            "victim_slo_drop_at_least_20pp",
        )
    )
    qualification_pass = base_valid and material
    return {
        "schema": SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "execution_receipt": str(receipt_path.resolve()),
        "execution_receipt_sha256": _sha256(receipt_path),
        "root": str(root),
        "arms": arms,
        "paired_effects": paired_effects,
        "aggregate_effect": {
            "median_p50_degradation_fraction": p50_degradation,
            "median_p99_ratio": p99_ratio,
            "median_slo_attainment_drop_percentage_points": slo_drop_pp,
            "transfer_slo_ms": slo_ms,
        },
        "gates": gates,
        "c7_real_lmcache_victim_pass": qualification_pass,
        "q3_service_horizon_pass": gates["every_arm_at_least_30s"],
        "actual_vllm_joint_control_run_allowed": qualification_pass,
        "controller_performance_claim_allowed": False,
        "performance_claim_allowed": False,
        "next_gate": (
            "actual_vllm_remote_local_receiver_credit_joint_control"
            if qualification_pass
            else "stop_unchanged_c7_controller_run_and_reassess_victim_coupling"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    value = analyze(args.root, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "c7_real_lmcache_victim_pass": value["c7_real_lmcache_victim_pass"],
        "actual_vllm_joint_control_run_allowed": value["actual_vllm_joint_control_run_allowed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
