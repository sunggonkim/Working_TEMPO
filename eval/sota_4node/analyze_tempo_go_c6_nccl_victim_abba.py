#!/usr/bin/env python3
"""Qualify a real NCCL victim against official LMCache/NIXL with ABBA order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


SCHEMA = "tempo-go-c6-nccl-victim-abba-analysis-v1"
CONTRACT_SCHEMA = "tempo-go-c6-qualification-contract-v1"
INCAST_CONTRACT_SCHEMA = "tempo-go-c6-nccl-receiver-incast-contract-v1"
RESULT_SCHEMA = "tempo-lmcache-nixl-contention-2node-1"
TRANSPORT_SCHEMA = "tempo-go-native-transport-receipt-v1"


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


def _ratio(numerator: float, denominator: float) -> float:
    _require(denominator > 0.0, "ratio denominator must be positive")
    return numerator / denominator


def _arm(root: Path, spec: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    name = spec["name"]
    expected_mode = spec["background_mode"]
    arm_root = root / name
    result_path = arm_root / "result.json"
    transport_path = arm_root / "native_transport_receipt.json"
    _require(result_path.is_file(), f"missing native result: {result_path}")
    _require(transport_path.is_file(), f"missing transport receipt: {transport_path}")
    result = _load(result_path)
    transport = _load(transport_path)

    _require(result.get("schema_version") == RESULT_SCHEMA, "native result schema differs")
    _require(result.get("evidence_state") == "live_official_component", "result is not live")
    _require(result.get("world_size") == 8 and result.get("nodes") == 2, "topology differs")
    _require(result.get("pair_count") == 4, "pair count differs")
    _require(result.get("overall_correctness_met") is True, "native correctness failed")
    _require(result.get("baseline", {}).get("proxy") is False, "proxy evidence is forbidden")
    _require(
        result.get("baseline", {}).get("backend") == "NIXL UCX",
        "official NIXL/UCX identity differs",
    )
    _require(
        transport.get("schema") == TRANSPORT_SCHEMA
        and transport.get("production_transport_verified") is True,
        "production transport is not verified",
    )
    _require(
        transport.get("transport", {}).get("nccl_net") == "AWS Libfabric",
        "NCCL did not use the NERSC AWS Libfabric path",
    )

    config = result.get("config")
    _require(isinstance(config, dict), "native config is missing")
    params = contract["nccl_victim_abba"]["parameters"]
    expected = {
        "requests": params["requests"],
        "kv_bytes": params["kv_mib"] * 1024 * 1024,
        "token_iters": params["token_iters"],
        "foreground_bytes": params["foreground_mib"] * 1024 * 1024,
        "block_delay_s": params["block_delay_s"],
        "minimum_active_duration_s": contract["nccl_victim_abba"][
            "minimum_active_duration_s"
        ],
        "maximum_blocks": params["maximum_blocks"],
        "process_group_timeout_s": params["process_group_timeout_s"],
        "nixl_transfer_timeout_s": params["nixl_transfer_timeout_s"],
        "background_mode": expected_mode,
    }
    for key, value in expected.items():
        _require(config.get(key) == value, f"{name} config differs: {key}")
    expected_traffic = contract["nccl_victim_abba"].get(
        "traffic_pattern", "paired_1to1"
    )
    _require(
        config.get("traffic_pattern", "paired_1to1") == expected_traffic,
        f"{name} traffic pattern differs",
    )
    _require(
        params["minimum_blocks"] <= config.get("blocks", 0) <= params["maximum_blocks"],
        f"{name} block count is outside the frozen bounds",
    )

    active = result.get("active_loop")
    _require(isinstance(active, dict), f"{name} active-loop receipt is missing")
    _require(active.get("horizon_met") is True, f"{name} did not reach the service horizon")
    _require(
        _finite_positive(active.get("rank_min_elapsed_ms"), "rank_min_elapsed_ms")
        >= 1000.0 * contract["nccl_victim_abba"]["minimum_active_duration_s"],
        f"{name} active horizon is shorter than frozen",
    )

    blocks = result.get("blocks")
    _require(isinstance(blocks, list) and len(blocks) == config["blocks"], "block list differs")
    for block in blocks:
        _require(block.get("correctness_met") is True, f"{name} block correctness failed")
        if expected_mode == "nccl_only":
            _require(
                block.get("expected_background_bytes") == 0
                and block.get("source_completed_bytes") == 0
                and block.get("receiver_verified_bytes") == 0,
                f"{name} NCCL-only control contains background bytes",
            )
        else:
            _require(
                block.get("expected_background_bytes", 0) > 0
                and block.get("full_bytes_completed") is True
                and block.get("full_bytes_verified") is True,
                f"{name} LMCache bytes did not complete and verify",
            )

    summary = result.get("summary")
    _require(isinstance(summary, dict), "summary is missing")
    p50 = _finite_positive(summary.get("global_token_tail_p50_ms"), "NCCL p50")
    p99 = _finite_positive(summary.get("global_token_tail_p99_ms"), "NCCL p99")
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
    return {
        "name": name,
        "background_mode": expected_mode,
        "host_pair": [rank_hosts[0], rank_hosts[4]],
        "blocks": config["blocks"],
        "rank_min_active_elapsed_ms": active["rank_min_elapsed_ms"],
        "nccl_collective_completion_p50_ms": p50,
        "nccl_collective_completion_p99_ms": p99,
        "background_completion_p99_ms": summary.get("background_completion_p99_ms"),
        "result": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "transport_receipt": str(transport_path.resolve()),
        "transport_receipt_sha256": _sha256(transport_path),
    }


def analyze(root: Path, contract_path: Path) -> dict[str, Any]:
    root = root.resolve()
    contract_path = contract_path.resolve()
    contract = _load(contract_path)
    _require(
        contract.get("schema") in {CONTRACT_SCHEMA, INCAST_CONTRACT_SCHEMA},
        "qualification contract differs",
    )
    specs = contract.get("nccl_victim_abba", {}).get("arms")
    _require(isinstance(specs, list) and len(specs) == 4, "ABBA arm contract differs")
    _require(
        [item.get("background_mode") for item in specs]
        == ["nccl_only", "nixl_ucx", "nixl_ucx", "nccl_only"],
        "ABBA order differs",
    )
    arms = [_arm(root, spec, contract) for spec in specs]
    _require(
        all(item["host_pair"] == arms[0]["host_pair"] for item in arms[1:]),
        "ABBA arms did not use the same two physical nodes",
    )
    baseline = [arms[0], arms[3]]
    aggressor = [arms[1], arms[2]]
    pairings = list(zip(baseline, aggressor, strict=True))
    pair_metrics = []
    for control, loaded in pairings:
        p50_ratio = _ratio(
            loaded["nccl_collective_completion_p50_ms"],
            control["nccl_collective_completion_p50_ms"],
        )
        p99_ratio = _ratio(
            loaded["nccl_collective_completion_p99_ms"],
            control["nccl_collective_completion_p99_ms"],
        )
        pair_metrics.append({
            "control": control["name"],
            "aggressor": loaded["name"],
            "p50_degradation_fraction": p50_ratio - 1.0,
            "p50_ratio": p50_ratio,
            "p99_ratio": p99_ratio,
        })

    p50_degradation = statistics.median(
        item["p50_degradation_fraction"] for item in pair_metrics
    )
    p99_ratio = statistics.median(item["p99_ratio"] for item in pair_metrics)
    gates = {
        "all_native_correct_and_transport_verified": True,
        "same_population_abba": True,
        "same_physical_node_pair_abba": True,
        "every_arm_at_least_30s": all(
            item["rank_min_active_elapsed_ms"] >= 30_000.0 for item in arms
        ),
        "victim_p50_degradation_at_least_25pct": p50_degradation >= 0.25,
        "victim_p99_at_least_2x": p99_ratio >= 2.0,
        "victim_slo_drop_at_least_20pp": False,
    }
    q1_pass = (
        gates["victim_p50_degradation_at_least_25pct"]
        or gates["victim_p99_at_least_2x"]
        or gates["victim_slo_drop_at_least_20pp"]
    )
    return {
        "schema": SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "root": str(root),
        "arms": arms,
        "paired_effects": pair_metrics,
        "aggregate_effect": {
            "median_p50_degradation_fraction": p50_degradation,
            "median_p99_ratio": p99_ratio,
        },
        "gates": gates,
        "q1_real_nccl_victim_pass": q1_pass,
        "q3_service_horizon_pass": gates["every_arm_at_least_30s"],
        "controller_performance_run_allowed": False,
        "performance_claim_allowed": False,
        "next_gate": (
            "decoder_output_completion_victim_and_fixed_pxd_recovery"
            if q1_pass
            else "increase_real_lmcache_or_decode_aggressor_coupling_without_tuning_controller"
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
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "q1_real_nccl_victim_pass": value["q1_real_nccl_victim_pass"],
        "q3_service_horizon_pass": value["q3_service_horizon_pass"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
