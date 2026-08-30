#!/usr/bin/env python3
"""Derive the C8 profile that binds global admission to vLLM priority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


MODE = "vllm_priority_business_dual_route_v2"
SOURCE_BALANCE_MODE = "telemetry_uncertainty_virtual_service_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-per-decoder", type=int, default=8)
    parser.add_argument("--minimum-admission-priority", type=int, default=800)
    parser.add_argument("--vllm-priority", type=int, default=-2)
    parser.add_argument("--telemetry-freshness-ms", type=int, default=500)
    parser.add_argument("--telemetry-refresh-timeout-ms", type=int, default=400)
    parser.add_argument(
        "--telemetry-maximum-collection-span-ms", type=int, default=400)
    parser.add_argument(
        "--decoder-background-max-wait-ms", type=int, default=60_000)
    parser.add_argument(
        "--source-balance-uncertainty-fraction", type=float, default=1.0)
    parser.add_argument(
        "--business-clean-pair-pressure-fraction", type=float, default=1.0)
    args = parser.parse_args()

    base = args.base.resolve()
    output = args.output.resolve()
    if not base.is_file() or output.exists():
        raise ValueError("profile input/output boundary is invalid")
    if args.capacity_per_decoder <= 0:
        raise ValueError("priority lane capacity must be positive")
    if args.minimum_admission_priority <= 0:
        raise ValueError("minimum admission priority must be positive")
    if args.vllm_priority not in {-2, -1}:
        raise ValueError("vLLM priority must be -1 or -2")
    if not (
        0 < args.telemetry_maximum_collection_span_ms
        <= args.telemetry_refresh_timeout_ms
        <= args.telemetry_freshness_ms
    ):
        raise ValueError(
            "telemetry windows must satisfy 0 < span <= timeout <= freshness")
    if args.decoder_background_max_wait_ms <= 0:
        raise ValueError("decoder background max wait must be positive")
    if not 0.0 < args.source_balance_uncertainty_fraction <= 1.0:
        raise ValueError("source balance uncertainty fraction must be in (0, 1]")
    if not 0.0 < args.business_clean_pair_pressure_fraction <= 1.0:
        raise ValueError(
            "business clean pair pressure fraction must be in (0, 1]")

    raw = json.loads(base.read_text(encoding="utf-8"))
    raw["profile_id"] = (
        "tempo-go-qwen25-perlmutter-c9-dual-route-business-lane-v1")
    telemetry = raw.get("telemetry")
    if not isinstance(telemetry, dict):
        raise ValueError("base profile telemetry is missing")
    telemetry.update({
        "freshness_ns": args.telemetry_freshness_ms * 1_000_000,
        "refresh_timeout_ns": args.telemetry_refresh_timeout_ms * 1_000_000,
        "maximum_collection_span_ns": (
            args.telemetry_maximum_collection_span_ms * 1_000_000),
    })
    controller = raw.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("base profile controller is missing")
    if (
        controller.get("overload_action") != "endpoint_queue_lease"
        or controller.get("endpoint_queue_admission_mode")
        != "headroom_first_v1"
        or controller.get("endpoint_queue_debt_mode")
        != "completion_credit_mesh_endpoint_queue_v1"
        or controller.get("mesh_control_mode") != "receiver_credit_pxd_v1"
    ):
        raise ValueError("base profile does not expose the C8 mesh queue seam")
    controller.update({
        "priority_service_lane_mode": MODE,
        "priority_service_lane_capacity": args.capacity_per_decoder,
        "priority_service_lane_min_admission_priority": (
            args.minimum_admission_priority),
        "priority_service_lane_priority": args.vllm_priority,
        "decoder_business_admission_mode": "priority_drain_v1",
        "decoder_business_background_max_wait_ns": (
            args.decoder_background_max_wait_ms * 1_000_000),
        "mesh_near_tie_source_balance_mode": SOURCE_BALANCE_MODE,
        "mesh_near_tie_source_balance_uncertainty_fraction": (
            args.source_balance_uncertainty_fraction),
        "business_clean_pair_pressure_fraction": (
            args.business_clean_pair_pressure_fraction),
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded = load_global_profile(output)
    config = loaded.orchestrator_config()
    if (
        config.priority_service_lane_mode != MODE
        or config.priority_service_lane_capacity != args.capacity_per_decoder
        or config.priority_service_lane_min_admission_priority
        != args.minimum_admission_priority
        or config.priority_service_lane_priority != args.vllm_priority
        or config.decoder_business_admission_mode != "priority_drain_v1"
        or config.decoder_business_background_max_wait_ns
        != args.decoder_background_max_wait_ms * 1_000_000
        or config.mesh_near_tie_source_balance_mode != SOURCE_BALANCE_MODE
        or config.mesh_near_tie_source_balance_uncertainty_fraction
        != args.source_balance_uncertainty_fraction
        or config.business_clean_pair_pressure_fraction
        != args.business_clean_pair_pressure_fraction
    ):
        raise RuntimeError("C8 priority service lane did not round-trip")
    if (
        loaded.telemetry.freshness_ns
        != args.telemetry_freshness_ms * 1_000_000
        or loaded.telemetry.refresh_timeout_ns
        != args.telemetry_refresh_timeout_ms * 1_000_000
        or loaded.telemetry.maximum_collection_span_ns
        != args.telemetry_maximum_collection_span_ms * 1_000_000
    ):
        raise RuntimeError("C8 control-plane telemetry budget did not round-trip")
    print(output)
    print("fingerprint", loaded.fingerprint_sha256)
    print("priority_service_lane_capacity", config.priority_service_lane_capacity)
    print("priority_service_lane_priority", config.priority_service_lane_priority)
    print("decoder_business_admission_mode", (
        config.decoder_business_admission_mode))
    print("decoder_business_background_max_wait_ns", (
        config.decoder_business_background_max_wait_ns))
    print("mesh_near_tie_source_balance_mode", (
        config.mesh_near_tie_source_balance_mode))
    print("mesh_near_tie_source_balance_uncertainty_fraction", (
        config.mesh_near_tie_source_balance_uncertainty_fraction))
    print("telemetry_freshness_ns", loaded.telemetry.freshness_ns)
    print("telemetry_refresh_timeout_ns", loaded.telemetry.refresh_timeout_ns)
    print(
        "telemetry_maximum_collection_span_ns",
        loaded.telemetry.maximum_collection_span_ns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
