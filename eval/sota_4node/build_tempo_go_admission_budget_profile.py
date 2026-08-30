#!/usr/bin/env python3
"""Build an immutable TEMPO-GO admission-policy candidate profile.

This candidate is intentionally derived from the held-out frozen global profile.
It preserves the service-proxy policy, endpoint/model/workload identity, tenant
contract and telemetry contract while changing explicitly named admission,
scaling or failure-policy fields.  It is a CPU/discovery artifact and does not
grant a native performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tempo.pd_global_profile import global_profile_fingerprint, load_global_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--maximum-queue-wait-ns", type=int, required=True)
    parser.add_argument(
        "--queue-reservation", action="append", default=[],
        metavar="TENANT=SLOTS",
        help="queued-slot reservation; repeat once per tenant",
    )
    parser.add_argument(
        "--queue-lease-on-timeout-tenants",
        help=(
            "optional comma-separated tenant IDs allowed to cross the "
            "global timeout into the bounded endpoint queue; omitted "
            "preserves the base profile"
        ),
    )
    parser.add_argument(
        "--proactive-scale-up-queue-fraction", type=float,
        help="optional current queue fraction for prewarmed-pair activation",
    )
    parser.add_argument(
        "--proactive-scale-up-wait-fraction", type=float,
        help="optional tenant wait fraction for prewarmed-pair activation",
    )
    parser.add_argument(
        "--proactive-scale-up-active-pair-penalty-ms", type=float,
        help="optional score penalty while proactive pair scaling is live",
    )
    parser.add_argument(
        "--route-failure-quarantine-mode",
        choices=("disabled", "deny_until_probe"),
        help=(
            "optional failure-injection variant; all other controller fields "
            "remain inherited from the base profile"
        ),
    )
    parser.add_argument(
        "--telemetry-failure-quarantine-mode",
        choices=("disabled", "deny_until_probe"),
        help=(
            "optional cumulative endpoint-failure circuit mode; this is "
            "separate from explicit request failure quarantine"
        ),
    )
    parser.add_argument(
        "--telemetry-failure-quarantine-scope",
        choices=("route", "pair"),
        help="scope of a cumulative endpoint-failure circuit",
    )
    parser.add_argument(
        "--survivor-capacity-reserve-fraction", type=float,
        help=(
            "optional fraction of surviving-pair capacity kept for urgent "
            "or minimum-service tenants after pair quarantine"
        ),
    )
    parser.add_argument(
        "--survivor-reserve-bypass-min-weight", type=float,
        help="optional tenant weight allowed to use the survivor reserve",
    )
    parser.add_argument(
        "--shared-fabric-control-mode",
        choices=("disabled", "global_budget_v3"),
        help="optional allocation-wide remote-fabric budget mode",
    )
    parser.add_argument(
        "--shared-remote-requests-capacity", type=int,
        help="optional allocation-wide concurrent remote request capacity",
    )
    parser.add_argument(
        "--shared-remote-kv-bytes-capacity", type=int,
        help="optional allocation-wide remote KV-byte capacity",
    )
    parser.add_argument(
        "--shared-remote-semantic-ops-capacity", type=int,
        help="optional allocation-wide remote semantic-operation capacity",
    )
    parser.add_argument(
        "--shared-remote-limit-floor-fraction", type=float,
        help="optional minimum fraction retained under shared pressure",
    )
    parser.add_argument(
        "--shared-remote-stagger-max-us", type=int,
        help="optional maximum allocation-wide remote dispatch stagger",
    )
    parser.add_argument(
        "--cross-layer-remote-receiver-guard-mode",
        choices=("disabled", "deny_while_hot"),
        help="optional LMCache receiver admission guard mode",
    )
    parser.add_argument(
        "--cross-layer-remote-receiver-guard-scope",
        choices=("pair", "shared_group"),
        help="optional pair or shared-fabric-group receiver guard scope",
    )
    parser.add_argument(
        "--cross-layer-remote-receiver-guard-group-id",
        help="optional explicit allocation-wide receiver guard group identity",
    )
    parser.add_argument(
        "--cross-layer-remote-receiver-guard-p99-ms", type=float,
        help="optional pair-scoped LMCache p99 ceiling for the receiver guard",
    )
    parser.add_argument(
        "--service-feasibility-mode",
        choices=("disabled", "deadline_residual_v1"),
        help="optional observed service-wave deadline feasibility guard",
    )
    parser.add_argument(
        "--service-forecast-safety-factor", type=float,
        help="optional conservative multiplier for service-wave forecasts",
    )
    parser.add_argument(
        "--protected-service-lane-mode",
        choices=(
            "disabled",
            "tenant_pair_edge_reservation_v1",
            "tenant_pair_edge_reservation_v2",
        ),
        help="optional tenant/P-D edge protected service reservation",
    )
    parser.add_argument(
        "--protected-service-lane-capacity", type=int,
        help="optional protected service slots per decoder/P-D edge",
    )
    parser.add_argument(
        "--protected-service-lane-min-admission-priority", type=int,
        help="optional minimum tenant priority for the protected lane",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.profile_id.strip():
        raise ValueError("profile id must be nonempty")
    if args.maximum_queue_wait_ns <= 0:
        raise ValueError("maximum queue wait must be positive")
    for name in (
        "proactive_scale_up_queue_fraction",
        "proactive_scale_up_wait_fraction",
    ):
        value = getattr(args, name)
        if value is not None and not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if (
        args.proactive_scale_up_active_pair_penalty_ms is not None
        and args.proactive_scale_up_active_pair_penalty_ms < 0.0
    ):
        raise ValueError(
            "proactive scale-up active-pair penalty must be non-negative")
    if (
        args.survivor_capacity_reserve_fraction is not None
        and not 0.0 <= args.survivor_capacity_reserve_fraction < 1.0
    ):
        raise ValueError(
            "survivor capacity reserve fraction must be in [0, 1)")
    if (
        args.survivor_reserve_bypass_min_weight is not None
        and args.survivor_reserve_bypass_min_weight < 0.0
    ):
        raise ValueError(
            "survivor reserve bypass weight must be non-negative")
    for name in (
        "shared_remote_requests_capacity",
        "shared_remote_kv_bytes_capacity",
        "shared_remote_semantic_ops_capacity",
        "shared_remote_stagger_max_us",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.shared_remote_limit_floor_fraction is not None and not (
        0.0 < args.shared_remote_limit_floor_fraction <= 1.0
    ):
        raise ValueError("shared remote limit floor fraction must be in (0, 1]")
    if (
        args.service_forecast_safety_factor is not None
        and args.service_forecast_safety_factor < 1.0
    ):
        raise ValueError("service forecast safety factor must be at least 1")
    for name in (
        "protected_service_lane_capacity",
        "protected_service_lane_min_admission_priority",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")
    shared_capacity_names = (
        "shared_remote_requests_capacity",
        "shared_remote_kv_bytes_capacity",
        "shared_remote_semantic_ops_capacity",
    )
    if (
        args.shared_fabric_control_mode == "global_budget_v3"
        and any(getattr(args, name) is None for name in shared_capacity_names)
    ):
        raise ValueError(
            "global_budget_v3 requires all shared remote capacities")
    base = args.base_profile.resolve()
    raw = json.loads(base.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("base global profile is not an object")
    if raw.get("schema") != "tempo-go-profile-v1":
        raise ValueError("base global profile schema differs")
    controller = raw.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("base global profile controller is not an object")
    candidate: dict[str, Any] = dict(raw)
    candidate["profile_id"] = args.profile_id
    candidate_controller = dict(controller)
    candidate_controller["maximum_queue_wait_ns"] = args.maximum_queue_wait_ns
    for name in (
        "proactive_scale_up_queue_fraction",
        "proactive_scale_up_wait_fraction",
        "proactive_scale_up_active_pair_penalty_ms",
    ):
        value = getattr(args, name)
        if value is not None:
            candidate_controller[name] = value
    if args.cross_layer_remote_receiver_guard_mode is not None:
        candidate_controller["cross_layer_remote_receiver_guard_mode"] = (
            args.cross_layer_remote_receiver_guard_mode
        )
    if args.cross_layer_remote_receiver_guard_scope is not None:
        candidate_controller["cross_layer_remote_receiver_guard_scope"] = (
            args.cross_layer_remote_receiver_guard_scope
        )
    if args.cross_layer_remote_receiver_guard_group_id is not None:
        if not args.cross_layer_remote_receiver_guard_group_id.strip():
            raise ValueError("receiver guard group id must be nonempty")
        candidate_controller[
            "cross_layer_remote_receiver_guard_group_id"
        ] = args.cross_layer_remote_receiver_guard_group_id
    if args.cross_layer_remote_receiver_guard_p99_ms is not None:
        if args.cross_layer_remote_receiver_guard_p99_ms <= 0.0:
            raise ValueError(
                "receiver guard p99 ceiling must be positive")
        candidate_controller["cross_layer_remote_receiver_guard_p99_ms"] = (
            args.cross_layer_remote_receiver_guard_p99_ms
        )
    if args.route_failure_quarantine_mode is not None:
        candidate_controller["route_failure_quarantine_mode"] = (
            args.route_failure_quarantine_mode
        )
    if args.telemetry_failure_quarantine_mode is not None:
        candidate_controller["telemetry_failure_quarantine_mode"] = (
            args.telemetry_failure_quarantine_mode
        )
    if args.telemetry_failure_quarantine_scope is not None:
        candidate_controller["telemetry_failure_quarantine_scope"] = (
            args.telemetry_failure_quarantine_scope
        )
    if args.survivor_capacity_reserve_fraction is not None:
        candidate_controller["survivor_capacity_reserve_fraction"] = (
            args.survivor_capacity_reserve_fraction
        )
    if args.survivor_reserve_bypass_min_weight is not None:
        candidate_controller["survivor_reserve_bypass_min_weight"] = (
            args.survivor_reserve_bypass_min_weight
        )
    for name in (
        "shared_fabric_control_mode",
        "shared_remote_requests_capacity",
        "shared_remote_kv_bytes_capacity",
        "shared_remote_semantic_ops_capacity",
        "shared_remote_limit_floor_fraction",
        "shared_remote_stagger_max_us",
    ):
        value = getattr(args, name)
        if value is not None:
            candidate_controller[name] = value
    for name in (
        "service_feasibility_mode",
        "service_forecast_safety_factor",
        "protected_service_lane_mode",
        "protected_service_lane_capacity",
        "protected_service_lane_min_admission_priority",
    ):
        value = getattr(args, name)
        if value is not None:
            candidate_controller[name] = value
    candidate["controller"] = candidate_controller
    tenants = candidate.get("tenants")
    if not isinstance(tenants, list) or not tenants:
        raise ValueError("base global profile tenants are missing")
    reservations: dict[str, int] = {}
    for value in args.queue_reservation:
        if "=" not in value:
            raise ValueError("queue reservation must be TENANT=SLOTS")
        tenant, slots_text = value.split("=", 1)
        if not tenant.strip() or not slots_text.isdigit():
            raise ValueError("queue reservation must be TENANT=nonnegative_int")
        slots = int(slots_text)
        if tenant in reservations:
            raise ValueError(f"duplicate queue reservation: {tenant}")
        reservations[tenant] = slots
    known = {
        str(item.get("tenant_id")) for item in tenants
        if isinstance(item, dict)
    }
    unknown = set(reservations) - known
    if unknown:
        raise ValueError(f"queue reservation has unknown tenant: {sorted(unknown)}")
    queue_lease_tenants = None
    if args.queue_lease_on_timeout_tenants is not None:
        queue_lease_tenants = {
            value.strip()
            for value in args.queue_lease_on_timeout_tenants.split(",")
            if value.strip()
        }
        unknown = queue_lease_tenants - known
        if unknown:
            raise ValueError(
                "queue-lease tenant is unknown: "
                f"{sorted(unknown)}"
            )
    for item in tenants:
        if not isinstance(item, dict):
            raise ValueError("base global profile tenant is not an object")
        tenant_id = str(item["tenant_id"])
        item["queue_reservation_slots"] = reservations.get(
            tenant_id, int(item.get("queue_reservation_slots", 0)))
        if queue_lease_tenants is not None:
            item["queue_lease_on_timeout"] = tenant_id in queue_lease_tenants
    candidate["fingerprint_sha256"] = global_profile_fingerprint(candidate)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_global_profile(output)
    if loaded.profile_id != args.profile_id:
        raise RuntimeError("admission-budget profile did not round-trip")
    if loaded.orchestrator_config().maximum_queue_wait_ns != (
        args.maximum_queue_wait_ns
    ):
        raise RuntimeError("admission-budget wait cap did not round-trip")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
