#!/usr/bin/env python3
"""Build the first audit-only TEMPO-GO discovery profile from frozen C4 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile
from tempo.pd_global_profile import (
    SCHEMA,
    TRANSPORT,
    global_profile_fingerprint,
    load_global_profile,
)


ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elastic-profile", type=Path, required=True)
    parser.add_argument("--endpoint-profile", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument(
        "--profile-id",
        default="tempo-go-qwen25-perlmutter-discovery-v1",
        help="immutable profile identity recorded in decision provenance",
    )
    parser.add_argument(
        "--remote-semantic-ops-safety-reserve", type=int, default=0,
        help=(
            "reserve this many endpoint semantic-op slots from global "
            "admission; keep zero for the legacy discovery profile"),
    )
    parser.add_argument(
        "--proactive-scale-up-queue-fraction", type=float, default=1.0,
        help=(
            "current queue occupancy fraction that permits prewarmed pair "
            "activation; 1.0 preserves the legacy trigger"),
    )
    parser.add_argument(
        "--proactive-scale-up-wait-fraction", type=float, default=1.0,
        help=(
            "tenant queue-wait fraction that permits prewarmed pair "
            "activation; 1.0 preserves the legacy trigger"),
    )
    parser.add_argument(
        "--proactive-scale-up-active-pair-penalty-ms", type=float, default=0.0,
        help=(
            "additional score applied to active pairs while a queue/SLO "
            "scale trigger is live; zero preserves legacy scoring"),
    )
    parser.add_argument(
        "--controller-maximum-queue-wait-ns", type=int,
        default=2_000_000_000,
        help=(
            "global admission wait cap; freeze this explicitly for each "
            "business-policy candidate"),
    )
    parser.add_argument(
        "--controller-queue-capacity", type=int,
        default=128,
        help=(
            "bounded global ingress queue capacity; freeze this explicitly "
            "for each admission-window candidate"),
    )
    parser.add_argument(
        "--telemetry-freshness-ns", type=int, default=100_000_000,
        help=(
            "allocation-wide telemetry age budget used by hierarchical "
            "identity validation; freeze this explicitly for each candidate"
        ),
    )
    parser.add_argument(
        "--telemetry-refresh-timeout-ns", type=int, default=50_000_000,
        help=(
            "bounded parallel telemetry refresh timeout; freeze this "
            "explicitly for each candidate"
        ),
    )
    parser.add_argument(
        "--telemetry-stale-grace-ns", type=int, default=0,
        help=(
            "bounded last-snapshot admission grace after a refresh timeout; "
            "zero preserves fail-closed telemetry behavior"
        ),
    )
    parser.add_argument(
        "--interactive-telemetry-stale-grace-ns", type=int, default=0,
        help=(
            "business-scoped telemetry grace for interactive admission after "
            "a request-triggered refresh failure"),
    )
    parser.add_argument(
        "--telemetry-maximum-collection-span-ns", type=int,
        default=50_000_000,
        help=(
            "maximum causal collection interval accepted by the telemetry "
            "adapter; freeze this explicitly for each candidate"
        ),
    )
    parser.add_argument(
        "--route-failure-quarantine-mode",
        choices=("disabled", "deny_until_probe"),
        default="disabled",
        help=(
            "convert explicit endpoint failure receipts into fail-closed "
            "route/pair quarantine until a later PROBE telemetry sample"),
    )
    parser.add_argument(
        "--pair-count", type=int, choices=(1, 2), default=2,
        help=(
            "number of inference P/D pairs represented by this profile; "
            "use one for P1PAIR plus an external co-job in the same allocation"
        ),
    )
    parser.add_argument(
        "--telemetry-failure-quarantine-mode",
        choices=("disabled", "deny_until_probe"),
        default="disabled",
    )
    parser.add_argument(
        "--telemetry-failure-quarantine-scope",
        choices=("route", "pair"), default="pair",
    )
    parser.add_argument(
        "--survivor-capacity-reserve-fraction", type=float, default=0.0,
    )
    parser.add_argument(
        "--survivor-reserve-bypass-min-weight", type=float, default=0.0,
    )
    parser.add_argument(
        "--cross-layer-remote-limit-floor-fraction", type=float, default=0.25,
        help="minimum remote transfer window fraction under live cross-layer pressure",
    )
    parser.add_argument(
        "--cross-layer-local-limit-floor-fraction", type=float, default=0.50,
        help="minimum local prefill window fraction under live cross-layer pressure",
    )
    parser.add_argument(
        "--cross-layer-stagger-max-us", type=int, default=2_000,
        help="maximum causal request-start stagger emitted by the joint controller",
    )
    parser.add_argument(
        "--cross-layer-control-mode",
        choices=("hard_window_v1", "soft_shadow_price_v2"),
        default="hard_window_v1",
        help=(
            "v2 treats cross-layer windows as resource shadow prices and "
            "retains a work-conserving enforced lease for the commit"),
    )
    parser.add_argument(
        "--cross-layer-shadow-price-ms", type=float, default=0.0,
        help=(
            "score penalty per unit of action-target overage for the v2 "
            "soft controller"),
    )
    parser.add_argument(
        "--cross-layer-critical-pressure-fraction", type=float, default=2.0,
        help=(
            "normalized safety-signal pressure at which v2 retains a hard "
            "critical guard"),
    )
    parser.add_argument(
        "--shared-fabric-control-mode",
        choices=("disabled", "global_budget_v3"), default="disabled",
    )
    parser.add_argument("--shared-remote-requests-capacity", type=int, default=0)
    parser.add_argument("--shared-remote-kv-bytes-capacity", type=int, default=0)
    parser.add_argument("--shared-remote-semantic-ops-capacity", type=int, default=0)
    parser.add_argument(
        "--shared-remote-limit-floor-fraction", type=float, default=0.25,
    )
    parser.add_argument("--shared-remote-stagger-max-us", type=int, default=2_000)
    parser.add_argument(
        "--prewarmed-pair-count", type=int, choices=(1, 2), default=None,
        help="prewarmed inference pairs available to the global controller",
    )
    parser.add_argument(
        "--maximum-active-pairs", type=int, choices=(1, 2), default=None,
        help="maximum inference pairs the controller may activate",
    )
    parser.add_argument(
        "--overload-action",
        choices=("reject_new_request", "endpoint_queue_lease"),
        default="reject_new_request",
        help=(
            "global overload action; endpoint_queue_lease forwards only "
            "tenants explicitly enabled below to native vLLM waiting queues"
        ),
    )
    parser.add_argument(
        "--endpoint-queue-debt-mode",
        choices=(
            "disabled",
            "work_conserving_endpoint_queue_v1",
            "completion_liveness_endpoint_queue_v2",
        ),
        default="disabled",
        help=(
            "when enabled, an explicit queue lease may carry measured "
            "endpoint-window debt into the native vLLM waiting queue; "
            "shared remote-fabric limits remain hard for hard-window or "
            "explicit transport-pressure envelopes"
        ),
    )
    parser.add_argument(
        "--endpoint-queue-admission-mode",
        choices=("after_timeout", "headroom_first_v1"),
        default="after_timeout",
        help=(
            "when headroom_first_v1 is selected, a tenant-opted-in queue "
            "lease may enter the native endpoint queue immediately after "
            "a fresh global snapshot proves service-lane headroom"
        ),
    )
    parser.add_argument(
        "--endpoint-queue-capacity", type=int, default=None,
        help=(
            "bounded TEMPO-owned endpoint queue leases; defaults to the "
            "global ingress queue capacity"
        ),
    )
    parser.add_argument(
        "--queue-lease-on-timeout-tenants",
        default="",
        help=(
            "comma-separated tenant IDs allowed to receive an explicit "
            "endpoint queue lease after global admission timeout"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prewarmed_pair_count = (
        args.pair_count
        if args.prewarmed_pair_count is None
        else args.prewarmed_pair_count
    )
    maximum_active_pairs = (
        args.pair_count
        if args.maximum_active_pairs is None
        else args.maximum_active_pairs
    )
    if prewarmed_pair_count > args.pair_count:
        raise ValueError("prewarmed pair count exceeds pair count")
    if not 1 <= maximum_active_pairs <= args.pair_count:
        raise ValueError("maximum active pairs is outside pair count")
    if args.remote_semantic_ops_safety_reserve < 0:
        raise ValueError("remote semantic-op safety reserve must be non-negative")
    if args.controller_maximum_queue_wait_ns <= 0:
        raise ValueError("controller maximum queue wait must be positive")
    if args.controller_queue_capacity <= 0:
        raise ValueError("controller queue capacity must be positive")
    if args.telemetry_freshness_ns <= 0:
        raise ValueError("telemetry freshness must be positive")
    if args.telemetry_refresh_timeout_ns <= 0:
        raise ValueError("telemetry refresh timeout must be positive")
    if args.telemetry_refresh_timeout_ns > args.telemetry_freshness_ns:
        raise ValueError("telemetry refresh timeout exceeds freshness")
    if args.telemetry_maximum_collection_span_ns <= 0:
        raise ValueError("telemetry collection span must be positive")
    if args.telemetry_maximum_collection_span_ns > args.telemetry_refresh_timeout_ns:
        raise ValueError("telemetry collection span exceeds refresh timeout")
    queue_lease_tenants = {
        value.strip() for value in args.queue_lease_on_timeout_tenants.split(",")
        if value.strip()
    }
    known_tenants = {"latency", "interactive", "batch", "background"}
    if not queue_lease_tenants <= known_tenants:
        raise ValueError("queue-lease tenant is not a known business tenant")
    if queue_lease_tenants and args.overload_action != "endpoint_queue_lease":
        raise ValueError(
            "queue-lease tenants require --overload-action endpoint_queue_lease"
        )
    for name in (
        "cross_layer_remote_limit_floor_fraction",
        "cross_layer_local_limit_floor_fraction",
    ):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in (0, 1]")
    if args.cross_layer_stagger_max_us < 0:
        raise ValueError("cross-layer stagger max must be non-negative")
    if args.telemetry_stale_grace_ns < 0:
        raise ValueError("telemetry stale grace must be non-negative")
    if args.interactive_telemetry_stale_grace_ns < 0:
        raise ValueError(
            "interactive telemetry stale grace must be non-negative")
    if args.cross_layer_shadow_price_ms < 0.0:
        raise ValueError("cross-layer shadow price must be non-negative")
    if args.cross_layer_critical_pressure_fraction < 1.0:
        raise ValueError(
            "cross-layer critical pressure fraction must be at least one")
    if (
        args.cross_layer_control_mode == "soft_shadow_price_v2"
        and args.cross_layer_shadow_price_ms <= 0.0
    ):
        raise ValueError(
            "soft-shadow-price-v2 requires a positive cross-layer shadow price")
    for name in (
        "proactive_scale_up_queue_fraction",
        "proactive_scale_up_wait_fraction",
    ):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in (0, 1]")
    if args.proactive_scale_up_active_pair_penalty_ms < 0.0:
        raise ValueError(
            "proactive-scale-up-active-pair-penalty-ms must be non-negative")
    if not 0.0 <= args.survivor_capacity_reserve_fraction < 1.0:
        raise ValueError("survivor capacity reserve fraction must be in [0, 1)")
    if args.survivor_reserve_bypass_min_weight < 0.0:
        raise ValueError("survivor reserve bypass weight must be non-negative")
    if not 0.0 < args.shared_remote_limit_floor_fraction <= 1.0:
        raise ValueError("shared remote limit floor fraction must be in (0, 1]")
    if args.shared_remote_stagger_max_us < 0:
        raise ValueError("shared remote stagger max must be non-negative")
    for name in (
        "shared_remote_requests_capacity",
        "shared_remote_kv_bytes_capacity",
        "shared_remote_semantic_ops_capacity",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    elastic = load_elastic_profile(args.elastic_profile.resolve())
    endpoint = load_endpoint_service_profile(args.endpoint_profile.resolve())
    capability = json.loads(
        args.capability_manifest.resolve().read_text(encoding="utf-8"))
    if not (
        capability.get("passed") is True
        and capability.get("native_only") is True
        and capability.get("node_count") == 4
        and capability.get("gpu_count") == 16
        and capability.get("transport_contract") == TRANSPORT
        and capability.get("privileged_nic_control") is False
    ):
        raise ValueError("native capability manifest is not policy-eligible")
    model_sha = _sha256(args.model_config.resolve())
    if model_sha != capability.get("model_config_sha256"):
        raise ValueError("model config differs from native capability receipt")
    if endpoint.elastic_profile_fingerprint_sha256 != elastic.fingerprint_sha256:
        raise ValueError("endpoint and elastic profile identity differs")
    capacities = []
    for pair_index in range(args.pair_count):
        capacities.append({
            "pair_index": pair_index,
            "decode_tokens": 4_096,
            "active_sequences": 16,
            "endpoint_requests": 16,
            "local_prefill_token_ms": (
                endpoint.controller.local_token_ms_window),
            "remote_prefill_token_ms": (
                endpoint.controller.remote_prefill_token_ms_window),
            "remote_kv_bytes": endpoint.controller.remote_kv_bytes_window,
            "remote_semantic_ops": (
                endpoint.controller.remote_semantic_ops_window),
        })
    raw = {
        "schema": SCHEMA,
        "profile_id": args.profile_id,
        "deployment_scope": "discovery",
        "transport": TRANSPORT,
        "topology": {
            "node_count": 4,
            "gpu_count": 16,
            "pair_count": args.pair_count,
            "prewarmed_pair_count": prewarmed_pair_count,
            "native_only": True,
            "route_immutable": True,
            "privileged_nic_control": False,
        },
        "causality": {
            "telemetry_clock": "frontend_perf_counter_interval_start",
            "decoder_credit_scope": "request_start_to_http_eof",
            "endpoint_credit_scope": "route_commit_to_first_response",
            "phase_label_policy_input": False,
            "physical_switch_label_policy_input": False,
            "future_arrivals_policy_input": False,
            "oracle_policy_input": False,
        },
        "identity": {
            "router_schema": ROUTER_SCHEMA,
            "endpoint_profile_schema": endpoint.schema,
            "endpoint_profile_id": endpoint.profile_id,
            "endpoint_profile_fingerprint_sha256": (
                endpoint.fingerprint_sha256),
            "endpoint_profile_deployment_scope": endpoint.deployment_scope,
            "elastic_profile_fingerprint_sha256": elastic.fingerprint_sha256,
            "workload_manifest_sha256": endpoint.workload_manifest_sha256,
            "model_config_sha256": model_sha,
        },
        "telemetry": {
            "agent_epoch_source": "slurm_job_id_frontend_start_ns",
            "freshness_ns": args.telemetry_freshness_ns,
            "refresh_timeout_ns": args.telemetry_refresh_timeout_ns,
            "maximum_collection_span_ns": (
                args.telemetry_maximum_collection_span_ns),
            "tokenizer_timeout_ns": 5_000_000_000,
            "controller_generation": 0,
            "endpoint_feedback_mode": "adaptive",
            "endpoint_routing_policy": "semantic_epoch_v1",
            "scheduler_observation_required": True,
        },
        "capacities": capacities,
        "tenants": [
            {
                "tenant_id": "latency", "weight": 4.0,
                "ttft_slo_ms": 1_000.0, "tpot_slo_ms": 100.0,
                "e2e_slo_ms": 4_000.0, "maximum_queue_wait_ns": 500_000_000,
                "minimum_service_fraction": 0.15,
                "queue_lease_on_timeout": "latency" in queue_lease_tenants,
            },
            {
                "tenant_id": "interactive", "weight": 2.0,
                "ttft_slo_ms": 2_000.0, "tpot_slo_ms": 150.0,
                "e2e_slo_ms": 8_000.0, "maximum_queue_wait_ns": 1_000_000_000,
                "minimum_service_fraction": 0.15,
                "telemetry_stale_grace_ns": (
                    args.interactive_telemetry_stale_grace_ns),
                "queue_lease_on_timeout": "interactive" in queue_lease_tenants,
            },
            {
                "tenant_id": "batch", "weight": 1.0,
                "ttft_slo_ms": 3_000.0, "tpot_slo_ms": 250.0,
                "e2e_slo_ms": 16_000.0, "maximum_queue_wait_ns": 2_000_000_000,
                "minimum_service_fraction": 0.10,
                "queue_lease_on_timeout": "batch" in queue_lease_tenants,
            },
            {
                "tenant_id": "background", "weight": 0.5,
                "ttft_slo_ms": 5_000.0, "tpot_slo_ms": 400.0,
                "e2e_slo_ms": 30_000.0, "maximum_queue_wait_ns": 5_000_000_000,
                "minimum_service_fraction": 0.05,
                "queue_lease_on_timeout": "background" in queue_lease_tenants,
            },
        ],
        "controller": {
            "queue_capacity": args.controller_queue_capacity,
            "minimum_active_pairs": 1,
            "maximum_active_pairs": maximum_active_pairs,
            "scale_up_utilization": 0.75,
            "scale_down_idle_ns": 5_000_000_000,
            "utilization_penalty_ms": 100.0,
            "activation_penalty_ms": 1.0,
            "probe_penalty_ms": 10.0,
            "maximum_queue_wait_ns": args.controller_maximum_queue_wait_ns,
            "telemetry_stale_grace_ns": args.telemetry_stale_grace_ns,
            "remote_semantic_ops_safety_reserve": (
                args.remote_semantic_ops_safety_reserve),
            "proactive_scale_up_queue_fraction": (
                args.proactive_scale_up_queue_fraction),
            "proactive_scale_up_wait_fraction": (
                args.proactive_scale_up_wait_fraction),
            "proactive_scale_up_active_pair_penalty_ms": (
                args.proactive_scale_up_active_pair_penalty_ms),
            "route_failure_quarantine_mode": (
                args.route_failure_quarantine_mode),
            "telemetry_failure_quarantine_mode": (
                args.telemetry_failure_quarantine_mode),
            "telemetry_failure_quarantine_scope": (
                args.telemetry_failure_quarantine_scope),
            "survivor_capacity_reserve_fraction": (
                args.survivor_capacity_reserve_fraction),
            "survivor_reserve_bypass_min_weight": (
                args.survivor_reserve_bypass_min_weight),
            "cross_layer_remote_limit_floor_fraction": (
                args.cross_layer_remote_limit_floor_fraction),
            "cross_layer_local_limit_floor_fraction": (
                args.cross_layer_local_limit_floor_fraction),
            "cross_layer_stagger_max_us": args.cross_layer_stagger_max_us,
            "cross_layer_control_mode": args.cross_layer_control_mode,
            "cross_layer_shadow_price_ms": args.cross_layer_shadow_price_ms,
            "cross_layer_critical_pressure_fraction": (
                args.cross_layer_critical_pressure_fraction),
            "shared_fabric_control_mode": args.shared_fabric_control_mode,
            "shared_remote_requests_capacity": (
                args.shared_remote_requests_capacity),
            "shared_remote_kv_bytes_capacity": (
                args.shared_remote_kv_bytes_capacity),
            "shared_remote_semantic_ops_capacity": (
                args.shared_remote_semantic_ops_capacity),
            "shared_remote_limit_floor_fraction": (
                args.shared_remote_limit_floor_fraction),
            "shared_remote_stagger_max_us": args.shared_remote_stagger_max_us,
            "overload_action": args.overload_action,
            "endpoint_queue_debt_mode": args.endpoint_queue_debt_mode,
            "endpoint_queue_admission_mode": (
                args.endpoint_queue_admission_mode),
            **(
                {"endpoint_queue_capacity": args.endpoint_queue_capacity}
                if args.endpoint_queue_capacity is not None else {}
            ),
        },
    }
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_global_profile(output)
    if loaded.fingerprint_sha256 != raw["fingerprint_sha256"]:
        raise RuntimeError("written global profile did not round-trip")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
