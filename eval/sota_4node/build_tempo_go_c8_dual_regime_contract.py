#!/usr/bin/env python3
"""Freeze the source-bound C8 local-protection/remote-activation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import load_global_profile


SCHEMA = "tempo-go-c8-dual-regime-contract-v1"
BASE_SCHEMA = "tempo-go-c7-joint-control-contract-v1"
REMOTE_REGIME = "dual_decoder_hot_p_only_remote_favorable"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--global-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-rate-per-decoder", type=float, default=22.4)
    parser.add_argument("--p-only-pool-per-owner", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    base_path = args.base.resolve()
    global_profile_path = args.global_profile.resolve()
    output = args.output.resolve()
    _require(base_path.is_file(), "base C7 contract is missing")
    _require(global_profile_path.is_file(), "C8 global profile is missing")
    _require(not output.exists(), "refusing to overwrite C8 contract")
    _require(args.local_rate_per_decoder > 0.0,
             "local rate per decoder must be positive")
    _require(1 <= args.p_only_pool_per_owner <= 16,
             "P_ONLY pool per owner must be in [1, 16]")
    raw = json.loads(base_path.read_text(encoding="utf-8"))
    _require(raw.get("schema") == BASE_SCHEMA, "base C7 schema differs")
    raw["schema"] = SCHEMA
    section = raw["joint_control"]
    base_global_spec = section.get("global_profile")
    _require(isinstance(base_global_spec, dict),
             "base C7 global profile is missing")
    base_global_path = (
        repo_root / str(base_global_spec.get("path", ""))).resolve()
    _require(base_global_path.is_file(), "base C7 global profile path differs")
    base_global_profile = load_global_profile(base_global_path)
    global_profile = load_global_profile(global_profile_path)
    global_config = global_profile.orchestrator_config()
    base_telemetry = base_global_profile.telemetry
    telemetry = global_profile.telemetry
    _require(
        global_config.priority_service_lane_mode in {
            "vllm_priority_remote_cache_v1",
            "vllm_priority_business_dual_route_v2",
        }
        and global_config.priority_service_lane_capacity > 0
        and global_config.priority_service_lane_min_admission_priority > 0
        and global_config.priority_service_lane_priority in {-2, -1},
        "C8 global profile does not bind the priority service lane",
    )
    _require(
        global_config.decoder_business_admission_mode == "priority_drain_v1"
        and global_config.decoder_business_background_max_wait_ns > 0,
        "C8 global profile does not bind decoder business admission",
    )
    _require(
        global_config.mesh_near_tie_source_balance_mode
        == "telemetry_uncertainty_virtual_service_v1"
        and 0.0
        < global_config.mesh_near_tie_source_balance_uncertainty_fraction
        <= 1.0,
        "C8 global profile does not bind telemetry near-tie source balance",
    )
    decoder_background_limits = [
        item.resources.active_sequences
        - global_config.priority_service_lane_capacity
        for item in sorted(
            global_profile.capacities, key=lambda value: value.pair_index)
    ]
    _require(
        len(decoder_background_limits) == 2
        and all(value > 0 for value in decoder_background_limits),
        "C8 decoder background limits are invalid",
    )
    _require(
        global_profile.identity == base_global_profile.identity
        and global_profile.topology == base_global_profile.topology
        and global_profile.causality == base_global_profile.causality
        and global_profile.capacities == base_global_profile.capacities
        and global_profile.tenants == base_global_profile.tenants,
        "C8 global profile changed identity/topology/business priors",
    )
    _require(
        telemetry.agent_epoch_source == base_telemetry.agent_epoch_source
        and telemetry.tokenizer_timeout_ns
        == base_telemetry.tokenizer_timeout_ns
        and telemetry.controller_generation
        == base_telemetry.controller_generation
        and telemetry.endpoint_feedback_mode
        == base_telemetry.endpoint_feedback_mode
        and telemetry.endpoint_routing_policy
        == base_telemetry.endpoint_routing_policy
        and telemetry.scheduler_observation_required
        == base_telemetry.scheduler_observation_required
        and telemetry.maximum_collection_span_ns
        <= telemetry.refresh_timeout_ns
        <= telemetry.freshness_ns,
        "C8 global profile changed telemetry identity or has invalid windows",
    )
    section["global_profile"] = {
        "path": global_profile_path.relative_to(repo_root).as_posix(),
        "sha256": _sha256(global_profile_path),
        "fingerprint_sha256": global_profile.fingerprint_sha256,
    }
    section["arms"] = [
        {"kind": "fixed", "name": "fixed_local_d0"},
        {"kind": "fixed", "name": "fixed_local_d1"},
        {"kind": "fixed", "name": "fixed_remote_p0d1"},
        {"kind": "fixed", "name": "fixed_remote_p1d0"},
        {"kind": "request_baseline", "name": "predictor"},
        {"kind": "request_baseline", "name": "queue_gpu"},
        {
            "kind": "managed_cross_layer",
            "name": "full_c7_managed_background",
        },
    ]
    section["headline_full_arm"] = "full_c7_managed_background"
    prior_blocks = {
        str(row["name"]): dict(row) for row in section["blocks"]
    }
    required_prior = {
        "00_control_a",
        "01_remote_cool_hot_d0",
        "02_combined_hot_d0",
        "03_remote_cool_hot_d1",
        "04_combined_hot_d1",
        "05_control_b",
    }
    _require(set(prior_blocks) == required_prior,
             "base C7 activation matrix differs")
    control_b = prior_blocks.pop("05_control_b")
    control_b["name"] = "06_control_b"
    remote_block = {
        "name": "05_p_only_dual_decoder_hot",
        # Keep one scalar label for backward-compatible raw receipts, while
        # the C8 analyzer and workload use the explicit two-decoder vector.
        "hot_decoder_index": 0,
        "hot_decoder_indices": [0, 1],
        "remote_aggressor_rate_per_s": 0.0,
        "remote_source_indices": [0, 1],
        "local_aggressor_rate_per_s": args.local_rate_per_decoder,
        "local_aggressor_decoder_indices": [0, 1],
        "pressure_regime": REMOTE_REGIME,
        "victim_cache_state": "p_only",
        "managed_background": False,
        "p_only_pool_per_owner": args.p_only_pool_per_owner,
        "p_only_preseed_before_measurement": True,
        "controller_receives_regime_label": False,
    }
    section["blocks"] = [
        prior_blocks["00_control_a"],
        prior_blocks["01_remote_cool_hot_d0"],
        prior_blocks["02_combined_hot_d0"],
        prior_blocks["03_remote_cool_hot_d1"],
        prior_blocks["04_combined_hot_d1"],
        remote_block,
        control_b,
    ]
    section["remote_activation"] = {
        "schema": "tempo-go-c8-remote-activation-workload-v1",
        "source_evidence": [
            "C1 decoder-local-hot fixed crossover",
            "C4 P_ONLY physical seed and exact full-source-hit probe",
        ],
        "local_rate_per_decoder": args.local_rate_per_decoder,
        "total_local_rate_per_s": 2 * args.local_rate_per_decoder,
        "local_decoder_indices": [0, 1],
        "remote_background_rate_per_s": 0.0,
        "victim_prompt_tokens": section["victim"]["prompt_tokens"],
        "victim_output_tokens": section["victim"]["output_tokens"],
        "victim_cache_state": "p_only",
        "p_only_pool_per_owner": args.p_only_pool_per_owner,
        "replicated_prefill_owners": [0, 1],
        "physical_preseed_outside_measurement": True,
        "exact_source_hit_required": True,
        "same_offered_population_in_every_arm": True,
        "controller_does_not_receive_regime_label": True,
        "vllm_scheduling_policy": "priority",
        "managed_remote_priority": (
            global_config.priority_service_lane_priority),
        "priority_service_lane_mode": (
            global_config.priority_service_lane_mode),
        "priority_service_lane_capacity_per_decoder": (
            global_config.priority_service_lane_capacity),
        "priority_service_lane_min_admission_priority": (
            global_config.priority_service_lane_min_admission_priority),
        "decoder_business_admission_mode": (
            global_config.decoder_business_admission_mode),
        "decoder_background_concurrency_limits": decoder_background_limits,
        "decoder_background_max_wait_ns": (
            global_config.decoder_business_background_max_wait_ns),
        "decoder_background_requests_are_delayed_not_dropped": True,
        "mesh_near_tie_source_balance_mode": (
            global_config.mesh_near_tie_source_balance_mode),
        "mesh_near_tie_source_balance_uncertainty_fraction": (
            global_config.
            mesh_near_tie_source_balance_uncertainty_fraction),
        "mesh_near_tie_source_balance_is_not_a_route_quota": True,
        "telemetry_freshness_ns": telemetry.freshness_ns,
        "telemetry_refresh_timeout_ns": telemetry.refresh_timeout_ns,
        "telemetry_maximum_collection_span_ns": (
            telemetry.maximum_collection_span_ns),
        "telemetry_per_fetch_timeout_ns": min(
            telemetry.refresh_timeout_ns,
            max(1_000_000, telemetry.maximum_collection_span_ns // 2),
        ),
        "all_endpoint_fetch_failure_policy": (
            "preserve_last_complete_batch_then_tenant_stale_grace"),
        "baseline_request_priority": 0,
        "purpose": (
            "make decoder-local prefill the measured bottleneck, preserve a "
            "bounded business decoder lane before vLLM queue saturation, and "
            "keep official LMCache P_ONLY reuse physically available"
        ),
    }
    section["remote_activation_gates"] = {
        "minimum_full_remote_fraction": 0.50,
        "minimum_cross_pair_remote_fraction": 0.10,
        "best_fixed_slo_retention_fraction": 0.95,
        "best_fixed_p99_ratio_ceiling": 1.10,
    }
    raw["candidate"] = {
        "id": "tempo-go-c8-dual-regime-v2-source-balanced",
        "base_contract": base_path.relative_to(repo_root).as_posix(),
        "purpose": (
            "preserve business-aware local protection and add a causal "
            "remote-favorable P_ONLY regime with telemetry-uncertainty-bound "
            "source/edge virtual service under dual decoder-local pressure"
        ),
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    raw["purpose"] = (
        "C8 same-population actual-vLLM dual-regime discovery: managed local "
        "protection plus official-LMCache remote activation"
    )

    inventory = dict(raw["source_inventory"])
    required_sources = (
        "eval/sota_4node/analyze_tempo_go_c8_dual_regime.py",
        "eval/sota_4node/run_tempo_go_c8_dual_regime_client.py",
        "eval/sota_4node/vllm_lmcache_tempo_go_c8_dual_regime_node.py",
        "eval/sota_4node/c8_dual_regime_node_entry.sh",
        "eval/sota_4node/run_tempo_go_c8_dual_regime_in_allocation.sh",
        "eval/sota_4node/require_perlmutter_4node_4h_interactive.sh",
        "eval/sota_4node/build_tempo_go_c8_dual_regime_contract.py",
        "eval/sota_4node/build_tempo_go_c8_priority_service_lane_profile.py",
        "tempo/pd_global_candidates.py",
        "tempo/pd_global_orchestrator.py",
        "tempo/pd_global_coordinator.py",
        "tempo/pd_global_profile.py",
        "tempo/pd_global_telemetry.py",
        "eval/sota_4node/tempo_pd_elastic_frontend.py",
        "eval/sota_4node/tempo_pd_elastic_router.py",
        "eval/sota_4node/run_tempo_go_c5_stream_client.py",
        "eval/sota_4node/run_tempo_go_c6_stream_client.py",
        "eval/sota_4node/run_tempo_go_c7_joint_control_client.py",
    )
    for relative in tuple(inventory) + required_sources:
        source = repo_root / relative
        _require(source.is_file(), f"C8 source file is missing: {relative}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
