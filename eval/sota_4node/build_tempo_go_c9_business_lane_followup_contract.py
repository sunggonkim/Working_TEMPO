#!/usr/bin/env python3
"""Freeze the post-C9 dual-route business-lane discovery contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import load_global_profile


MODE = "vllm_priority_business_dual_route_v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, object]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-base", type=Path, required=True)
    parser.add_argument("--parent-c9", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--start-delay-s", type=int,
        help=(
            "override the exogenous co-job start delay; use this only when "
            "the measured vLLM startup envelope is longer than the base "
            "discovery delay"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    parent_base_path = args.parent_base.resolve()
    parent_c9_path = args.parent_c9.resolve()
    profile_path = args.profile.resolve()
    output_base = args.output_base.resolve()
    output = args.output.resolve()
    for path in (parent_base_path, parent_c9_path, profile_path):
        _require(path.is_file(), f"contract input is missing: {path}")
    _require(not output_base.exists() and not output.exists(),
             "follow-up output already exists")

    profile = load_global_profile(profile_path)
    config = profile.orchestrator_config()
    _require(
        config.priority_service_lane_mode == MODE
        and config.priority_service_lane_capacity == 8
        and config.priority_service_lane_min_admission_priority == 800
        and config.priority_service_lane_priority == -2
        and config.decoder_business_admission_mode == "priority_drain_v1",
        "dual-route business profile differs",
    )

    base = json.loads(parent_base_path.read_text(encoding="utf-8"))
    _require(base.get("schema") == "tempo-go-c8-dual-regime-contract-v1",
             "parent C8 schema differs")
    section = base["joint_control"]
    section["global_profile"] = {
        "path": profile_path.relative_to(repo_root).as_posix(),
        "sha256": _sha256(profile_path),
        "fingerprint_sha256": profile.fingerprint_sha256,
    }
    remote = section["remote_activation"]
    remote["priority_service_lane_mode"] = MODE
    remote["purpose"] = (
        "When official LMCache/NCCL fabric service is overloaded, let the "
        "global business decision lease either a cache-safe remote lane or a "
        "local vLLM priority lane while decoder background drains."
    )
    base["candidate"] = {
        "id": "tempo-go-c9-dual-route-business-lane-v2",
        "base_contract": parent_base_path.relative_to(repo_root).as_posix(),
        "supersedes_failed_preflight_contract": (
            "eval/sota_4node/tempo_go_c9_business_lane_base_contract_v4.json"),
        "failed_preflight_receipt": (
            "results/tempo_go_c9_business_lane_followup_job_57622087/"
            "00_app_global_only_a/block_failure_receipt.json"),
        "failed_preflight_was_not_a_performance_result": True,
        "purpose": (
            "Repair the measured remote-only priority-lane dead end without "
            "using phase, future-arrival, or physical-switch oracle input."
        ),
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    base["purpose"] = (
        "Post-C9 source-bound discovery for cross-layer route-symmetric "
        "business priority admission on actual four-node vLLM/LMCache."
    )
    base["claim_boundary"] = {
        "controller_performance_claim_allowed": True,
        "performance_claim_allowed": True,
        "independent_validation_claim_allowed": False,
        "purpose": (
            "The inner C8 lifecycle is executable as a source-bound discovery; "
            "the enclosing C9 follow-up contract forbids a paper claim."
        ),
    }
    base["causal_business_lane_followup"] = {
        "schema": "tempo-go-c9-business-lane-followup-v1",
        "post_hoc_from_job": "57622087",
        "discovery_only": True,
        "mechanism": MODE,
        "remote_only_mode_preserved": True,
        "local_lane_requires_global_commit": True,
        "background_arrivals_unchanged": True,
        "controller_receives_phase_or_future_arrivals": False,
        "native_c8_binding_accepts_exact_contract_mode": True,
    }
    inventory = dict(base["source_inventory"])
    followup_sources = (
        "tempo/pd_global_orchestrator.py",
        "eval/sota_4node/tempo_pd_elastic_router.py",
        "eval/sota_4node/build_tempo_go_c8_priority_service_lane_profile.py",
        "eval/sota_4node/build_tempo_go_c8_dual_regime_contract.py",
        "eval/sota_4node/build_tempo_go_c9_business_lane_followup_contract.py",
        "eval/sota_4node/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh",
        "eval/sota_4node/analyze_tempo_go_c9_causal_burst_discovery.py",
        "eval/sota_4node/c8_dual_regime_node_entry.sh",
        "eval/sota_4node/c9_gate_node_entry.sh",
        "eval/sota_4node/vllm_lmcache_tempo_go_c9_gate_node.py",
        "eval/sota_4node/vllm_lmcache_tempo_go_c8_dual_regime_node.py",
    )
    for relative in tuple(inventory) + followup_sources:
        source = (repo_root / relative).resolve()
        _require(source.is_file() and repo_root in source.parents,
                 f"follow-up source is missing: {relative}")
        inventory[relative] = _sha256(source)
    base["source_inventory"] = dict(sorted(inventory.items()))
    _write(output_base, base)

    outer = json.loads(parent_c9_path.read_text(encoding="utf-8"))
    _require(
        outer.get("schema") == "tempo-go-c9-causal-burst-discovery-v1",
        "parent C9 schema differs",
    )
    outer["purpose"] = (
        "ABBA discovery of route-symmetric business priority admission under "
        "the same actual NCCL plus official LMCache receiver-incast burst."
    )
    outer["claim_boundary"] = {
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "discovery_only": True,
        "reason": (
            "The mechanism was selected after diagnosing job 57622087; this "
            "same-allocation follow-up is causal discovery, not independent "
            "paper evidence."
        ),
    }
    outer["system_under_test"].update({
        "base_contract": output_base.relative_to(repo_root).as_posix(),
        "base_contract_sha256": _sha256(output_base),
        "node_entry": "eval/sota_4node/c9_gate_node_entry.sh",
        "source_policy": "source_bound_dual_route_business_lane_v1",
    })
    outer["execution"]["order"] = [
        {"name": "00_app_global_only_a", "arm": "app_global_only",
         "port_slot": 2600},
        {"name": "01_full_business_lane_a",
         "arm": "full_c7_managed_background", "port_slot": 2640},
        {"name": "02_full_business_lane_b",
         "arm": "full_c7_managed_background", "port_slot": 2680},
        {"name": "03_app_global_only_b", "arm": "app_global_only",
         "port_slot": 2720},
    ]
    outer["execution"]["paired_indices"] = [[0, 1], [3, 2]]
    outer["execution"]["cooldown_s"] = 30
    if args.start_delay_s is not None:
        _require(0 <= args.start_delay_s <= 1200,
                 "start delay must be between 0 and 1200 seconds")
        outer["burst"]["start_delay_s"] = args.start_delay_s
    outer["gates"].update({
        "observer_support_scope": (
            "remote_favorable_victim_global_decisions"),
        "minimum_full_supported_observer_fraction": 0.75,
        "minimum_remote_background_completion_fraction": 0.99,
        "minimum_remote_background_completion_ratio_to_blind": 0.99,
    })
    outer["mechanism"] = {
        "schema": "tempo-go-c9-dual-route-business-lane-mechanism-v2",
        "mode": MODE,
        "causal_failure_receipt": (
            "results/tempo_go_c9_causal_burst_job_57622087/"
            "campaign_failure_receipt.json"),
        "measured_dead_end": (
            "remote candidates became deadline-infeasible while local "
            "candidates hit tenant_protected_capacity_reserve; the old "
            "priority lane authorized remote cache routes only"),
        "new_action": (
            "globally committed interactive work may consume a bounded local "
            "or cache-safe remote vLLM priority lane"),
        "background_is_not_globally_dropped_in_remote_regime": True,
        "failed_v4_preflight_was_not_retried": True,
    }
    outer["source_inventory"] = {
        relative: _sha256(repo_root / relative)
        for relative in followup_sources
    }
    _write(output, outer)
    print(output_base)
    print("base_contract_sha256", _sha256(output_base))
    print(output)
    print("contract_sha256", _sha256(output))
    print("source_inventory_count", len(inventory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
