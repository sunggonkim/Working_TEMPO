#!/usr/bin/env python3
"""Freeze a C7 remote-cool/combined-hot activation-matrix contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--remote-rate-per-s", type=float, default=7.8,
        help="remote aggressor rate used by selective and combined blocks")
    parser.add_argument(
        "--local-rate-per-s", type=float, default=None,
        help="local decoder aggressor rate; defaults to remote rate")
    parser.add_argument(
        "--ingress-policy", choices=("shared_pool", "interactive_reserved"),
        default="shared_pool",
        help="client ingress policy frozen into the activation matrix")
    parser.add_argument(
        "--interactive-reserved-workers", type=int, default=0,
        help="workers reserved for interactive requests under the selected policy")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    base = args.base.resolve()
    profile = args.profile.resolve()
    output = args.output.resolve()
    _require(base.is_file(), "base C7 contract is missing")
    _require(profile.is_file(), "goodput profile is missing")
    _require(not output.exists(), "refusing to overwrite activation matrix contract")
    _require(args.remote_rate_per_s > 0.0, "remote rate must be positive")
    local_rate = (
        args.remote_rate_per_s
        if args.local_rate_per_s is None else args.local_rate_per_s)
    _require(local_rate >= 0.0, "local rate must be non-negative")
    if args.ingress_policy == "shared_pool":
        _require(args.interactive_reserved_workers == 0,
                 "shared_pool cannot reserve interactive workers")
    else:
        _require(args.interactive_reserved_workers > 0,
                 "interactive_reserved requires reserved workers")
    _require(repo_root in profile.parents, "profile must be in repository")

    raw = json.loads(base.read_text(encoding="utf-8"))
    _require(raw.get("schema") == "tempo-go-c7-joint-control-contract-v1",
             "base contract schema differs")
    section = raw["joint_control"]
    arms = section.get("arms")
    _require(isinstance(arms, list), "C7 arm inventory is missing")
    if not any(item.get("name") == "full_c7_managed_background"
               for item in arms if isinstance(item, dict)):
        arms.append({
            "kind": "managed_cross_layer",
            "name": "full_c7_managed_background",
        })
    # The production-scale scheme owns business/background admission as well
    # as victim routing.  Keep the unmanaged full arm as an ablation, but do
    # not accidentally headline it as the global orchestrator.
    section["headline_full_arm"] = "full_c7_managed_background"
    section["blocks"] = [
        {"name": "00_control_a", "hot_decoder_index": None},
        {
            "name": "01_remote_cool_hot_d0",
            "hot_decoder_index": 0,
            "remote_aggressor_rate_per_s": args.remote_rate_per_s,
            "local_aggressor_rate_per_s": 0.0,
            "remote_source_indices": [1],
            "pressure_regime": "decoder_hot_remote_edge_selective",
        },
        {
            "name": "02_combined_hot_d0",
            "hot_decoder_index": 0,
            "remote_aggressor_rate_per_s": args.remote_rate_per_s,
            "local_aggressor_rate_per_s": local_rate,
            "remote_source_indices": [0, 1],
            "pressure_regime": "decoder_hot_remote_fabric_hot",
        },
        {
            "name": "03_remote_cool_hot_d1",
            "hot_decoder_index": 1,
            "remote_aggressor_rate_per_s": args.remote_rate_per_s,
            "local_aggressor_rate_per_s": 0.0,
            "remote_source_indices": [1],
            "pressure_regime": "decoder_hot_remote_edge_selective",
        },
        {
            "name": "04_combined_hot_d1",
            "hot_decoder_index": 1,
            "remote_aggressor_rate_per_s": args.remote_rate_per_s,
            "local_aggressor_rate_per_s": local_rate,
            "remote_source_indices": [0, 1],
            "pressure_regime": "decoder_hot_remote_fabric_hot",
        },
        {"name": "05_control_b", "hot_decoder_index": None},
    ]
    profile_rel = profile.relative_to(repo_root).as_posix()
    profile_raw = json.loads(profile.read_text(encoding="utf-8"))
    section["global_profile"] = {
        "path": profile_rel,
        "sha256": _sha256(profile),
        "fingerprint_sha256": profile_raw["fingerprint_sha256"],
    }
    section["activation_matrix"] = {
        "schema": "tempo-go-c7-remote-activation-matrix-v1",
        "remote_selective_rate_per_s": args.remote_rate_per_s,
        "remote_hot_rate_per_s": args.remote_rate_per_s,
        "local_hot_rate_per_s": local_rate,
        "local_cool_rate_per_s": 0.0,
        "remote_cool_source_indices": [1],
        "remote_hot_source_indices": [0, 1],
        "controller_does_not_receive_phase_label": True,
        "purpose": (
            "causally separate decoder/receiver pressure from remote source/fabric "
            "pressure before evaluating remote actuation"
        ),
    }
    section["ingress"] = {
        "schema": "tempo-go-c7-ingress-policy-v1",
        "policy": args.ingress_policy,
        "interactive_reserved_workers": args.interactive_reserved_workers,
        "background_workers": (
            section["max_workers"] - args.interactive_reserved_workers),
        "same_open_loop_clock": True,
        "same_offered_population": True,
        "controller_does_not_receive_ingress_lane": True,
        "purpose": (
            "prevent background executor backlog from delaying interactive admission "
            "while preserving the same exogenous arrivals"),
    }
    # The node qualification seam requires the discovery boundary to remain
    # enabled.  The matrix itself is still non-claiming through `candidate`
    # metadata and the downstream analyzer's performance_claim_allowed=false.

    inventory = dict(raw["source_inventory"])
    for relative in tuple(inventory) + (
        "eval/sota_4node/run_tempo_go_c7_joint_control_client.py",
        "eval/sota_4node/run_tempo_go_c6_decoder_victim_client.py",
        "eval/sota_4node/run_tempo_pd_contention_fixed_client.py",
        "eval/sota_4node/run_tempo_pd_stream_metrics_v1.py",
        "eval/sota_4node/vllm_lmcache_tempo_go_c6_qualification_node.py",
        "eval/sota_4node/c7_joint_control_node_entry.sh",
        "eval/sota_4node/vllm_lmcache_tempo_go_c7_joint_control_node.py",
        "tempo/pd_global_orchestrator.py",
        "tempo/pd_global_coordinator.py",
        "tempo/pd_global_agent.py",
        "tempo/pd_global_profile.py",
    ):
        source = repo_root / relative
        _require(source.is_file(), f"source inventory file is missing: {relative}")
        inventory[relative] = _sha256(source)
    raw["source_inventory"] = dict(sorted(inventory.items()))
    raw["candidate"] = {
        "id": "c7-remote-activation-matrix-v1",
        "base_contract": base.relative_to(repo_root).as_posix(),
        "purpose": (
            "separate remote/fabric-cool decoder-hot phases from combined-hot phases "
            "for causal global route/admission activation"
        ),
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print("profile_sha256", section["global_profile"]["sha256"])
    print("profile_fingerprint", section["global_profile"]["fingerprint_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
