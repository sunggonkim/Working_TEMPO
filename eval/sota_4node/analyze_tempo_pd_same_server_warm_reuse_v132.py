#!/usr/bin/env python3
"""Fail-closed production analysis for the arm-isolated warm-reuse crossover."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

from eval.sota_4node import analyze_tempo_pd_same_server_balanced_v71 as base


_ORDER = (
    "fixed_local", "tempo", "lmcache_remote",
    "lmcache_remote", "tempo", "fixed_local",
)
_ARM_KEY = {"fixed_local": "local", "tempo": "tempo", "lmcache_remote": "remote"}
_STABLE_OFFSET = {"local": 100, "tempo": 200, "remote": 300}
_ORIGINAL_LOAD = base._load
_ACTUAL_CONTRACTS: dict[int, dict] = {}
_MEASURED_PROMPTS: dict[str, list[dict[str, str]]] = {arm: [] for arm in _ORDER}


def _prompt_hashes(value: dict, prefix: str) -> dict[str, str]:
    result = {}
    for row in value.get("requests", []):
        request_id = row.get("request_id")
        digest = row.get("prompt_sha256")
        if not isinstance(request_id, str) or not request_id.startswith(prefix):
            raise ValueError("warm-reuse request prefix mismatch")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("warm-reuse prompt hash missing")
        result[request_id[len(prefix):]] = digest
    if not result:
        raise ValueError("warm-reuse prompt hashes missing")
    return result


def _validate_contract(value: dict, expected_arm: str, expected_phase: str) -> dict:
    contract = value.get("same_server_balanced_contract")
    if not isinstance(contract, dict):
        raise ValueError("warm-reuse contract missing")
    arm_key = _ARM_KEY[expected_arm]
    if contract.get("arm") != expected_arm or contract.get("phase") != expected_phase:
        raise ValueError("warm-reuse arm/phase mismatch")
    if contract.get("nonce_offset") != _STABLE_OFFSET[arm_key]:
        raise ValueError("warm-reuse stable nonce mismatch")
    for key, expected in (
        ("cache_keys_disjoint_across_all_blocks", False),
        ("cache_keys_reused_within_arm", True),
        ("cache_keys_disjoint_across_arms", True),
    ):
        if contract.get(key) is not expected:
            raise ValueError(f"warm-reuse {key} mismatch")
    if contract.get("cache_reuse_contract") != (
            "same-prompt-warm-and-measured-within-arm-v131"):
        raise ValueError("warm-reuse identity mismatch")
    return contract


def _load(path: Path) -> dict:
    value = _ORIGINAL_LOAD(path)
    contract = value.get("same_server_balanced_contract")
    if not isinstance(contract, dict):
        raise ValueError("measured contract missing")
    sequence_index = contract.get("sequence_index")
    if type(sequence_index) is not int or not 0 <= sequence_index < len(_ORDER):
        raise ValueError("measured sequence index mismatch")
    expected_arm = _ORDER[sequence_index]
    actual = _validate_contract(value, expected_arm, "measured")
    _ACTUAL_CONTRACTS[sequence_index] = copy.deepcopy(actual)
    _MEASURED_PROMPTS[expected_arm].append(
        _prompt_hashes(value, str(actual["request_prefix"]))
    )
    adapted = copy.deepcopy(value)
    adapted_contract = adapted["same_server_balanced_contract"]
    adapted_contract["nonce_offset"] = 400 + 100 * sequence_index
    adapted_contract["cache_keys_disjoint_across_all_blocks"] = True
    return adapted


def _stage_root() -> Path:
    return Path(sys.argv[sys.argv.index("--stage-root") + 1]).resolve()


def _output() -> Path:
    return Path(sys.argv[sys.argv.index("--output") + 1]).resolve()


def _validate_warm_and_measured(stage_root: Path) -> bool:
    warm_order = ("fixed_local", "lmcache_remote", "tempo")
    warm_maps = {}
    for sequence_index, arm in enumerate(warm_order):
        path = (stage_root / "same_server_balanced_warm" /
                f"{sequence_index:02d}_{arm}_r0.raw.json")
        value = json.loads(path.read_text(encoding="utf-8"))
        contract = _validate_contract(value, arm, "warm")
        warm_maps[arm] = _prompt_hashes(value, str(contract["request_prefix"]))
    if set(_ACTUAL_CONTRACTS) != set(range(6)):
        raise ValueError("exact six measured contracts required")
    warm_hashes = {
        arm: sorted(prompt_map.values()) for arm, prompt_map in warm_maps.items()
    }
    for arm in set(_ORDER):
        if len(_MEASURED_PROMPTS[arm]) != 2:
            raise ValueError("exact two measured prompt maps per arm required")
        if any(sorted(prompt_map.values()) != warm_hashes[arm]
               for prompt_map in _MEASURED_PROMPTS[arm]):
            raise ValueError("prompt keys were not reused within arm")
    hash_lists = list(warm_hashes.values())
    for left in range(len(hash_lists)):
        for right in range(left + 1, len(hash_lists)):
            if hash_lists[left] == hash_lists[right]:
                raise ValueError("prompt keys were not isolated across arms")
    return True


def main() -> int:
    original = base._load
    base._load = _load
    try:
        status = base.main()
    finally:
        base._load = original
    reuse_valid = _validate_warm_and_measured(_stage_root())
    output = _output()
    value = json.loads(output.read_text(encoding="utf-8"))
    value["schema"] = "tempo-pd-same-server-warm-reuse-analysis-132"
    value["contracts_by_sequence"] = [
        _ACTUAL_CONTRACTS[index] for index in range(6)
    ]
    tempo, local, remote = value["tempo"], value["fixed_local"], value["lmcache_remote"]
    tp, lp, rp = (row["performance"] for row in (tempo, local, remote))
    pair_local = value["paired_tempo_minus_local"]
    pair_remote = value["paired_tempo_minus_lmcache"]
    routes, reasons = tempo["routes"], tempo["reasons"]
    gates = {
        "arm_isolated_warm_reuse_contract": reuse_valid,
        "exact_normalized_workload_schedule_outputs": value["gates"][
            "exact_normalized_workload_schedule_outputs"],
        "fixed_local_routes_48_local": local["routes"] == {
            "decoder_local_recompute_or_cache": 48},
        "lmcache_routes_48_remote": remote["routes"] == {
            "remote_prefill_live_kv": 48},
        "tempo_routes_exactly_48": sum(routes.values()) == 48 and set(routes) <= {
            "decoder_local_recompute_or_cache", "remote_prefill_live_kv"},
        "production_reason_geometry": (
            sum(count for reason, count in reasons.items()
                if reason.endswith("output16_direct_local_fast_path")) == 12
            and sum(count for reason, count in reasons.items()
                    if reason.endswith("output128_direct_local_fast_path")) == 12
            and sum(count for reason, count in reasons.items()
                    if "mean_pair_interval_ns=" in reason) == 24),
        "all_tempo_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_goodput_retains_95pct_local": (
            tp["slo_goodput"]["request_goodput_per_s"] >=
            0.95 * lp["slo_goodput"]["request_goodput_per_s"]),
        "tempo_goodput_beats_lmcache": (
            tp["slo_goodput"]["request_goodput_per_s"] >
            rp["slo_goodput"]["request_goodput_per_s"]),
        "tempo_throughput_beats_lmcache": (
            tp["request_throughput_per_s"] > rp["request_throughput_per_s"]),
        "tempo_paired_local_noninferior": (
            pair_local["e2e_win_count"] >= 24
            and pair_local["e2e_delta_median_ms"] <= 10.0),
        "tempo_paired_beats_lmcache": (
            pair_remote["e2e_win_count"] >= 25
            and pair_remote["e2e_delta_median_ms"] < 0.0),
        "tempo_e2e_p99_within_5pct_local": (
            tp["e2e_ms"]["p99"] <= 1.05 * lp["e2e_ms"]["p99"]),
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    value["gates"] = gates
    value["passes"] = all(gates.values())
    value["verdict"] = (
        "promising_production_warm_reuse_controller" if value["passes"]
        else "reject_production_warm_reuse_controller")
    value["claim_boundary"] = (
        "One live Qwen2.5-7B TP4+TP4 P/D lifecycle; arm-isolated keys repeated "
        "across warmup and two measured replicates; production router; 48 req/s."
    )
    output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"],
                      "failed": [k for k, passed in gates.items() if not passed]},
                     sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
