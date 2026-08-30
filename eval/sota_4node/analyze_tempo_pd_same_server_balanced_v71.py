#!/usr/bin/env python3
"""Analyze a two-replicate order-balanced same-server crossover."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base


_ORDER = (
    "fixed_local", "tempo", "lmcache_remote",
    "lmcache_remote", "tempo", "fixed_local",
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: object required")
    return value


def _normalize(raw: dict, arm: str) -> tuple[dict, dict]:
    value = copy.deepcopy(raw)
    contract = value.get("same_server_balanced_contract")
    if not isinstance(contract, dict) or contract.get("arm") != arm:
        raise ValueError(f"{arm}: balanced contract mismatch")
    if contract.get("phase") != "measured":
        raise ValueError(f"{arm}: measured phase required")
    prefix = contract.get("request_prefix")
    prompt_sha = contract.get("base_prompt_sha256")
    if not isinstance(prefix, str) or not isinstance(prompt_sha, dict):
        raise ValueError(f"{arm}: normalization metadata missing")

    def original(request_id):
        if not isinstance(request_id, str) or not request_id.startswith(prefix):
            raise ValueError(f"{arm}: request prefix mismatch")
        return request_id[len(prefix):]

    for row in value["requests"]:
        request_id = original(row["request_id"])
        row["request_id"] = request_id
        row["prompt_sha256"] = prompt_sha[request_id]
        if isinstance(row.get("router"), dict):
            row["router"]["request_id"] = request_id
    for row in value["router_decisions"]:
        row["request_id"] = original(row["request_id"])
    value["workload"]["sha256"] = contract["base_semantic_sha256"]
    return value, contract


def _combine(rows: list[dict], arm: str) -> dict:
    if len(rows) != 2:
        raise ValueError(f"{arm}: exactly two replicates required")
    metrics = []
    outputs = {}
    contracts = {}
    routes: dict[str, int] = {}
    reasons: dict[str, int] = {}
    windows = 0.0
    for replicate, row in enumerate(rows):
        windows += row["performance"]["measurement_window_s"]
        for metric in row["request_metrics"]:
            item = copy.deepcopy(metric)
            item["request_id"] = f"r{replicate}-{item['request_id']}"
            metrics.append(item)
        for key, value in row["_outputs"].items():
            outputs[f"r{replicate}-{key}"] = value
        for key, value in row["_contracts"].items():
            contracts[f"r{replicate}-{key}"] = value
        for key, value in row["routes"].items():
            routes[key] = routes.get(key, 0) + value
        for key, value in row["reasons"].items():
            reasons[key] = reasons.get(key, 0) + value
    passed = [row for row in metrics if row["slo_pass"]]
    tokens = sum(row["completion_tokens"] for row in metrics)
    passed_tokens = sum(row["completion_tokens"] for row in passed)
    return {
        "label": arm, "mode": rows[0]["mode"],
        "model_config_sha256": rows[0]["model_config_sha256"],
        "workload_sha256": rows[0]["workload_sha256"],
        "request_count": len(metrics), "routes": routes, "reasons": reasons,
        "performance": {
            "measurement_window_s": windows,
            "request_throughput_per_s": len(metrics) / windows,
            "output_token_throughput_per_s": tokens / windows,
            "ttft_ms": base._distribution([row["ttft_ms"] for row in metrics]),
            "tpot_ms": base._distribution([row["tpot_ms"] for row in metrics]),
            "itl_ms": base._distribution([value for row in metrics for value in row["itl_ms"]]),
            "e2e_ms": base._distribution([row["e2e_ms"] for row in metrics]),
            "slo_goodput": {
                "successful_requests": len(passed),
                "success_fraction": len(passed) / len(metrics),
                "request_goodput_per_s": len(passed) / windows,
                "output_token_goodput_per_s": passed_tokens / windows,
            },
        },
        "request_metrics": metrics, "_outputs": outputs, "_contracts": contracts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.stage_root / "same_server_balanced_measured"
    parsed_by_arm = {key: [] for key in set(_ORDER)}
    contracts_by_sequence = []
    for sequence_index, arm in enumerate(_ORDER):
        replicate = sum(1 for prior in _ORDER[:sequence_index] if prior == arm)
        key = f"{sequence_index:02d}_{arm}_r{replicate}"
        normalized, contract = _normalize(_load(root / f"{key}.raw.json"), arm)
        if contract.get("sequence_index") != sequence_index or contract.get("replicate") != replicate:
            raise ValueError(f"{key}: sequence identity mismatch")
        contracts_by_sequence.append(contract)
        parsed_by_arm[arm].append(base._parse_run(
            key, normalized, ttft_slo_ms=3000, tpot_slo_ms=250, e2e_slo_ms=12000))

    first = parsed_by_arm[_ORDER[0]][0]
    exact = all(
        row["model_config_sha256"] == first["model_config_sha256"]
        and row["workload_sha256"] == first["workload_sha256"]
        and row["_contracts"] == first["_contracts"]
        and row["_outputs"] == first["_outputs"]
        for rows in parsed_by_arm.values() for row in rows
    )
    roots = {row["server_epoch_root"] for row in contracts_by_sequence}
    order_exact = (
        len(roots) == 1
        and tuple(row["arm"] for row in contracts_by_sequence) == _ORDER
        and [row["nonce_offset"] for row in contracts_by_sequence] == [400, 500, 600, 700, 800, 900]
        and all(row.get("cache_keys_disjoint_across_all_blocks") is True
                for row in contracts_by_sequence)
    )
    combined = {arm: _combine(rows, arm) for arm, rows in parsed_by_arm.items()}
    local, tempo, remote = (combined[key] for key in ("fixed_local", "tempo", "lmcache_remote"))
    lp, tp, rp = (row["performance"] for row in (local, tempo, remote))
    pair_local = base._paired(tempo, local) if exact else None
    pair_remote = base._paired(tempo, remote) if exact else None
    gates = {
        "one_live_server_order_balanced_contract": order_exact,
        "exact_normalized_workload_schedule_outputs": exact,
        "fixed_local_routes_48_local": local["routes"] == {"decoder_local_recompute_or_cache": 48},
        "lmcache_routes_48_remote": remote["routes"] == {"remote_prefill_live_kv": 48},
        "tempo_routes_32_local_16_remote": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 32, "remote_prefill_live_kv": 16},
        "all_tempo_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_goodput_beats_local": tp["slo_goodput"]["request_goodput_per_s"] > lp["slo_goodput"]["request_goodput_per_s"],
        "tempo_goodput_beats_lmcache": tp["slo_goodput"]["request_goodput_per_s"] > rp["slo_goodput"]["request_goodput_per_s"],
        "tempo_paired_majority_beats_local": pair_local is not None and pair_local["e2e_win_count"] >= 25,
        "tempo_paired_majority_beats_lmcache": pair_remote is not None and pair_remote["e2e_win_count"] >= 25,
        "tempo_paired_median_beats_local": pair_local is not None and pair_local["e2e_delta_median_ms"] < 0,
        "tempo_paired_median_beats_lmcache": pair_remote is not None and pair_remote["e2e_delta_median_ms"] < 0,
        "tempo_e2e_p99_beats_lmcache": tp["e2e_ms"]["p99"] < rp["e2e_ms"]["p99"],
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    public = lambda row: {key: value for key, value in row.items() if not key.startswith("_")}
    result = {
        "schema": "tempo-pd-same-server-balanced-analysis-71",
        "arm_order": list(_ORDER),
        "fixed_local": public(local), "tempo": public(tempo), "lmcache_remote": public(remote),
        "per_replicate": {
            arm: [public(row) for row in rows] for arm, rows in parsed_by_arm.items()
        },
        "paired_tempo_minus_local": pair_local,
        "paired_tempo_minus_lmcache": pair_remote,
        "contracts_by_sequence": contracts_by_sequence,
        "gates": gates, "passes": all(gates.values()),
        "verdict": "promising_order_balanced_controller" if all(gates.values()) else "revise_order_balanced_controller",
        "claim_boundary": (
            "One live server lifecycle, six cold-key-disjoint blocks, two replicates per arm, "
            "with measured order local/TEMPO/LMCache/LMCache/TEMPO/local."
        ),
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
