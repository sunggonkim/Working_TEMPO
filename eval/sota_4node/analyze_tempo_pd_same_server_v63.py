#!/usr/bin/env python3
"""Analyze local, TEMPO, and LMCache arms from one live server epoch."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base


_PREFIX = {
    "fixed_local": "ss-local-measured-",
    "tempo": "ss-tempo-measured-",
    "lmcache_remote": "ss-remote-measured-",
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: object required")
    return value


def _normalize(raw: dict, arm: str) -> tuple[dict, dict]:
    value = copy.deepcopy(raw)
    contract = value.get("same_server_contract")
    if not isinstance(contract, dict) or contract.get("arm") != arm:
        raise ValueError(f"{arm}: same-server contract mismatch")
    if contract.get("phase") != "measured":
        raise ValueError(f"{arm}: measured phase required")
    prefix = _PREFIX[arm]
    prompt_sha = contract.get("base_prompt_sha256")
    if not isinstance(prompt_sha, dict):
        raise ValueError(f"{arm}: base prompt hashes missing")

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.stage_root / "same_server_measured"
    paths = {
        "fixed_local": root / "fixed_local.raw.json",
        "tempo": root / "tempo.raw.json",
        "lmcache_remote": root / "lmcache_remote.raw.json",
    }
    normalized = {}
    contracts = {}
    for arm, path in paths.items():
        normalized[arm], contracts[arm] = _normalize(_load(path), arm)
    parsed = {
        arm: base._parse_run(
            arm, raw, ttft_slo_ms=3000, tpot_slo_ms=250, e2e_slo_ms=12000)
        for arm, raw in normalized.items()
    }
    local, tempo, remote = parsed["fixed_local"], parsed["tempo"], parsed["lmcache_remote"]
    lp, tp, rp = (row["performance"] for row in (local, tempo, remote))
    semantic_exact = (
        local["model_config_sha256"] == tempo["model_config_sha256"] == remote["model_config_sha256"]
        and local["workload_sha256"] == tempo["workload_sha256"] == remote["workload_sha256"]
        and local["_contracts"] == tempo["_contracts"] == remote["_contracts"]
        and local["_outputs"] == tempo["_outputs"] == remote["_outputs"]
    )
    roots = {row["server_epoch_root"] for row in contracts.values()}
    same_server_contract = (
        len(roots) == 1
        and [contracts[arm]["sequence_index"]
             for arm in ("fixed_local", "tempo", "lmcache_remote")] == [0, 1, 2]
        and {contracts[arm]["nonce_offset"] for arm in contracts} == {500, 600, 700}
        and all(row.get("cache_keys_disjoint_across_arms") is True for row in contracts.values())
        and len({tuple(row["prompt_token_counts"]) for row in contracts.values()}) == 1
    )
    gates = {
        "same_live_server_epoch_contract": same_server_contract,
        "exact_normalized_workload_schedule_outputs": semantic_exact,
        "fixed_local_routes_24_local": local["routes"] == {"decoder_local_recompute_or_cache": 24},
        "lmcache_routes_24_remote": remote["routes"] == {"remote_prefill_live_kv": 24},
        "tempo_routes_16_local_8_remote": tempo["routes"] == {
            "decoder_local_recompute_or_cache": 16, "remote_prefill_live_kv": 8},
        "all_tempo_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_goodput_beats_local": tp["slo_goodput"]["request_goodput_per_s"] > lp["slo_goodput"]["request_goodput_per_s"],
        "tempo_goodput_beats_lmcache": tp["slo_goodput"]["request_goodput_per_s"] > rp["slo_goodput"]["request_goodput_per_s"],
        "tempo_e2e_p50_beats_both": tp["e2e_ms"]["p50"] < min(lp["e2e_ms"]["p50"], rp["e2e_ms"]["p50"]),
        "tempo_e2e_p99_beats_both": tp["e2e_ms"]["p99"] < min(lp["e2e_ms"]["p99"], rp["e2e_ms"]["p99"]),
        "tempo_tpot_p99_beats_lmcache": tp["tpot_ms"]["p99"] < rp["tpot_ms"]["p99"],
    }
    public = lambda row: {key: value for key, value in row.items() if not key.startswith("_")}
    result = {
        "schema": "tempo-pd-same-server-analysis-63",
        "arm_order": ["fixed_local", "tempo", "lmcache_remote"],
        "fixed_local": public(local), "tempo": public(tempo), "lmcache_remote": public(remote),
        "paired_tempo_minus_local": base._paired(tempo, local) if semantic_exact else None,
        "paired_tempo_minus_lmcache": base._paired(tempo, remote) if semantic_exact else None,
        "same_server_contracts": contracts,
        "gates": gates, "passes": all(gates.values()),
        "verdict": "promising_same_server_controller" if all(gates.values()) else "revise_same_server_controller",
        "claim_boundary": (
            "One bounded same-server campaign with cold-key-disjoint arms; order is local, TEMPO, LMCache."
        ),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": gates}, sort_keys=True))
    return 0 if result["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
