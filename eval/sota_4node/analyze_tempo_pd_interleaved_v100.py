#!/usr/bin/env python3
"""Analyze one request-level-interleaved local/TEMPO/LMCache raw artifact."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base
from eval.sota_4node.finalize_tempo_pd_output16_mixed_v98 import _pairs


_MODE = {"local": "fixed_local", "tempo": "tempo_auto",
         "remote": "lmcache_always_remote"}


def _arm_raw(raw: dict, arm: str) -> dict:
    value = copy.deepcopy(raw)
    prefix = f"ssi-{arm}-"
    requests = [row for row in value["requests"] if row["request_id"].startswith(prefix)]
    decisions = [row for row in value["router_decisions"]
                 if row["request_id"].startswith(prefix)]
    if len(requests) != 48 or len(decisions) != 48:
        raise ValueError(f"{arm}: exact 48 rows required")
    contract = value["same_server_interleaved_contract"]
    prompt_sha = contract["base_prompt_sha256"]
    def normalize(request_id: str) -> tuple[str, str]:
        rest = request_id[len(prefix):]
        replicate_text, marker, base_id = rest.partition("-measured-")
        if marker != "-measured-" or replicate_text not in ("r0", "r1"):
            raise ValueError(f"{arm}: invalid request identity")
        return f"{replicate_text}-{base_id}", base_id
    for row in requests:
        request_id, base_id = normalize(row["request_id"])
        row["request_id"] = request_id
        row["prompt_sha256"] = prompt_sha[base_id]
        row["router"]["request_id"] = request_id
    for row in decisions:
        row["request_id"], _ = normalize(row["request_id"])
    value["requests"] = requests
    value["router_decisions"] = decisions
    value["run"]["mode"] = _MODE[arm]
    value["workload"]["sha256"] = contract["semantic_sha256"]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    contract = raw.get("same_server_interleaved_contract")
    if not isinstance(contract, dict) or contract.get("arm_counts") != {
            "local": 48, "tempo": 48, "remote": 48}:
        raise ValueError("interleaved contract mismatch")
    parsed = {arm: base._parse_run(
        arm, _arm_raw(raw, arm), ttft_slo_ms=3000,
        tpot_slo_ms=250, e2e_slo_ms=8000) for arm in _MODE}
    local, tempo, remote = parsed["local"], parsed["tempo"], parsed["remote"]
    exact = (tempo["_outputs"] == local["_outputs"] == remote["_outputs"])
    if not exact:
        raise ValueError("interleaved output equivalence failed")
    buckets = {}
    gates = {"exact_output_equivalence": exact}
    for bucket in ("512", "1230", "2048"):
        tl = _pairs(tempo, local, f"mix{bucket}-")
        tr = _pairs(tempo, remote, f"mix{bucket}-")
        buckets[bucket] = {"tempo_minus_local": tl, "tempo_minus_lmcache": tr}
    for bucket in ("512", "1230"):
        chosen = buckets[bucket]["tempo_minus_lmcache"]
        gates[f"bucket_{bucket}_remote_noninferior"] = (
            chosen["wins"] >= 8 and chosen["median_delta_ms"] <= 10.0)
    gates["bucket_1230_remote_beats_local"] = (
        buckets["1230"]["tempo_minus_local"]["wins"] >= 9
        and buckets["1230"]["tempo_minus_local"]["median_delta_ms"] < 0.0)
    gates["bucket_2048_local_noninferior"] = (
        buckets["2048"]["tempo_minus_local"]["wins"] >= 8
        and buckets["2048"]["tempo_minus_local"]["median_delta_ms"] <= 10.0)
    gates["bucket_2048_local_beats_lmcache"] = (
        buckets["2048"]["tempo_minus_lmcache"]["wins"] >= 9
        and buckets["2048"]["tempo_minus_lmcache"]["median_delta_ms"] < 0.0)
    gates["routes_16_local_32_remote"] = tempo["routes"] == {
        "decoder_local_recompute_or_cache": 16, "remote_prefill_live_kv": 32}
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    result = {
        "schema": "tempo-pd-request-interleaved-analysis-100",
        "contract": contract, "fixed_local": public(local), "tempo": public(tempo),
        "lmcache_remote": public(remote), "prompt_buckets": buckets,
        "gates": gates, "passes": all(gates.values()),
        "verdict": ("promising_request_interleaved_output16_split"
                    if all(gates.values()) else "reject_request_interleaved_output16_split"),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "buckets": buckets,
                      "failed": [k for k, v in gates.items() if not v]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
