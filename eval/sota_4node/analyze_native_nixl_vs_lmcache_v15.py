#!/usr/bin/env python3
"""Compare native vLLM Nixl pull with official LMCache remote P/D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--lmcache", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parsed = {
        label: base._parse_run(label, _load(path), ttft_slo_ms=3000,
                               tpot_slo_ms=250, e2e_slo_ms=12000)
        for label, path in (
            ("local", args.local), ("lmcache", args.lmcache), ("native", args.native)
        )
    }
    first = parsed["local"]
    correctness = all(
        row["model_config_sha256"] == first["model_config_sha256"]
        and row["workload_sha256"] == first["workload_sha256"]
        and row["_contracts"] == first["_contracts"]
        and row["_outputs"] == first["_outputs"]
        for row in parsed.values()
    )
    native_vs_lmcache = base._paired(parsed["native"], parsed["lmcache"]) if correctness else None
    native_vs_local = base._paired(parsed["native"], parsed["local"]) if correctness else None
    native_perf = parsed["native"]["performance"]
    lmcache_perf = parsed["lmcache"]["performance"]
    count = parsed["native"]["request_count"]
    reasons = parsed["native"]["reasons"]
    gates = {
        "same_model_workload_schedule_outputs": correctness,
        "native_all_requests_remote": parsed["native"]["routes"] == {
            "remote_prefill_live_kv": count
        },
        "native_route_provenance_exact": reasons == {
            "fixed_native_vllm_nixl_remote_candidate": count
        },
        "native_e2e_wins_at_least_two_thirds_vs_lmcache": (
            native_vs_lmcache is not None
            and native_vs_lmcache["e2e_win_count"] >= math.ceil(count * 2 / 3)
        ),
        "native_paired_e2e_median_beats_lmcache_by_5ms": (
            native_vs_lmcache is not None
            and native_vs_lmcache["e2e_delta_median_ms"] <= -5.0
        ),
        "native_request_goodput_beats_lmcache": (
            native_perf["slo_goodput"]["request_goodput_per_s"]
            > lmcache_perf["slo_goodput"]["request_goodput_per_s"]
        ),
        "native_tpot_p99_not_worse_than_lmcache": (
            native_perf["tpot_ms"]["p99"] <= lmcache_perf["tpot_ms"]["p99"]
        ),
    }

    def public(row: dict) -> dict:
        return {key: value for key, value in row.items() if not key.startswith("_")}

    result = {
        "schema": "tempo-native-nixl-vs-lmcache-analysis-15",
        "runs": {key: public(value) for key, value in parsed.items()},
        "native_vs_lmcache": native_vs_lmcache,
        "native_vs_local": native_vs_local,
        "gates": gates,
        "passes_native_backend_gate": all(gates.values()),
        "verdict": "adopt_native_nixl_backend" if all(gates.values()) else "reject_native_nixl_backend",
        "claim_boundary": (
            "Same-allocation actual-vLLM TP4x2-replica P/D backend comparison; "
            "native Nixl is an upstream vLLM backend, not a new TEMPO transport."
        ),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
