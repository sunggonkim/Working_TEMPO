#!/usr/bin/env python3
"""Final exact comparison of evidence-gated TEMPO against official LMCache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_performance_v1 as base


def _load(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local = base._parse_run("fixed_local", _load(
        args.reference / "crossover_local/raw.json"),
        ttft_slo_ms=3000, tpot_slo_ms=250, e2e_slo_ms=12000)
    lmcache = base._parse_run("official_lmcache", _load(
        args.reference / "crossover_remote/raw.json"),
        ttft_slo_ms=3000, tpot_slo_ms=250, e2e_slo_ms=12000)
    tempo = base._parse_run("tempo", _load(
        args.candidate_root / "tempo_credit_admission/raw.json"),
        ttft_slo_ms=3000, tpot_slo_ms=250, e2e_slo_ms=12000)
    exact = (
        local["model_config_sha256"] == lmcache["model_config_sha256"]
        == tempo["model_config_sha256"]
        and local["workload_sha256"] == lmcache["workload_sha256"]
        == tempo["workload_sha256"]
        and local["_contracts"] == lmcache["_contracts"] == tempo["_contracts"]
        and local["_outputs"] == lmcache["_outputs"] == tempo["_outputs"]
    )
    paired = base._paired(tempo, lmcache) if exact else None
    lp = lmcache["performance"]
    tp = tempo["performance"]
    reasons = tempo["reasons"]
    gates = {
        "same_model_workload_schedule_outputs": exact,
        "official_lmcache_all_remote": lmcache["routes"] == {"remote_prefill_live_kv": 9},
        "tempo_all_fail_local_with_exact_evidence_reason": (
            tempo["routes"] == {"decoder_local_recompute_or_cache": 9}
            and reasons == {"fail_local_remote_correctness_or_5ms_gate_unproven": 9}
        ),
        "tempo_all_requests_slo_valid": tp["slo_goodput"]["success_fraction"] == 1.0,
        "tempo_goodput_improves": (
            tp["slo_goodput"]["request_goodput_per_s"]
            > lp["slo_goodput"]["request_goodput_per_s"]
        ),
        "tempo_e2e_p50_improves_ge_5_percent": (
            tp["e2e_ms"]["p50"] <= lp["e2e_ms"]["p50"] * 0.95
        ),
        "tempo_tpot_p99_not_worse": tp["tpot_ms"]["p99"] <= lp["tpot_ms"]["p99"],
        "tempo_paired_wins_at_least_two_thirds": (
            paired is not None and paired["e2e_win_count"] >= 6
        ),
    }
    public = lambda row: {k: v for k, v in row.items() if not k.startswith("_")}
    result = {
        "schema": "tempo-pd-final-evidence-gated-analysis-29",
        "fixed_local_sanity": public(local),
        "official_lmcache_always_remote": public(lmcache),
        "tempo_evidence_gated": public(tempo),
        "paired_tempo_minus_lmcache": paired,
        "gates": gates,
        "passes_final_component_gate": all(gates.values()),
        "verdict": "promising_actual_pd_controller" if all(gates.values()) else "revise_or_stop",
        "claim_boundary": (
            "Actual vLLM-owned-KV 4-node TP4x2-replica P/D screen. TEMPO's measured "
            "admission contribution is avoiding an unprofitable LMCache transfer; it is "
            "not a claim of a new transport or universal SOTA."
        ),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "gates": gates}, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
