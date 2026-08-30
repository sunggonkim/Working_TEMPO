#!/usr/bin/env python3
"""Corrected N21 aggregate without M20 completion-barrier fields."""
from __future__ import annotations
import statistics
from typing import Any
from eval.sota_4node import run_vllm_lmcache_tp16_headstart25_n21_entry as n21

def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    result = n21.c9._aggregate(records, trace, args)
    ordered = sorted(records, key=lambda row: int(row["rank"]))
    candidates = [row for row in result["blocks"] if row["mode"] == n21.CANDIDATE_MODE]
    by_mode = {mode: [row for row in result["blocks"] if row["mode"] == mode]
               for mode in (n21.old.FG, n21.old.LMCACHE, n21.CANDIDATE_MODE)}
    for block in candidates:
        index = int(block["block_index"])
        raw = [row["blocks"][index] for row in ordered]
        outer_finish_ms = float(raw[0]["client_finished_from_origin_ns"]) / 1e6
        block["admission_to_response_ms"] = outer_finish_ms
        block["service_makespan_ms"] = max(outer_finish_ms, block["background_completion_from_start_ms"])
        block["headstart_elapsed_ms"] = max(float(row["headstart_elapsed_ns"]) / 1e6 for row in raw)
        block["transfer_completed_before_decode_sources"] = sum(
            bool(row["transfer_completed_before_decode"]) for row in raw[:8])
    result["mode_metrics"][n21.CANDIDATE_MODE]["service_makespan_p50_ms"] = statistics.median(
        row["service_makespan_ms"] for row in candidates)
    deltas = []
    for paired in result["paired"]:
        prompt = int(paired["prompt_index"])
        candidate = next(row for row in candidates if row["prompt_index"] == prompt)
        baseline = next(row for row in by_mode[n21.old.LMCACHE] if row["prompt_index"] == prompt)
        delta = candidate["service_makespan_ms"] - baseline["service_makespan_ms"]
        paired["tempo_minus_lmcache_service_makespan_ms"] = delta
        paired["tempo_admission_to_response_ms"] = candidate["admission_to_response_ms"]
        deltas.append(delta)
    cand_e2e = result["mode_metrics"][n21.CANDIDATE_MODE]["e2e_p50_ms"]
    fg_e2e = result["mode_metrics"][n21.old.FG]["e2e_p50_ms"]
    cand_tpot = result["mode_metrics"][n21.CANDIDATE_MODE]["tpot_p99_max_ms"]
    lm_tpot = result["mode_metrics"][n21.old.LMCACHE]["tpot_p99_max_ms"]
    gates = {"correctness_output_trace": bool(result["overall_correctness_met"]),
             "headstart_at_least_25ms": all(b["headstart_elapsed_ms"] >= 25 for b in candidates),
             "all_candidate_post_foreground_drain_zero": all(b["post_foreground_drain_ms"] == 0 for b in candidates),
             "paired_service_median_le_minus_5ms": statistics.median(deltas) <= -5,
             "paired_service_meaningful_wins_ge_2": sum(d <= -5 for d in deltas) >= 2,
             "candidate_request_e2e_p50_le_1_03x_fg": cand_e2e <= 1.03 * fg_e2e,
             "candidate_tpot_p99_le_1_10x_lmcache": cand_tpot <= 1.10 * lm_tpot}
    result["schema_version"] = n21.RESULT_SCHEMA
    result["contract_id"] = n21.CONTRACT_ID
    result["config"].update(candidate_mode=n21.CANDIDATE_MODE,
        predecode_headstart_ms=n21.HEADSTART_MS,
        headstart_included_in_admission_latency=True,
        completion_wait_before_decode=False)
    result["candidate_gates"] = gates
    result["screen_outcome"] = ("invalid_correctness_output_or_trace" if not result["overall_correctness_met"]
        else "headstart25_candidate_pass" if all(gates.values()) else "headstart25_candidate_revise")
    return result

def main() -> None:
    n21._aggregate = _aggregate
    n21.main()

if __name__ == "__main__": main()
