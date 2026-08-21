#!/usr/bin/env python3
"""Analyze canonical four-arm Elastic-PD validation evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from eval.sota_4node import analyze_tempo_pd_elastic_balanced_v445 as legacy


ARMS = ("local", "remote", "predictor", "tempo")
ROUTES = {"decoder_local_chunked_prefill", "official_lmcache_remote_prefill"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_finite_real(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_vllm_load_snapshot(decision: dict) -> bool:
    if not isinstance(decision, dict):
        return False
    engine_indices = decision.get("vllm_load_engine_indices")
    fetch_ms = decision.get("vllm_load_fetch_ms")
    kv_usage = decision.get("vllm_kv_cache_usage_perc")
    common = (
        decision.get("vllm_load_snapshot_schema")
        == "tempo-vllm-load-snapshot-v1"
        and isinstance(decision.get("vllm_load_model_name"), str)
        and bool(decision["vllm_load_model_name"])
        and type(decision.get("vllm_load_sampled_ns")) is int
        and decision["vllm_load_sampled_ns"] > 0
        and _is_finite_real(fetch_ms)
        and fetch_ms >= 0
    )
    if not common:
        return False
    if decision.get("vllm_load_decision_mode") == "disabled":
        return (
            decision.get("vllm_load_snapshot_source")
            == "explicitly_disabled_no_request_rpc"
            and decision.get("vllm_load_endpoint") is None
            and engine_indices == []
            and fetch_ms == 0
            and decision.get("vllm_num_requests_running") is None
            and decision.get("vllm_num_requests_waiting") is None
            and kv_usage is None
        )
    return (
        decision.get("vllm_load_snapshot_source")
        == "local_decoder_prometheus_request_start"
        and decision.get("vllm_load_decision_mode") == "observe_only"
        and decision.get("vllm_load_endpoint") == "/metrics"
        and isinstance(engine_indices, list)
        and bool(engine_indices)
        and all(type(value) is int and value >= 0 for value in engine_indices)
        and engine_indices == sorted(set(engine_indices))
        and type(decision.get("vllm_num_requests_running")) is int
        and decision["vllm_num_requests_running"] >= 0
        and type(decision.get("vllm_num_requests_waiting")) is int
        and decision["vllm_num_requests_waiting"] >= 0
        and _is_finite_real(kv_usage)
        and 0 <= kv_usage <= 1
    )


def _explicit_cold_artifact_contract_valid(contract: dict) -> bool:
    return (
        isinstance(contract, dict)
        and contract.get("phase") == "measured"
        and contract.get("cache_keys_disjoint_across_blocks") is True
        and contract.get(
            "cache_keys_stable_across_warm_and_measured") is False
        and contract.get("cache_key_isolation_scope")
        == "phase_arm_replicate_and_item"
        and contract.get("warm_preparation")
        == "unmeasured_only_no_measured_key_reuse"
        and contract.get("measured_cache_residency")
        == "cold_disjoint_prompt_keys"
    )


def _explicit_cold_completion_valid(decision: dict) -> bool:
    if (
        not isinstance(decision, dict)
        or decision.get("benchmark_cold_measured") is not True
        or decision.get("decision_cache_residency") != "unknown"
    ):
        return False
    route = decision.get("route")
    if route == "decoder_local_chunked_prefill":
        return (
            decision.get("cache_residency") == "confirmed_miss"
            and decision.get("completion_cache_residency")
            == "confirmed_miss"
            and decision.get("lmcache_source_cached_tokens") is None
            and decision.get("lmcache_source_full_hit_observed") is None
        )
    if route == "official_lmcache_remote_prefill":
        return (
            decision.get("cache_residency") == "prefill_only"
            and decision.get("completion_cache_residency")
            == "prefill_only"
            and decision.get("lmcache_source_cached_tokens") == 0
            and decision.get("lmcache_source_full_hit_observed") is False
        )
    return False


def _tempo_pair_affinity_matches_mode(
    decision: dict, *, cold_measured: bool,
) -> bool:
    if not isinstance(decision, dict):
        return False
    owner_indices = decision.get("frontend_pair_affinity_owner_indices")
    evidence_ids = decision.get(
        "frontend_pair_affinity_evidence_request_ids")
    if cold_measured:
        return (
            decision.get("frontend_pair_affinity_policy")
            == "warm-prompt-sha256-owner-set-v2"
            and decision.get("frontend_pair_affinity_required") is False
            and decision.get(
                "frontend_pair_affinity_owner_count_required") == 1
            and decision.get("frontend_pair_affinity_hit") is False
            and decision.get("frontend_pair_affinity_created") is False
            and owner_indices == []
            and decision.get("frontend_pair_affinity_replica_count") == 0
            and evidence_ids == []
            and decision.get(
                "frontend_pair_affinity_registration_source")
            == "reservation_or_unproven"
        )
    return (
        decision.get("frontend_pair_affinity_policy")
        == "warm-prompt-sha256-owner-set-v2"
        and decision.get("frontend_pair_affinity_hit") is True
        and decision.get(
            "frontend_pair_affinity_owner_count_required") == 2
        and owner_indices == [0, 1]
        and decision.get("frontend_pair_affinity_replica_count") == 2
        and decision.get("frontend_pair_index") in owner_indices
        and decision.get("frontend_pair_affinity_owner_index")
        == decision.get("frontend_pair_index")
        and decision.get("frontend_pair_affinity_registration_source")
        == "completed_warm_probe_eof"
        and isinstance(evidence_ids, list)
        and len(evidence_ids) == 2
        and all(
            isinstance(value, str)
            and "-warm-" in value
            and "-warm-seed-" not in value
            for value in evidence_ids
        )
        and sum(
            "-affinity-shadow-p" in value for value in evidence_ids
        ) == 1
    )
def _p99(values: list[float]) -> float:
    _require(values, "p99 requires values")
    return sorted(values)[min(len(values) - 1, int(len(values) * .99))]


def _metrics(row: dict) -> dict[str, float]:
    arrivals = row["token_arrival_offsets_ns"]
    dispatch = row["dispatch_offset_ns"]
    intervals = [(b - a) / 1_000_000 for a, b in zip(arrivals, arrivals[1:])]
    return {
        "ttft_ms": (arrivals[0] - dispatch) / 1_000_000,
        "e2e_ms": (row["stream_end_offset_ns"] - dispatch) / 1_000_000,
        "tpot_ms": statistics.median(intervals) if intervals else 0.0,
    }


def _group(prompt_tokens: int) -> str:
    return "prompt_512" if prompt_tokens <= 512 else (
        "prompt_2k" if prompt_tokens <= 2048 else "prompt_4k"
    )


def _prompt_tokens(request: dict, decision: dict) -> int:
    value = request.get("usage", {}).get("prompt_tokens")
    _require(type(value) is int and value > 0, "prompt token count missing")
    proofs = request.get("output_token_proofs", [])
    if (
        decision.get("route") == "official_lmcache_remote_prefill"
        and isinstance(proofs, list)
        and proofs.count("official_lmcache_proxy_single_prefill_token") == 1
    ):
        value -= 1
    _require(value > 0, "normalized prompt token count invalid")
    return value


def _load(stage_root: Path) -> tuple[dict, dict[tuple[str, int, int], dict]]:
    public = json.loads((stage_root / "raw.json").read_text())
    orchestration = public.get("elastic_balanced_orchestration")
    _require(isinstance(orchestration, dict), "elastic orchestration missing")
    artifacts = orchestration.get("artifacts")
    _require(isinstance(artifacts, dict), "artifact map missing")
    rows = {}
    for artifact_key, raw_path in sorted(artifacts.items()):
        artifact = json.loads(Path(raw_path).read_text())
        contract = artifact.get("elastic_balanced_contract", {})
        arm = contract.get("arm")
        replicate = contract.get("replicate")
        _require(arm in ARMS and type(replicate) is int, f"{artifact_key}: arm/replicate")
        requests = artifact.get("requests")
        decisions = artifact.get("router_decisions")
        _require(isinstance(requests, list) and isinstance(decisions, list),
                 f"{artifact_key}: request/decision rows missing")
        decision_map = {row.get("request_id"): row for row in decisions}
        _require(len(decision_map) == len(decisions) == len(requests),
                 f"{artifact_key}: request/decision identity mismatch")
        for item, request in enumerate(requests):
            decision = decision_map.get(request.get("request_id"))
            _require(decision is not None, f"{artifact_key}: decision missing")
            prompt_tokens = _prompt_tokens(request, decision)
            key = (arm, replicate, item)
            _require(key not in rows, f"duplicate paired row: {key}")
            rows[key] = {
                **_metrics(request),
                "group": _group(prompt_tokens),
                "prompt_tokens": prompt_tokens,
                "output_tokens": request.get("requested_max_tokens"),
                "output_sha256": request.get("output_text_sha256"),
                "route": decision.get("route"),
                "request": request,
                "decision": decision,
                "artifact_contract": contract,
            }
    return public, rows


def _summary(values: list[dict]) -> dict:
    return {
        metric: {
            "median": statistics.median(row[metric] for row in values),
            "p99_nearest_rank": _p99([row[metric] for row in values]),
            "max": max(row[metric] for row in values),
        }
        for metric in ("ttft_ms", "e2e_ms", "tpot_ms")
    }


def analyze(stage_root: Path) -> dict:
    public, rows = _load(stage_root)
    cache_mode_markers = {
        row["decision"].get("benchmark_cold_measured")
        for row in rows.values()
    }
    _require(
        cache_mode_markers in ({True}, {False}, {None}),
        "mixed or invalid benchmark cold-measured evidence",
    )
    cold_measured = cache_mode_markers == {True}
    orchestration = public.get("elastic_balanced_orchestration")
    _require(isinstance(orchestration, dict), "elastic orchestration missing")
    _require(
        orchestration.get("cache_keys_stable_across_phases")
        is (not cold_measured),
        "orchestration cache mode disagrees with request evidence",
    )
    warm_paths = sorted(
        (stage_root / "elastic_balanced_warm").glob("*.raw.json"))
    _require(len(warm_paths) == 4, "exactly four warm artifacts are required")
    warm_prompt_hashes = set()
    for warm_path in warm_paths:
        warm_artifact = json.loads(warm_path.read_text())
        warm_requests = warm_artifact.get("requests")
        _require(
            isinstance(warm_requests, list) and warm_requests,
            f"{warm_path}: warm request rows missing",
        )
        for warm_request in warm_requests:
            prompt_hash = warm_request.get("prompt_sha256")
            _require(
                isinstance(prompt_hash, str) and len(prompt_hash) == 64,
                f"{warm_path}: warm prompt hash missing",
            )
            warm_prompt_hashes.add(prompt_hash)
    measured_prompt_hashes = [
        row["request"].get("prompt_sha256") for row in rows.values()
    ]
    _require(
        all(
            isinstance(value, str) and len(value) == 64
            for value in measured_prompt_hashes
        ),
        "measured prompt hash missing",
    )
    by_arm = defaultdict(list)
    by_group = defaultdict(lambda: defaultdict(list))
    correctness = {
        "all_streams_valid": True,
        "all_routes_terminal": True,
        "all_outputs_present": True,
        "no_transfer_or_timeout_error": True,
        "no_terminal_queue": True,
        "no_hidden_fallback_reason": True,
        "all_frontend_pair_reservations_released": True,
        "all_decisions_match_cache_mode": True,
        "all_artifact_cache_contracts_match_mode": True,
        "tempo_pair_affinity_matches_cache_mode": True,
        "cold_measured_prompts_disjoint_from_warm": (
            not cold_measured
            or not warm_prompt_hashes.intersection(measured_prompt_hashes)
        ),
        "cold_measured_prompts_unique_across_blocks": (
            not cold_measured
            or len(set(measured_prompt_hashes)) == len(measured_prompt_hashes)),
        "all_request_start_vllm_load_snapshots_valid": True,
    }
    for (arm, _replicate, _item), row in rows.items():
        by_arm[arm].append(row)
        by_group[row["group"]][arm].append(row)
        decision = row["decision"]
        request = row["request"]
        correctness["all_streams_valid"] &= request.get("valid") is True
        correctness["all_routes_terminal"] &= decision.get("phase") == "complete"
        correctness["all_outputs_present"] &= isinstance(row["output_sha256"], str)
        correctness["no_transfer_or_timeout_error"] &= (
            request.get("error") is None and decision.get("error") is None
        )
        correctness["no_terminal_queue"] &= decision.get("route") in ROUTES
        correctness["no_hidden_fallback_reason"] &= "fallback" not in str(
            decision.get("reason", "")
        ).lower()
        correctness["all_frontend_pair_reservations_released"] &= (
            decision.get("frontend_pair_released") is True)
        correctness[
            "all_request_start_vllm_load_snapshots_valid"
        ] &= _valid_vllm_load_snapshot(decision)
        if cold_measured:
            correctness["all_decisions_match_cache_mode"] &= (
                _explicit_cold_completion_valid(decision))
            correctness["all_artifact_cache_contracts_match_mode"] &= (
                _explicit_cold_artifact_contract_valid(
                    row["artifact_contract"]))
        else:
            correctness["all_decisions_match_cache_mode"] &= (
                decision.get("benchmark_cold_measured") in (False, None))
            correctness["all_artifact_cache_contracts_match_mode"] &= (
                row["artifact_contract"].get(
                    "cache_keys_stable_across_warm_and_measured") is True
                and row["artifact_contract"].get(
                    "measured_cache_residency") == "prefill_only_warm"
            )
        if arm == "tempo":
            correctness["tempo_pair_affinity_matches_cache_mode"] &= (
                _tempo_pair_affinity_matches_mode(
                    decision, cold_measured=cold_measured))
    _require(all(by_arm[arm] for arm in ARMS), "all four arms are required")

    pair_rows = []
    for replicate, item in sorted({(key[1], key[2]) for key in rows}):
        values = {arm: rows.get((arm, replicate, item)) for arm in ARMS}
        _require(all(values.values()), f"incomplete paired item: {(replicate, item)}")
        hashes = {values[arm]["output_sha256"] for arm in ARMS}
        geometry = {
            (values[arm]["prompt_tokens"], values[arm]["output_tokens"])
            for arm in ARMS
        }
        pair_rows.append({
            "replicate": replicate,
            "item": item,
            "group": values["tempo"]["group"],
            "output_exact": len(hashes) == 1,
            "geometry_exact": len(geometry) == 1,
            "tempo_minus_local_e2e_ms": values["tempo"]["e2e_ms"] - values["local"]["e2e_ms"],
            "tempo_minus_remote_e2e_ms": values["tempo"]["e2e_ms"] - values["remote"]["e2e_ms"],
            "tempo_minus_predictor_e2e_ms": values["tempo"]["e2e_ms"] - values["predictor"]["e2e_ms"],
            "tempo_minus_best_fixed_e2e_ms": values["tempo"]["e2e_ms"] - min(
                values["local"]["e2e_ms"], values["remote"]["e2e_ms"]
            ),
            "tempo_route": values["tempo"]["route"],
            "tempo_vllm_load_decision_mode": values["tempo"][
                "decision"].get("vllm_load_decision_mode"),
            "tempo_vllm_num_requests_running": values["tempo"][
                "decision"].get("vllm_num_requests_running"),
            "tempo_vllm_num_requests_waiting": values["tempo"][
                "decision"].get("vllm_num_requests_waiting"),
            "tempo_vllm_kv_cache_usage_perc": values["tempo"][
                "decision"].get("vllm_kv_cache_usage_perc"),
            "tempo_vllm_load_fetch_ms": values["tempo"][
                "decision"].get("vllm_load_fetch_ms"),
        })

    aggregate = {arm: _summary(by_arm[arm]) for arm in ARMS}
    best_fixed = min(("local", "remote"), key=lambda arm: aggregate[arm]["e2e_ms"]["median"])
    best_fixed_median = aggregate[best_fixed]["e2e_ms"]["median"]
    tempo_median = aggregate["tempo"]["e2e_ms"]["median"]
    predictor_median = aggregate["predictor"]["e2e_ms"]["median"]
    group_gates = {}
    group_summaries = {}
    for group, values in sorted(by_group.items()):
        _require(all(len(values[arm]) >= 3 for arm in ARMS),
                 f"workload group lacks three paired samples: {group}")
        summaries = {arm: _summary(values[arm]) for arm in ARMS}
        fixed = min(("local", "remote"), key=lambda arm: summaries[arm]["e2e_ms"]["median"])
        group_rows = [row for row in pair_rows if row["group"] == group]
        deltas = [row["tempo_minus_best_fixed_e2e_ms"] for row in group_rows]
        group_summaries[group] = {"best_fixed_arm": fixed, "arm_summary": summaries,
                                  "paired_samples": len(group_rows)}
        group_gates[group] = {
            "paired_win_fraction_ge_60pct": sum(delta < 0 for delta in deltas) / len(deltas) >= .60,
            "e2e_p99_not_over_105pct_best_fixed": summaries["tempo"]["e2e_ms"]["p99_nearest_rank"] <= 1.05 * summaries[fixed]["e2e_ms"]["p99_nearest_rank"],
            "tpot_p99_not_over_105pct_best_fixed": summaries["tempo"]["tpot_ms"]["p99_nearest_rank"] <= 1.05 * summaries[fixed]["tpot_ms"]["p99_nearest_rank"],
        }

    def route_pairs(route: str, counterfactual: str):
        result = []
        for replicate, item in sorted({(key[1], key[2]) for key in rows}):
            tempo = rows[("tempo", replicate, item)]
            if tempo["route"] == route:
                result.append((tempo["e2e_ms"], rows[(counterfactual, replicate, item)]["e2e_ms"]))
        return result

    def branch_improvement(pairs):
        return (
            statistics.median(
                (other - selected) / other
                for selected, other in pairs
            )
            if pairs else None
        )

    remote_pairs = route_pairs(
        "official_lmcache_remote_prefill", "local")
    local_pairs = route_pairs(
        "decoder_local_chunked_prefill", "remote")
    remote_branch_improvement = branch_improvement(remote_pairs)
    local_branch_improvement = branch_improvement(local_pairs)

    durations = {}
    for arm in ARMS:
        values = by_arm[arm]
        start = min(row["request"]["dispatch_offset_ns"] for row in values)
        end = max(row["request"]["stream_end_offset_ns"] for row in values)
        durations[arm] = len(values) / max((end - start) / 1_000_000_000, 1e-9)
    best_goodput = max(durations["local"], durations["remote"])
    best_deltas = [row["tempo_minus_best_fixed_e2e_ms"] for row in pair_rows]
    gates = {
        "correctness_and_lifecycle": all(correctness.values()) and all(
            row["output_exact"] and row["geometry_exact"] for row in pair_rows
        ),
        "tempo_e2e_improves_best_fixed_ge_10pct": (best_fixed_median - tempo_median) / best_fixed_median >= .10,
        "tempo_e2e_improves_predictor_ge_5pct": (predictor_median - tempo_median) / predictor_median >= .05,
        "goodput_improves_best_fixed_ge_5pct": (durations["tempo"] - best_goodput) / best_goodput >= .05,
        "paired_win_fraction_ge_75pct": sum(delta < 0 for delta in best_deltas) / len(best_deltas) >= .75,
        "worst_paired_regression_le_100ms": max(best_deltas) <= 100.0,
        "remote_selected_and_wins_local_counterfactual_ge_5pct": (
            remote_branch_improvement is not None
            and remote_branch_improvement >= .05),
        "local_selected_and_wins_remote_counterfactual_ge_5pct": (
            local_branch_improvement is not None
            and local_branch_improvement >= .05),
        "all_workload_groups_pass": all(all(value.values()) for value in group_gates.values()),
    }
    legacy_result = legacy.analyze(stage_root)
    return {
        "schema": "tempo-elastic-pd-analysis-canonical",
        "cache_contract_mode": (
            "cold_disjoint_prompt_keys"
            if cold_measured else "prefill_only_warm"),
        "measurement_valid": gates["correctness_and_lifecycle"],
        "arm_summary": aggregate,
        "best_fixed_arm": best_fixed,
        "paired_rows": pair_rows,
        "group_summaries": group_summaries,
        "group_gates": group_gates,
        "request_goodput": durations,
        "route_counterfactual": {
            "remote_selected_count": len(remote_pairs),
            "remote_selected_vs_local_median_improvement_pct": (
                remote_branch_improvement * 100
                if remote_branch_improvement is not None else None),
            "local_selected_count": len(local_pairs),
            "local_selected_vs_remote_median_improvement_pct": (
                local_branch_improvement * 100
                if local_branch_improvement is not None else None),
        },
        "route_counts": {
            f"{arm}:{route}": sum(row["route"] == route for key, row in rows.items() if key[0] == arm)
            for arm in ARMS for route in sorted(ROUTES)
        },
        "improvement_pct": {
            "tempo_vs_best_fixed_e2e": (best_fixed_median - tempo_median) / best_fixed_median * 100,
            "tempo_vs_predictor_e2e": (predictor_median - tempo_median) / predictor_median * 100,
            "tempo_vs_best_fixed_goodput": (durations["tempo"] - best_goodput) / best_goodput * 100,
        },
        "correctness": correctness,
        "candidate_gates": gates,
        "candidate_passes": all(gates.values()),
        "verdict": (
            "finalize_tempo_elastic_pd" if all(gates.values())
            else "continue_tempo_elastic_pd_discovery"),
        "claim_boundary": "same Perlmutter A100 four-node actual vLLM P/D topology, identical GPU budget, requests, cache namespace, and official LMCacheConnectorV1 data plane; not a transport-speed, Mooncake, universal-SOTA, or production-readiness claim",
        "legacy_screen_reference": legacy_result,
        "source_orchestration": public.get("elastic_balanced_orchestration"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite")
    result = analyze(args.stage_root.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"], "candidate_passes": result["candidate_passes"], "candidate_gates": result["candidate_gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
