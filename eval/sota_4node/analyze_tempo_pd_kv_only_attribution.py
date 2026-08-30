#!/usr/bin/env python3
"""Summarize P-only KV-path attribution without naming a switch bottleneck."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from eval.sota_4node import analyze_tempo_pd_endpoint_characterization as endpoint
from eval.sota_4node import run_tempo_pd_contention_fixed_client as evidence_client


SCHEMA = "tempo-pd-kv-only-characterization-v3"
CLIENT_SCHEMA = "tempo-pd-kv-only-attribution-client-v2"
BLOCK_SCHEMA = "tempo-pd-kv-only-attribution-block-v2"
_CLIENT_SCHEMAS = frozenset({
    "tempo-pd-kv-only-attribution-client-v1",
    CLIENT_SCHEMA,
})
_BLOCK_SCHEMAS = frozenset({
    "tempo-pd-kv-only-attribution-block-v1",
    BLOCK_SCHEMA,
})
_FRACTION_SIGNALS = (
    "rx_pause_fraction_max",
    "tx_pause_fraction_max",
    "receive_overflow_fraction_max",
    "ecn_fraction_max",
)
_FAULT_SIGNALS = ("resource_nacks", "retries", "timeouts")
_HOST_SIGNALS = (
    "host_posted_cycles_per_packet_max",
    "host_nonposted_cycles_per_packet_max",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _latencies(
    rows: list[dict[str, object]], *, allow_empty: bool = False,
) -> dict[str, object]:
    if not rows:
        _require(allow_empty, "latency group is empty")
        return {
            "count": 0,
            "e2e_median_ms": None,
            "e2e_p99_ms": None,
            "ttft_median_ms": None,
            "ttft_p99_ms": None,
        }
    _require(all(row.get("valid") is True for row in rows),
             "latency group contains an invalid request")
    e2e = [
        (row["stream_end_offset_ns"] - row["dispatch_offset_ns"]) / 1_000_000.0
        for row in rows
    ]
    ttft = [
        (row["token_arrival_offsets_ns"][0] - row["dispatch_offset_ns"])
        / 1_000_000.0
        for row in rows
    ]
    return {
        "count": len(rows),
        "e2e_median_ms": statistics.median(e2e),
        "e2e_p99_ms": endpoint._percentile(e2e, 0.99),
        "ttft_median_ms": statistics.median(ttft),
        "ttft_p99_ms": endpoint._percentile(ttft, 0.99),
    }


def _weighted_endpoint_mean(
    cumulative: dict[str, dict[str, object]],
    *,
    role_suffix: str,
    metric: str,
) -> float | None:
    numerator = 0.0
    denominator = 0
    for endpoint_id, row in cumulative.items():
        if not endpoint_id.endswith(role_suffix):
            continue
        count = int(row["delta"]["vllm:request_success_total"])
        value = row["derived"][metric]
        if count and value is not None:
            numerator += count * float(value)
            denominator += count
    return numerator / denominator if denominator else None


def _role_delta_totals(
    cumulative: dict[str, dict[str, object]], *, role_suffix: str,
) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for endpoint_id, row in cumulative.items():
        if not endpoint_id.endswith(role_suffix):
            continue
        for name, value in row["delta"].items():
            totals[name] = totals.get(name, 0) + value
    return totals


def _cassini_summary(
    stages: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    rows = []
    for stage_name in ("midpoint", "after"):
        for endpoint_id, probe in stages[stage_name].items():
            sample = probe["cassini"]
            rows.append({
                "stage": stage_name,
                "endpoint_id": endpoint_id,
                "valid": sample["valid"],
                "invalid_reason": sample["invalid_reason"],
                "window_ms": sample["window_ms"],
                "signals": sample["signals"],
            })
    valid = [row for row in rows if row["valid"] is True]
    fractions = {
        name: max(
            (float(row["signals"][name]) for row in valid
             if row["signals"].get(name) is not None),
            default=None,
        )
        for name in _FRACTION_SIGNALS
    }
    faults = {
        name: sum(
            int(row["signals"][name]) for row in valid
            if row["signals"].get(name) is not None
        )
        for name in _FAULT_SIGNALS
    }
    hosts = {
        name: max(
            (float(row["signals"][name]) for row in valid
             if row["signals"].get(name) is not None),
            default=None,
        )
        for name in _HOST_SIGNALS
    }
    event_observed = any(value not in (None, 0, 0.0)
                         for value in fractions.values()) or any(faults.values())
    return {
        "samples_total": len(rows),
        "samples_valid": len(valid),
        "invalid_samples": [
            {key: row[key] for key in (
                "stage", "endpoint_id", "invalid_reason", "window_ms")}
            for row in rows if row["valid"] is not True
        ],
        "fraction_max": fractions,
        "fault_count": faults,
        "host_cycles_per_packet_max": hosts,
        "fabric_event_observed_in_valid_samples": event_observed,
        "invalid_is_missing_not_zero": True,
    }


def _block(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(raw.get("schema") == "tempo-pd-stream-metrics-raw-1",
             "child raw schema mismatch")
    _require(raw.get("validation", {}).get("all_streams_valid") is True,
             "child stream validation failed")
    contract = raw.get("kv_only_attribution_contract")
    _require(isinstance(contract, dict)
             and contract.get("schema") in _BLOCK_SCHEMAS,
             "P-only block contract is missing")
    _require(contract.get("background_full_source_hits_exact") is True,
             "P-only source-hit contract failed")
    if "background_decision_cache_contract_exact" in contract:
        _require(
            contract.get("background_decision_cache_contract_exact") is True,
            "P-only decision cache contract failed",
        )
    evidence = raw.get("endpoint_evidence")
    evidence_client._validate_endpoint_evidence_bundle(evidence)
    stages = {
        name: endpoint._endpoint_index(evidence[name])
        for name in ("before", "midpoint", "after")
    }
    background = [
        row for row in raw["requests"]
        if (
            "-measured-warm-" in row["request_id"]
            or "-cache-p-only-measured-" in row["request_id"]
        )
    ]
    foreground = [
        row for row in raw["requests"]
        if "-foreground-" in row["request_id"]
    ]
    decoder_hot = [
        row for row in raw["requests"]
        if "-decoderhot-measured-" in row["request_id"]
    ]
    _require(len(background) == contract["request_counts"][
        "p_only_remote_background"], "background request count differs")
    _require(len(foreground) == contract["request_counts"]["foreground"],
             "foreground request count differs")
    _require(
        len(decoder_hot)
        == contract["request_counts"].get("decoder_hot_background", 0),
        "decoder-hot request count differs",
    )
    decisions = raw.get("router_decisions")
    _require(isinstance(decisions, list), "router decisions are missing")
    decision_index = {row.get("request_id"): row for row in decisions}
    _require(len(decision_index) == len(decisions),
             "router decisions contain duplicate request IDs")
    _require(
        set(decision_index) == {row["request_id"] for row in raw["requests"]},
        "router decision request IDs differ",
    )
    _require(all(
        row.get("decoder_prefix_caching") is False
        for row in decisions
    ), "a measured decision enabled decoder prefix caching")
    for row in background:
        decision = decision_index[row["request_id"]]
        _require(
            decision.get("route") == "official_lmcache_remote_prefill"
            and decision.get("lmcache_source_cached_tokens") == 4094
            and decision.get("lmcache_source_full_hit_observed") is True,
            "P-only background decision is not an exact full source hit",
        )
    expected_foreground_route = (
        "decoder_local_chunked_prefill"
        if contract["foreground_arm"] == "local"
        else "official_lmcache_remote_prefill"
    )
    for row in foreground:
        decision = decision_index[row["request_id"]]
        _require(decision.get("route") == expected_foreground_route,
                 "foreground route differs from pinned arm")
        if contract["foreground_arm"] == "remote":
            _require(
                decision.get("lmcache_source_cached_tokens") == 0
                and decision.get("lmcache_source_full_hit_observed") is False,
                "remote foreground is not an exact cold source miss",
            )
        else:
            _require(
                decision.get("lmcache_source_cached_tokens") is None
                and decision.get("lmcache_source_full_hit_observed") is None,
                "local foreground unexpectedly has LMCache source evidence",
            )
    for row in decoder_hot:
        decision = decision_index[row["request_id"]]
        _require(
            decision.get("route") == "decoder_local_chunked_prefill"
            and decision.get("lmcache_source_cached_tokens") is None,
            "decoder-hot request differs from its pinned local route",
        )
    cumulative = endpoint._vllm_cumulative_block(
        stages["before"], stages["after"])
    _require(isinstance(cumulative, dict), "vLLM cumulative evidence is missing")
    prefill_inference = _weighted_endpoint_mean(
        cumulative,
        role_suffix="-prefill",
        metric="mean_inference_time_seconds",
    )
    decoder_inference = _weighted_endpoint_mean(
        cumulative,
        role_suffix="-decoder",
        metric="mean_inference_time_seconds",
    )
    prefill_queue = _weighted_endpoint_mean(
        cumulative,
        role_suffix="-prefill",
        metric="mean_queue_time_seconds",
    )
    decoder_queue = _weighted_endpoint_mean(
        cumulative,
        role_suffix="-decoder",
        metric="mean_queue_time_seconds",
    )
    prefill_kv_computed = _weighted_endpoint_mean(
        cumulative,
        role_suffix="-prefill",
        metric="mean_prefill_kv_computed_tokens",
    )
    prefill_totals = _role_delta_totals(
        cumulative, role_suffix="-prefill")
    prompt_tokens = float(prefill_totals.get(
        "vllm:prompt_tokens_total", 0))
    cached_tokens = float(prefill_totals.get(
        "vllm:prompt_tokens_cached_total", 0))
    foreground_latency = _latencies(foreground)
    client_window_s = float(raw["run"]["client_window_ns"]) / 1_000_000_000.0
    _require(client_window_s > 0.0, "client window is not positive")
    compute_sum_ms = 1000.0 * (
        (prefill_inference or 0.0) + (decoder_inference or 0.0))
    diagnostic_residual = (
        foreground_latency["e2e_median_ms"] - compute_sum_ms
        if contract["foreground_arm"] == "remote" else None
    )
    midpoint_load = {}
    for endpoint_id, probe in stages["midpoint"].items():
        metrics = probe["endpoint"]["metrics"]
        midpoint_load[endpoint_id] = {
            "running_requests": metrics["running_requests"]["value"],
            "waiting_requests": metrics["waiting_requests"]["value"],
            "kv_cache_usage_fraction": metrics[
                "kv_cache_usage_fraction"]["value"],
        }
    return {
        "artifact": str(path.resolve()),
        "background_rate_per_s": contract["background_rate_per_s"],
        "replicate_index": int(contract.get("replicate_index", 0)),
        "block_sequence_index": contract.get("block_sequence_index"),
        "arm_order_policy": contract.get("arm_order_policy", "local_remote"),
        "semantic_schedule_sha256": contract["semantic_schedule_sha256"],
        "foreground_arm": contract["foreground_arm"],
        "contention_state": (
            "c3_both_hot"
            if background and decoder_hot else
            "c2_p_only_remote_hot"
            if background else
            "c1_decoder_hot"
            if decoder_hot else
            "c0_cool"
        ),
        "foreground": foreground_latency,
        "background": _latencies(background, allow_empty=True),
        "decoder_hot_background": _latencies(
            decoder_hot, allow_empty=True),
        "all_requests_valid": contract.get("all_requests_valid") is True,
        "route_and_cache_contract": {
            "decision_count": len(decisions),
            "pinned_routes_exact": True,
            "p_only_full_source_hits_exact": True,
            "remote_foreground_cold_misses_exact": True,
            "decoder_prefix_caching_disabled_exact": True,
        },
        "completion_rate": {
            "client_window_s": client_window_s,
            "all_requests_per_s": len(raw["requests"]) / client_window_s,
            "background_requests_per_s": len(background) / client_window_s,
            "foreground_requests_per_s": len(foreground) / client_window_s,
            "decoder_hot_requests_per_s": len(decoder_hot) / client_window_s,
        },
        "midpoint_load": midpoint_load,
        "vllm_cumulative": cumulative,
        "endpoint_service_diagnostic": {
            "prefill_mean_inference_ms": (
                1000.0 * prefill_inference if prefill_inference is not None else None),
            "decoder_mean_inference_ms": (
                1000.0 * decoder_inference if decoder_inference is not None else None),
            "prefill_mean_queue_ms": (
                1000.0 * prefill_queue if prefill_queue is not None else None),
            "decoder_mean_queue_ms": (
                1000.0 * decoder_queue if decoder_queue is not None else None),
            "prefill_mean_kv_computed_tokens": prefill_kv_computed,
            "prefill_prompt_tokens_total": int(prompt_tokens),
            "prefill_prompt_tokens_cached_total": int(cached_tokens),
            "prefill_prompt_cached_fraction": (
                cached_tokens / prompt_tokens if prompt_tokens else None),
            "prefill_request_success_total": int(prefill_totals.get(
                "vllm:request_success_total", 0)),
            "endpoint_inference_sum_ms": compute_sum_ms,
            "remote_client_median_minus_endpoint_inference_sum_ms": (
                diagnostic_residual),
            "residual_is_diagnostic_not_per_request_decomposition": True,
        },
        "cassini": _cassini_summary(stages),
    }


def analyze(input_path: Path) -> dict[str, object]:
    parent = json.loads(input_path.read_text(encoding="utf-8"))
    _require(parent.get("schema") in _CLIENT_SCHEMAS,
             "P-only attribution parent schema mismatch")
    _require(parent.get("performance_claim_allowed") is False,
             "attribution input incorrectly permits a performance claim")
    artifacts = parent.get("artifacts")
    rates = parent.get("rates_per_s")
    repetitions = int(parent.get("repetitions_per_rate", 1))
    _require(1 <= repetitions <= 4,
             "attribution repetition count is invalid")
    _require(isinstance(rates, list) and rates,
             "attribution rate ladder is missing")
    _require(
        isinstance(artifacts, dict)
        and len(artifacts) == 2 * len(rates) * repetitions,
             "attribution artifact map is incomplete")
    blocks = [_block(Path(path)) for path in artifacts.values()]
    indexed = {
        (
            block["background_rate_per_s"],
            block["replicate_index"],
            block["foreground_arm"],
        ): block
        for block in blocks
    }
    _require(len(indexed) == len(blocks), "attribution blocks are duplicated")
    paired_replicates = []
    baseline_remote = statistics.median([
        indexed[(rates[0], replicate, "remote")]["foreground"][
            "e2e_median_ms"]
        for replicate in range(repetitions)
    ])
    first_double = None
    first_drain = None
    for rate in rates:
        for replicate in range(repetitions):
            local = indexed[(rate, replicate, "local")]
            remote = indexed[(rate, replicate, "remote")]
            _require(
                local["semantic_schedule_sha256"]
                == remote["semantic_schedule_sha256"],
                "local/remote semantic schedule hashes differ",
            )
            local_median = local["foreground"]["e2e_median_ms"]
            remote_median = remote["foreground"]["e2e_median_ms"]
            winner = "local" if local_median < remote_median else "remote"
            sequence = (
                "local_remote"
                if local["block_sequence_index"] is None
                or local["block_sequence_index"]
                < remote["block_sequence_index"]
                else "remote_local"
            )
            paired_replicates.append({
                "background_rate_per_s": rate,
                "replicate_index": replicate,
                "measured_arm_order": sequence,
                "contention_state": remote["contention_state"],
                "winner": winner,
                "local_foreground_median_ms": local_median,
                "remote_foreground_median_ms": remote_median,
                "local_gain_over_remote": (
                    remote_median - local_median) / remote_median,
                "remote_gain_over_local": (
                    local_median - remote_median) / local_median,
                "winner_margin_over_loser": (
                    abs(remote_median - local_median)
                    / max(remote_median, local_median)),
                "remote_inflation_vs_low_rate": (
                    remote_median / baseline_remote),
                "local_decoder_hot_median_ms": local[
                    "decoder_hot_background"]["e2e_median_ms"],
                "remote_decoder_hot_median_ms": remote[
                    "decoder_hot_background"]["e2e_median_ms"],
                "remote_diagnostic_residual_ms": remote[
                    "endpoint_service_diagnostic"][
                        "remote_client_median_minus_endpoint_inference_sum_ms"],
                "remote_background_completion_rate_per_s": remote[
                    "completion_rate"]["background_requests_per_s"],
                "remote_client_window_s": remote[
                    "completion_rate"]["client_window_s"],
                "local_decoder_hot_completion_rate_per_s": local[
                    "completion_rate"]["decoder_hot_requests_per_s"],
                "remote_decoder_hot_completion_rate_per_s": remote[
                    "completion_rate"]["decoder_hot_requests_per_s"],
            })

    paired = []
    baseline_winners = {
        row["winner"] for row in paired_replicates
        if row["background_rate_per_s"] == rates[0]
    }
    for rate in rates:
        rows = [
            row for row in paired_replicates
            if row["background_rate_per_s"] == rate
        ]
        local_median = statistics.median(
            row["local_foreground_median_ms"] for row in rows)
        remote_median = statistics.median(
            row["remote_foreground_median_ms"] for row in rows)
        inflation = remote_median / baseline_remote
        if first_double is None and inflation >= 2.0:
            first_double = rate
        if (
            first_drain is None
            and statistics.median(
                row["remote_client_window_s"] for row in rows)
            >= 1.10 * parent["phase_duration_ms"] / 1000.0
        ):
            first_drain = rate
        winner_counts = {
            arm: sum(row["winner"] == arm for row in rows)
            for arm in ("local", "remote")
        }
        aggregate_winner = (
            "local" if local_median < remote_median else "remote")
        paired.append({
            "background_rate_per_s": rate,
            "replicate_count": len(rows),
            "contention_state": rows[0]["contention_state"],
            "aggregate_winner": aggregate_winner,
            "winner_counts": winner_counts,
            "local_foreground_median_ms": local_median,
            "remote_foreground_median_ms": remote_median,
            "local_gain_over_remote": (remote_median - local_median) / remote_median,
            "remote_inflation_vs_low_rate": inflation,
            "remote_diagnostic_residual_ms": statistics.median(
                row["remote_diagnostic_residual_ms"] for row in rows),
            "remote_background_completion_rate_per_s": statistics.median(
                row["remote_background_completion_rate_per_s"] for row in rows),
            "remote_client_window_s": statistics.median(
                row["remote_client_window_s"] for row in rows),
            "local_decoder_hot_completion_rate_per_s": statistics.median(
                row["local_decoder_hot_completion_rate_per_s"] for row in rows),
            "remote_decoder_hot_completion_rate_per_s": statistics.median(
                row["remote_decoder_hot_completion_rate_per_s"] for row in rows),
        })
    p_only_proof_blocks = [
        block for block in blocks
        if block["foreground_arm"] == "local"
        and block["background"]["count"] > 0
    ]
    p_only_long_prefill_removed = bool(p_only_proof_blocks) and all(
        block["endpoint_service_diagnostic"][
            "prefill_mean_kv_computed_tokens"] is not None
        and block["endpoint_service_diagnostic"][
            "prefill_mean_kv_computed_tokens"] <= 1.0
        and block["endpoint_service_diagnostic"][
            "prefill_prompt_cached_fraction"] is not None
        and block["endpoint_service_diagnostic"][
            "prefill_prompt_cached_fraction"] >= 0.999
        for block in p_only_proof_blocks
    )
    return {
        "schema": SCHEMA,
        "source": str(input_path.resolve()),
        "preseed": parent["preseed"],
        "workload_mode": parent.get("workload_mode"),
        "decoder_hot_rate_per_s": parent.get("decoder_hot_rate_per_s", 0.0),
        "repetitions_per_rate": repetitions,
        "arm_order_policy": parent.get("arm_order_policy", "local_remote"),
        "blocks": blocks,
        "paired_replicate_summary": paired_replicates,
        "paired_rate_summary": paired,
        "winner_flip_rates_from_lowest_rate": [
            row["background_rate_per_s"] for row in paired
            if len(baseline_winners) == 1
            and row["aggregate_winner"] not in baseline_winners
        ],
        "first_rate_with_2x_remote_foreground_median": first_double,
        "first_rate_with_over_10pct_remote_drain": first_drain,
        "max_observed_remote_background_completion_rate_per_s": max(
            row["remote_background_completion_rate_per_s"]
            for row in paired_replicates),
        "measured_request_count": sum(
            block["foreground"]["count"]
            + block["background"]["count"]
            + block["decoder_hot_background"]["count"]
            for block in blocks
        ),
        "all_measured_requests_valid": all(
            block["all_requests_valid"]
            and block["foreground"]["count"] > 0
            and (
                block["background"]["count"] > 0
                or block["background_rate_per_s"] == 0
            )
            and (
                block["decoder_hot_background"]["count"] > 0
                or float(parent.get("decoder_hot_rate_per_s", 0.0)) == 0.0
            )
            for block in blocks
        ),
        "p_only_source_compute_attribution": {
            "local_foreground_blocks_isolate_prefill_endpoint_to_p_only_tenant": True,
            "long_producer_prefill_removed": p_only_long_prefill_removed,
            "expected_residual_recompute_tokens_per_request": 1,
            "zero_producer_compute_claim_allowed": False,
        },
        "invariants": {
            "endpoint_count": 4,
            "snapshots_per_endpoint_per_block": 3,
            "preseed_outside_measurement_window": True,
            "background_full_source_hits_exact": True,
            "pinned_routes_exact": True,
            "remote_foreground_cold_misses_exact": True,
            "paired_semantic_schedule_hashes_equal": True,
            "decoder_prefix_caching": False,
            "synthetic_network_background": False,
            "cross_endpoint_clock_subtraction": False,
        },
        "interpretation_boundary": {
            "component_attribution_only": True,
            "performance_claim_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
            "zero_pause_or_ecn_proves_uncongested_fabric": False,
            "invalid_cassini_sample_is_missing_not_zero": True,
        },
    }


def main() -> int:
    args = _parse()
    _require(args.input.is_file(), "input artifact is missing")
    _require(not args.output.exists(), "refusing to overwrite output")
    result = analyze(args.input)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": SCHEMA, "output": str(args.output.resolve())},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
