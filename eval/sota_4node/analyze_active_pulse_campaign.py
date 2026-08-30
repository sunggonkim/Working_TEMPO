#!/usr/bin/env python3
"""Analyze explicit repeated LMCache active-pulse result files.

Only ``result.json`` paths supplied with ``--run`` and their exact
``rank_0.json`` ... ``rank_7.json`` siblings are opened.  This is a research
screen: the output deliberately contains no promotion verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


WORLD_SIZE = 8
RANKS_PER_NODE = 4
RESULT_SCHEMA = "tempo-lmcache-epoch-2node-2"
RANK_SCHEMA = "tempo-lmcache-epoch-rank-2"
ANALYSIS_SCHEMA = "tempo-active-pulse-campaign-analysis-1"
SLO_MS = 4.0
ZERO_MS = 1.0e-9


class AnalysisError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _number(value: Any, field: str) -> float:
    _require(type(value) in (int, float), f"{field} must be numeric")
    value = float(value)
    _require(math.isfinite(value) and value >= 0.0, f"{field} must be finite and nonnegative")
    return value


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "percentile requires samples")
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))]


def slo_metrics(tails_ms: Sequence[float], threshold_ms: float = SLO_MS) -> dict[str, Any]:
    """Return both hit fraction and goodput using *all* elapsed token time."""

    values = [_number(value, "token tail") for value in tails_ms]
    _require(bool(values) and sum(values) > 0.0, "SLO metrics require positive elapsed time")
    successes = sum(value <= threshold_ms for value in values)
    return {
        "threshold_ms": threshold_ms,
        "successful_tokens": successes,
        "total_tokens": len(values),
        "success_fraction": successes / len(values),
        "time_normalized_goodput_tokens_per_s": successes * 1000.0 / sum(values),
        "all_token_tail_time_ms": sum(values),
    }


def _infer_variant(result: dict[str, Any], tempo_sources: Sequence[dict[str, Any]]) -> str:
    executions = {block.get("execution") for block in tempo_sources}
    executions.discard(None)
    if len(executions) == 1:
        return str(next(iter(executions)))
    semantics = result.get("scheduler_semantics", {})
    if isinstance(semantics, dict) and isinstance(semantics.get("name"), str):
        return semantics["name"]
    return "active_pulse"


def _schedule_id(result: dict[str, Any], tempo_sources: Sequence[dict[str, Any]], variant: str) -> str:
    signatures = {block.get("active_service_plan_signature") for block in tempo_sources}
    signatures.discard(None)
    if len(signatures) == 1:
        return str(next(iter(signatures)))
    config = result["config"]
    signature = config.get("epoch_plan_signature")
    return str(signature) if signature else variant


def analyze_run(result_path: Path, label: str | None = None) -> dict[str, Any]:
    _require(result_path.name == "result.json", "input must be an explicit result.json path")
    result = _load(result_path)
    _require(result.get("schema_version") == RESULT_SCHEMA, "unexpected result schema")
    _require(result.get("world_size") == WORLD_SIZE and result.get("nodes") == 2,
             "result must be a 2-node/8-rank run")
    config = result.get("config")
    _require(isinstance(config, dict), "result.config must be an object")
    tokens = config.get("tokens")
    _require(type(tokens) is int and tokens > 0, "config.tokens must be positive")
    sequence = result.get("block_sequence")
    _require(isinstance(sequence, list) and sequence, "block_sequence must be nonempty")

    ranks = []
    for rank in range(WORLD_SIZE):
        record = _load(result_path.parent / f"rank_{rank}.json")
        _require(record.get("schema_version") == RANK_SCHEMA, f"rank_{rank} schema mismatch")
        _require(record.get("rank") == rank and record.get("world_size") == WORLD_SIZE,
                 f"rank_{rank} identity mismatch")
        _require(record.get("config") == config, f"rank_{rank} config mismatch")
        blocks = record.get("blocks")
        _require(isinstance(blocks, list) and len(blocks) == len(sequence),
                 f"rank_{rank} block count mismatch")
        ranks.append(record)

    result_blocks = result.get("blocks")
    _require(isinstance(result_blocks, list) and len(result_blocks) == len(sequence),
             "result block count mismatch")
    local_bytes = int(config["requests"]) * int(config["kv_bytes"])
    tails_by_mode: dict[str, list[float]] = defaultdict(list)
    block_metrics = []
    exact_bytes_and_correctness = True
    tempo_source_blocks: list[dict[str, Any]] = []

    for index, mode in enumerate(sequence):
        rank_blocks = [rank["blocks"][index] for rank in ranks]
        for rank, block in enumerate(rank_blocks):
            _require(block.get("block_index") == index and block.get("mode") == mode,
                     f"rank_{rank} block {index} identity mismatch")
            latency = block.get("token_latency_ms")
            _require(isinstance(latency, list) and len(latency) == tokens,
                     f"rank_{rank} block {index} token count mismatch")
        tails = [max(_number(block["token_latency_ms"][token], "token latency")
                     for block in rank_blocks) for token in range(tokens)]
        tails_by_mode[mode].extend(tails)
        expected = 0 if mode == "fg_only" else local_bytes
        byte_ok = True
        for rank, block in enumerate(rank_blocks):
            source = rank < RANKS_PER_NODE
            expected_source = expected if source else 0
            expected_receive = expected if not source else 0
            byte_ok &= (
                block.get("expected_source_bytes") == expected_source
                and block.get("expected_receive_bytes") == expected_receive
                and block.get("background_completed_bytes") == expected_source
                and block.get("receiver_verified_bytes") == expected_receive
                and block.get("correctness_met") is True
                and block.get("transfer_errors") == []
            )
        aggregate = result_blocks[index]
        global_expected = expected * RANKS_PER_NODE
        byte_ok &= (
            aggregate.get("mode") == mode
            and aggregate.get("expected_background_bytes") == global_expected
            and aggregate.get("background_completed_bytes") == global_expected
            and aggregate.get("receiver_verified_bytes") == global_expected
            and aggregate.get("correctness_met") is True
            and aggregate.get("transfer_errors") == []
        )
        exact_bytes_and_correctness &= byte_ok
        metric = slo_metrics(tails)
        block_metrics.append({
            "block_index": index,
            "mode": mode,
            "p50_ms": statistics.median(tails),
            "p99_ms": percentile(tails, 0.99),
            "slo_4ms": metric,
            "exact_bytes_and_correctness_met": byte_ok,
        })
        if mode == "tempo_epoch":
            tempo_source_blocks.extend(rank_blocks[:RANKS_PER_NODE])

    exact_bytes_and_correctness &= result.get("overall_correctness_met") is True
    _require("tempo_epoch" in tails_by_mode and "lmcache_greedy" in tails_by_mode,
             "campaign requires tempo_epoch and lmcache_greedy modes")

    adherence = all(
        block.get("schedule_start_adherence_met") is True
        and all(record.get("started_within_scheduled_token") is True
                for record in block.get("transfer_records", []))
        for block in tempo_source_blocks
    )
    absolute_deadline = all(
        block.get("candidate_relative_token_deadline_used") is False
        and block.get("absolute_service_deadline_met") is True
        and round(_number(block.get("background_finish_from_block_start_ms"), "finish") * 1_000_000)
            <= block.get("absolute_service_deadline_ns", -1)
        and all(record.get("finished_by_plan_deadline") is True
                and record.get("deadline_semantics") == "absolute_from_block_start"
                for record in block.get("transfer_records", []))
        for block in tempo_source_blocks
    )
    no_drain = all(
        _number(block.get("post_foreground_drain_ms"), "drain") <= ZERO_MS
        and block.get("no_post_foreground_drain_met") is True
        for block in tempo_source_blocks
    )
    lag_values = []
    for block in tempo_source_blocks:
        if "lag_model_validated" in block:
            lag_values.append(block["lag_model_validated"] is True)
        elif "actual_start_lag_cap_met" in block:
            lag_values.append(block["actual_start_lag_cap_met"] is True)
    lag_model_validated = all(lag_values) if len(lag_values) == len(tempo_source_blocks) else None
    service_execution_valid = exact_bytes_and_correctness and adherence and absolute_deadline and no_drain

    modes = {}
    for mode, tails in tails_by_mode.items():
        modes[mode] = {
            "p50_ms": statistics.median(tails),
            "p99_ms": percentile(tails, 0.99),
            "slo_4ms": slo_metrics(tails),
        }
    greedy, tempo = modes["lmcache_greedy"], modes["tempo_epoch"]
    greedy_blocks = [item for item in block_metrics if item["mode"] == "lmcache_greedy"]
    tempo_blocks = [item for item in block_metrics if item["mode"] == "tempo_epoch"]
    _require(len(greedy_blocks) == len(tempo_blocks), "unpaired greedy/TEMPO occurrences")
    pairs = []
    for occurrence, (greedy_block, tempo_block) in enumerate(zip(greedy_blocks, tempo_blocks)):
        pairs.append({
            "occurrence": occurrence,
            "tempo_wins_p99": tempo_block["p99_ms"] < greedy_block["p99_ms"],
            "tempo_wins_4ms_goodput": (
                tempo_block["slo_4ms"]["time_normalized_goodput_tokens_per_s"]
                > greedy_block["slo_4ms"]["time_normalized_goodput_tokens_per_s"]
            ),
        })
    variant = _infer_variant(result, tempo_source_blocks)
    return {
        "label": label or variant,
        "result_path": str(result_path),
        "variant": variant,
        "schedule_id": _schedule_id(result, tempo_source_blocks, variant),
        "validation": {
            "exact_bytes_and_correctness_met": exact_bytes_and_correctness,
            "schedule_start_adherence_met": adherence,
            "absolute_service_deadline_met": absolute_deadline,
            "zero_post_foreground_drain_met": no_drain,
            "service_execution_valid": service_execution_valid,
            "lag_model_validated": lag_model_validated,
        },
        "modes": modes,
        "tempo_vs_greedy": {
            "p50_delta_ms": tempo["p50_ms"] - greedy["p50_ms"],
            "p99_delta_ms": tempo["p99_ms"] - greedy["p99_ms"],
            "success_fraction_delta": tempo["slo_4ms"]["success_fraction"] - greedy["slo_4ms"]["success_fraction"],
            "time_normalized_goodput_delta_tokens_per_s": (
                tempo["slo_4ms"]["time_normalized_goodput_tokens_per_s"]
                - greedy["slo_4ms"]["time_normalized_goodput_tokens_per_s"]
            ),
            "paired_occurrences": pairs,
            "paired_p99_wins": sum(pair["tempo_wins_p99"] for pair in pairs),
            "paired_4ms_goodput_wins": sum(pair["tempo_wins_4ms_goodput"] for pair in pairs),
        },
    }


def analyze_campaign(runs: Sequence[tuple[str | None, Path]]) -> dict[str, Any]:
    _require(bool(runs), "at least one --run is required")
    analyzed = [analyze_run(path, label) for label, path in runs]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in analyzed:
        grouped[run["schedule_id"]].append(run)
    schedules = []
    for schedule_id, repeats in grouped.items():
        service_valid = all(run["validation"]["service_execution_valid"] for run in repeats)
        goodput_wins = all(run["tempo_vs_greedy"]["time_normalized_goodput_delta_tokens_per_s"] > 0
                           for run in repeats)
        tail_wins = all(run["tempo_vs_greedy"]["p99_delta_ms"] < 0 for run in repeats)
        repeatable_goodput = len(repeats) >= 2 and service_valid and goodput_wins
        schedules.append({
            "schedule_id": schedule_id,
            "labels": [run["label"] for run in repeats],
            "repeat_count": len(repeats),
            "all_service_execution_valid": service_valid,
            "all_lag_models_validated": all(run["validation"]["lag_model_validated"] is True for run in repeats),
            "repeatable_goodput_signal": repeatable_goodput,
            "repeatable_tail_signal": len(repeats) >= 2 and service_valid and tail_wins,
            "outcome": "repeatable_goodput_signal" if repeatable_goodput else "no_repeatable_goodput_signal",
        })
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "claim_scope": "research_screen_not_promotion",
        "runs": analyzed,
        "schedules": schedules,
    }


def _run_spec(value: str) -> tuple[str | None, Path]:
    if "=" not in value:
        return None, Path(value)
    label, raw_path = value.split("=", 1)
    _require(bool(label) and bool(raw_path), "--run must be label=path or path")
    return label, Path(raw_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=result.json or result.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = analyze_campaign([_run_spec(value) for value in args.run])
    except AnalysisError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
