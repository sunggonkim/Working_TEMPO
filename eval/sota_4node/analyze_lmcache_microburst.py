#!/usr/bin/env python3
"""Analyze repeated two-node LMCache/TEMPO microburst screens.

The analyzer deliberately opens only the result paths supplied on the command
line and the eight explicitly named rank files next to each result.  It does
not discover runs or walk directories.  The reported goodput is a synthetic
global-token SLO goodput: successful global token steps divided by the sum of
their global (maximum-rank) step latencies.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


WORLD_SIZE = 8
NODES = 2
MODE_ORDER = (
    "fg_only",
    "lmcache_greedy",
    "lmcache_static_serial",
    "tempo_epoch",
)
LATIN_ROWS = tuple(
    tuple(MODE_ORDER[(column + row) % len(MODE_ORDER)] for column in range(len(MODE_ORDER)))
    for row in range(len(MODE_ORDER))
)
BLOCK_MODES = tuple(mode for row in LATIN_ROWS for mode in row)
SLO_THRESHOLDS_MS = (4.0, 4.5, 5.0)
RESULT_SCHEMA = "tempo-lmcache-epoch-2node-2"
RANK_SCHEMA = "tempo-lmcache-epoch-rank-2"
ANALYSIS_SCHEMA = "tempo-lmcache-microburst-analysis-1"
ZERO_TOLERANCE_MS = 1.0e-9


class AnalysisError(ValueError):
    """Raised when a supplied run is not valid comparable evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return value


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    _require(type(value) is int, f"{field} must be an integer")
    if positive:
        _require(value > 0, f"{field} must be positive")
    return value


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    _require(type(value) in (int, float), f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    if positive:
        _require(result > 0.0, f"{field} must be positive")
    else:
        _require(result >= 0.0, f"{field} must be nonnegative")
    return result


def _true(value: Any, field: str) -> None:
    _require(value is True, f"{field} must be true")


def _false(value: Any, field: str) -> None:
    _require(value is False, f"{field} must be false")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {label} {path}: {exc}") from exc
    return _mapping(value, label)


def percentile(values: Iterable[float], fraction: float) -> float:
    """Use the same nearest-rank percentile definition as the runner."""

    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "percentile requires at least one value")
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _close(actual: float, expected: float, field: str) -> None:
    _require(
        math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=1.0e-9),
        f"{field} differs from rank-derived value ({actual} != {expected})",
    )


def _lower_is_better_percent(baseline: float, candidate: float) -> float:
    _require(baseline > 0.0, "comparison baseline must be positive")
    return (baseline - candidate) / baseline * 100.0


def _higher_is_better_percent(baseline: float, candidate: float) -> float | None:
    if baseline == 0.0:
        return None
    return (candidate - baseline) / baseline * 100.0


def _slo_goodput(tails_ms: Sequence[float], threshold_ms: float) -> dict[str, Any]:
    _require(bool(tails_ms), "SLO goodput requires token samples")
    elapsed_ms = sum(tails_ms)
    _require(elapsed_ms > 0.0, "SLO goodput elapsed time must be positive")
    successful = sum(value <= threshold_ms for value in tails_ms)
    return {
        "threshold_ms": threshold_ms,
        "successful_global_tokens": successful,
        "total_global_tokens": len(tails_ms),
        "success_fraction": successful / len(tails_ms),
        "synthetic_global_token_slo_goodput_tokens_per_s": successful * 1000.0 / elapsed_ms,
    }


def _validate_result_header(result: dict[str, Any], result_path: Path) -> tuple[dict[str, Any], int]:
    _require(result_path.name == "result.json", "each input must be an explicitly named result.json")
    _require(result.get("schema_version") == RESULT_SCHEMA, "unexpected result schema_version")
    _require(result.get("world_size") == WORLD_SIZE, "result world_size must be 8")
    _require(result.get("nodes") == NODES, "result nodes must be 2")
    _require(tuple(result.get("block_sequence", ())) == BLOCK_MODES, "unexpected result block sequence")
    _require(
        result.get("evidence_state") == "live_official_component_with_compatibility_shim",
        "result is not official-component live evidence",
    )
    baseline = _mapping(result.get("baseline"), "result.baseline")
    _require(baseline.get("name") == "LMCache NixlChannel", "unexpected baseline name")
    _false(baseline.get("proxy"), "result.baseline.proxy")
    _true(result.get("overall_correctness_met"), "result.overall_correctness_met")
    _true(result.get("tempo_epoch_execution_valid"), "result.tempo_epoch_execution_valid")
    _require(
        result.get("screen_outcome") == "valid_measurement_requires_performance_comparison",
        "runner did not produce a valid performance measurement",
    )
    semantics = _mapping(result.get("scheduler_semantics"), "result.scheduler_semantics")
    _false(semantics.get("hot_path_global_control"), "scheduler_semantics.hot_path_global_control")
    config = _mapping(result.get("config"), "result.config")
    tokens = _integer(config.get("tokens"), "result.config.tokens", positive=True)
    _false(config.get("hot_path_global_control"), "result.config.hot_path_global_control")
    return config, tokens


def _load_and_validate_ranks(
    result_path: Path,
    result: dict[str, Any],
    config: dict[str, Any],
    tokens: int,
) -> list[dict[str, Any]]:
    ranks: list[dict[str, Any]] = []
    for rank in range(WORLD_SIZE):
        rank_path = result_path.parent / f"rank_{rank}.json"
        record = _load_object(rank_path, f"rank_{rank}")
        _require(record.get("schema_version") == RANK_SCHEMA, f"rank_{rank} schema mismatch")
        _require(record.get("rank") == rank, f"rank_{rank} contains the wrong rank")
        _require(record.get("world_size") == WORLD_SIZE, f"rank_{rank} world_size mismatch")
        _require(record.get("nodes") == NODES, f"rank_{rank} nodes mismatch")
        _require(record.get("config") == config, f"rank_{rank} config differs from result")
        blocks = _sequence(record.get("blocks"), f"rank_{rank}.blocks")
        _require(len(blocks) == len(BLOCK_MODES), f"rank_{rank} must contain 16 blocks")
        for block_index, expected_mode in enumerate(BLOCK_MODES):
            block = _mapping(blocks[block_index], f"rank_{rank}.blocks[{block_index}]")
            _require(block.get("block_index") == block_index, f"rank_{rank} block index mismatch")
            _require(block.get("mode") == expected_mode, f"rank_{rank} block mode mismatch")
            _true(block.get("correctness_met"), f"rank_{rank}.blocks[{block_index}].correctness_met")
            _require(block.get("transfer_errors") == [], f"rank_{rank} block has transfer errors")
            latencies = _sequence(
                block.get("token_latency_ms"), f"rank_{rank}.blocks[{block_index}].token_latency_ms"
            )
            _require(len(latencies) == tokens, f"rank_{rank} block token count mismatch")
            for token_index, value in enumerate(latencies):
                _number(
                    value,
                    f"rank_{rank}.blocks[{block_index}].token_latency_ms[{token_index}]",
                    positive=True,
                )
            if expected_mode == "tempo_epoch":
                _true(
                    block.get("schedule_start_adherence_met"),
                    f"rank_{rank}.blocks[{block_index}].schedule_start_adherence_met",
                )
                _true(
                    block.get("plan_deadline_met"),
                    f"rank_{rank}.blocks[{block_index}].plan_deadline_met",
                )
                drain = _number(
                    block.get("post_foreground_drain_ms"),
                    f"rank_{rank}.blocks[{block_index}].post_foreground_drain_ms",
                )
                _require(drain <= ZERO_TOLERANCE_MS, f"rank_{rank} TEMPO block has post-foreground drain")
        ranks.append(record)
    return ranks


def _rank_derived_tails(
    ranks: Sequence[dict[str, Any]], tokens: int
) -> tuple[dict[str, list[float]], list[list[float]]]:
    tails_by_mode = {mode: [] for mode in MODE_ORDER}
    tails_by_block: list[list[float]] = []
    for block_index, mode in enumerate(BLOCK_MODES):
        block_tails = [
            max(float(rank["blocks"][block_index]["token_latency_ms"][token]) for rank in ranks)
            for token in range(tokens)
        ]
        tails_by_block.append(block_tails)
        tails_by_mode[mode].extend(block_tails)
    return tails_by_mode, tails_by_block


def _validate_result_blocks(
    result: dict[str, Any],
    tails_by_mode: dict[str, list[float]],
    tails_by_block: Sequence[Sequence[float]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[float]]]:
    blocks = _sequence(result.get("blocks"), "result.blocks")
    _require(len(blocks) == len(BLOCK_MODES), "result must contain 16 blocks")
    finishes_by_mode = {mode: [] for mode in MODE_ORDER}
    for block_index, expected_mode in enumerate(BLOCK_MODES):
        block = _mapping(blocks[block_index], f"result.blocks[{block_index}]")
        _require(block.get("block_index") == block_index, "result block index mismatch")
        _require(block.get("mode") == expected_mode, "result block mode mismatch")
        _true(block.get("correctness_met"), f"result.blocks[{block_index}].correctness_met")
        _require(block.get("transfer_errors") == [], "result block has transfer errors")
        observed_p50 = _number(
            block.get("global_token_tail_p50_ms"),
            f"result.blocks[{block_index}].global_token_tail_p50_ms",
            positive=True,
        )
        observed_p99 = _number(
            block.get("global_token_tail_p99_ms"),
            f"result.blocks[{block_index}].global_token_tail_p99_ms",
            positive=True,
        )
        _close(observed_p50, statistics.median(tails_by_block[block_index]), "block tail p50")
        _close(observed_p99, percentile(tails_by_block[block_index], 0.99), "block tail p99")
        finish = _number(
            block.get("background_finish_from_block_start_ms"),
            f"result.blocks[{block_index}].background_finish_from_block_start_ms",
        )
        finishes_by_mode[expected_mode].append(finish)
        if expected_mode == "tempo_epoch":
            _true(block.get("schedule_start_adherence_met"), "TEMPO block schedule adherence")
            _true(block.get("plan_deadline_met"), "TEMPO block deadline")
            drain = _number(block.get("post_foreground_drain_ms"), "TEMPO block drain")
            _require(drain <= ZERO_TOLERANCE_MS, "TEMPO result block has post-foreground drain")

    result_modes = _mapping(result.get("modes"), "result.modes")
    _require(set(result_modes) == set(MODE_ORDER), "result modes differ from the four-mode screen")
    public_modes: dict[str, dict[str, Any]] = {}
    for mode in MODE_ORDER:
        observed = _mapping(result_modes[mode], f"result.modes.{mode}")
        _true(observed.get("correctness_met"), f"result.modes.{mode}.correctness_met")
        observed_p50 = _number(
            observed.get("global_token_tail_p50_ms"),
            f"result.modes.{mode}.global_token_tail_p50_ms",
            positive=True,
        )
        observed_p99 = _number(
            observed.get("global_token_tail_p99_ms"),
            f"result.modes.{mode}.global_token_tail_p99_ms",
            positive=True,
        )
        _close(observed_p50, statistics.median(tails_by_mode[mode]), f"{mode} tail p50")
        _close(observed_p99, percentile(tails_by_mode[mode], 0.99), f"{mode} tail p99")
        finish_p50 = _number(
            observed.get("background_finish_p50_ms"),
            f"result.modes.{mode}.background_finish_p50_ms",
        )
        finish_p99 = _number(
            observed.get("background_finish_p99_ms"),
            f"result.modes.{mode}.background_finish_p99_ms",
        )
        _close(finish_p50, statistics.median(finishes_by_mode[mode]), f"{mode} finish p50")
        _close(finish_p99, percentile(finishes_by_mode[mode], 0.99), f"{mode} finish p99")
        drain_p99 = _number(
            observed.get("post_foreground_drain_p99_ms"),
            f"result.modes.{mode}.post_foreground_drain_p99_ms",
        )
        if mode == "tempo_epoch":
            _true(observed.get("schedule_start_adherence_met"), "TEMPO mode schedule adherence")
            _true(observed.get("plan_deadline_met"), "TEMPO mode deadline")
            _require(drain_p99 <= ZERO_TOLERANCE_MS, "TEMPO mode has post-foreground drain")
        public_modes[mode] = {
            "global_token_tail_p50_ms": observed_p50,
            "global_token_tail_p99_ms": observed_p99,
            "background_finish_p50_ms": finish_p50,
            "background_finish_p99_ms": finish_p99,
            "post_foreground_drain_p99_ms": drain_p99,
            "slo_goodput": {
                f"{threshold:.1f}": _slo_goodput(tails_by_mode[mode], threshold)
                for threshold in SLO_THRESHOLDS_MS
            },
        }
    return public_modes, finishes_by_mode


def _paired_occurrences(
    result: dict[str, Any], tails_by_block: Sequence[Sequence[float]]
) -> dict[str, Any]:
    result_blocks = result["blocks"]
    greedy_indices = [index for index, mode in enumerate(BLOCK_MODES) if mode == "lmcache_greedy"]
    tempo_indices = [index for index, mode in enumerate(BLOCK_MODES) if mode == "tempo_epoch"]
    _require(len(greedy_indices) == len(tempo_indices) == 4, "expected four occurrences per mode")
    pairs: list[dict[str, Any]] = []
    tail_wins = completion_wins = goodput_wins = 0
    tail_ties = completion_ties = goodput_ties = 0
    for occurrence, (greedy_index, tempo_index) in enumerate(zip(greedy_indices, tempo_indices)):
        greedy_tail = percentile(tails_by_block[greedy_index], 0.99)
        tempo_tail = percentile(tails_by_block[tempo_index], 0.99)
        greedy_finish = float(result_blocks[greedy_index]["background_finish_from_block_start_ms"])
        tempo_finish = float(result_blocks[tempo_index]["background_finish_from_block_start_ms"])
        greedy_goodput = _slo_goodput(tails_by_block[greedy_index], 4.0)[
            "synthetic_global_token_slo_goodput_tokens_per_s"
        ]
        tempo_goodput = _slo_goodput(tails_by_block[tempo_index], 4.0)[
            "synthetic_global_token_slo_goodput_tokens_per_s"
        ]
        tail_wins += tempo_tail < greedy_tail
        completion_wins += tempo_finish < greedy_finish
        goodput_wins += tempo_goodput > greedy_goodput
        tail_ties += tempo_tail == greedy_tail
        completion_ties += tempo_finish == greedy_finish
        goodput_ties += tempo_goodput == greedy_goodput
        pairs.append(
            {
                "occurrence": occurrence,
                "greedy_block_index": greedy_index,
                "tempo_block_index": tempo_index,
                "greedy_tail_p99_ms": greedy_tail,
                "tempo_tail_p99_ms": tempo_tail,
                "tail_improvement_percent": _lower_is_better_percent(greedy_tail, tempo_tail),
                "greedy_completion_ms": greedy_finish,
                "tempo_completion_ms": tempo_finish,
                "completion_improvement_percent": _lower_is_better_percent(
                    greedy_finish, tempo_finish
                ),
                "greedy_4ms_slo_goodput_tokens_per_s": greedy_goodput,
                "tempo_4ms_slo_goodput_tokens_per_s": tempo_goodput,
                "tempo_wins_tail": tempo_tail < greedy_tail,
                "tempo_wins_completion": tempo_finish < greedy_finish,
                "tempo_wins_4ms_slo_goodput": tempo_goodput > greedy_goodput,
            }
        )
    return {
        "pairs": pairs,
        "summary": {
            "occurrences": len(pairs),
            "tempo_tail_wins": tail_wins,
            "tail_ties": tail_ties,
            "tempo_completion_wins": completion_wins,
            "completion_ties": completion_ties,
            "tempo_4ms_slo_goodput_wins": goodput_wins,
            "4ms_slo_goodput_ties": goodput_ties,
        },
    }


def _analyze_valid_run(result_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _load_object(result_path, "result")
    config, tokens = _validate_result_header(result, result_path)
    ranks = _load_and_validate_ranks(result_path, result, config, tokens)
    tails_by_mode, tails_by_block = _rank_derived_tails(ranks, tokens)
    public_modes, finishes_by_mode = _validate_result_blocks(result, tails_by_mode, tails_by_block)
    paired = _paired_occurrences(result, tails_by_block)

    greedy = public_modes["lmcache_greedy"]
    tempo = public_modes["tempo_epoch"]
    slo_comparison: dict[str, Any] = {}
    for threshold in SLO_THRESHOLDS_MS:
        key = f"{threshold:.1f}"
        greedy_rate = greedy["slo_goodput"][key][
            "synthetic_global_token_slo_goodput_tokens_per_s"
        ]
        tempo_rate = tempo["slo_goodput"][key][
            "synthetic_global_token_slo_goodput_tokens_per_s"
        ]
        slo_comparison[key] = {
            "greedy_tokens_per_s": greedy_rate,
            "tempo_tokens_per_s": tempo_rate,
            "tempo_improves": tempo_rate > greedy_rate,
            "improvement_percent": _higher_is_better_percent(greedy_rate, tempo_rate),
        }
    tail_improvement = _lower_is_better_percent(
        greedy["global_token_tail_p99_ms"], tempo["global_token_tail_p99_ms"]
    )
    completion_improvement = _lower_is_better_percent(
        greedy["background_finish_p50_ms"], tempo["background_finish_p50_ms"]
    )
    public = {
        "result_path": str(result_path),
        "valid": True,
        "validation": {
            "schema_valid": True,
            "all_rank_files_present": True,
            "overall_correctness_met": True,
            "tempo_epoch_execution_valid": True,
            "tempo_plan_deadline_met": True,
            "tempo_zero_post_foreground_drain": True,
        },
        "mode_metrics": public_modes,
        "tempo_vs_lmcache_greedy": {
            "tail_p99_improvement_percent": tail_improvement,
            "completion_p50_improvement_percent": completion_improvement,
            "slo_goodput": slo_comparison,
        },
        "paired_occurrences": paired,
    }
    internal = {
        "tails_by_mode": tails_by_mode,
        "finishes_by_mode": finishes_by_mode,
        "tail_improvement_percent": tail_improvement,
        "deadline_met": True,
        "zero_drain": True,
        "paired_summary": paired["summary"],
    }
    return public, internal


def _aggregate_valid_runs(internals: Sequence[dict[str, Any]]) -> dict[str, Any]:
    combined_tails = {mode: [] for mode in MODE_ORDER}
    combined_finishes = {mode: [] for mode in MODE_ORDER}
    paired_totals = {
        "occurrences": 0,
        "tempo_tail_wins": 0,
        "tail_ties": 0,
        "tempo_completion_wins": 0,
        "completion_ties": 0,
        "tempo_4ms_slo_goodput_wins": 0,
        "4ms_slo_goodput_ties": 0,
    }
    for internal in internals:
        for mode in MODE_ORDER:
            combined_tails[mode].extend(internal["tails_by_mode"][mode])
            combined_finishes[mode].extend(internal["finishes_by_mode"][mode])
        for key in paired_totals:
            paired_totals[key] += int(internal["paired_summary"][key])

    modes: dict[str, Any] = {}
    if internals:
        for mode in MODE_ORDER:
            modes[mode] = {
                "global_token_tail_p50_ms": statistics.median(combined_tails[mode]),
                "global_token_tail_p99_ms": percentile(combined_tails[mode], 0.99),
                "background_finish_p50_ms": statistics.median(combined_finishes[mode]),
                "background_finish_p99_ms": percentile(combined_finishes[mode], 0.99),
                "slo_goodput": {
                    f"{threshold:.1f}": _slo_goodput(combined_tails[mode], threshold)
                    for threshold in SLO_THRESHOLDS_MS
                },
            }

    comparison: dict[str, Any] = {}
    if internals:
        greedy = modes["lmcache_greedy"]
        tempo = modes["tempo_epoch"]
        slo = {}
        for threshold in SLO_THRESHOLDS_MS:
            key = f"{threshold:.1f}"
            greedy_rate = greedy["slo_goodput"][key][
                "synthetic_global_token_slo_goodput_tokens_per_s"
            ]
            tempo_rate = tempo["slo_goodput"][key][
                "synthetic_global_token_slo_goodput_tokens_per_s"
            ]
            slo[key] = {
                "greedy_tokens_per_s": greedy_rate,
                "tempo_tokens_per_s": tempo_rate,
                "tempo_improves": tempo_rate > greedy_rate,
                "improvement_percent": _higher_is_better_percent(greedy_rate, tempo_rate),
            }
        comparison = {
            "tail_p99_improvement_percent": _lower_is_better_percent(
                greedy["global_token_tail_p99_ms"], tempo["global_token_tail_p99_ms"]
            ),
            "completion_p50_improvement_percent": _lower_is_better_percent(
                greedy["background_finish_p50_ms"], tempo["background_finish_p50_ms"]
            ),
            "slo_goodput": slo,
            "paired_occurrence_wins": paired_totals,
        }
    return {"mode_metrics": modes, "tempo_vs_lmcache_greedy": comparison}


def analyze_paths(result_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Validate explicit runs, recompute rank-derived metrics, and apply gates."""

    _require(bool(result_paths), "at least one explicit result.json path is required")
    paths = [Path(path) for path in result_paths]
    normalized = [str(path.absolute()) for path in paths]
    _require(len(set(normalized)) == len(paths), "duplicate result paths are not allowed")

    public_runs: list[dict[str, Any]] = []
    internals: list[dict[str, Any]] = []
    for path in paths:
        try:
            public, internal = _analyze_valid_run(path)
        except (AnalysisError, KeyError, TypeError, IndexError) as exc:
            public_runs.append(
                {
                    "result_path": str(path),
                    "valid": False,
                    "validation_errors": [str(exc)],
                }
            )
        else:
            public_runs.append(public)
            internals.append(internal)

    aggregate = _aggregate_valid_runs(internals)
    run_count = len(paths)
    valid_count = len(internals)
    all_valid = valid_count == run_count
    improvements = [float(internal["tail_improvement_percent"]) for internal in internals]
    every_tail_improves = all_valid and all(value > 0.0 for value in improvements)
    required_ten_percent = math.ceil(run_count / 2)
    ten_percent_count = sum(value >= 10.0 for value in improvements)
    at_least_half_ten = all_valid and ten_percent_count >= required_ten_percent
    all_deadlines = all_valid and all(bool(item["deadline_met"]) for item in internals)
    all_zero_drain = all_valid and all(bool(item["zero_drain"]) for item in internals)
    aggregate_comparison = aggregate["tempo_vs_lmcache_greedy"]
    aggregate_4ms_improves = bool(
        all_valid
        and aggregate_comparison
        and aggregate_comparison["slo_goodput"]["4.0"]["tempo_improves"]
    )
    promising = all(
        (
            all_valid,
            every_tail_improves,
            at_least_half_ten,
            all_deadlines,
            all_zero_drain,
            aggregate_4ms_improves,
        )
    )
    gates = {
        "all_runs_valid": all_valid,
        "every_run_tail_p99_improves": every_tail_improves,
        "at_least_half_runs_tail_p99_improve_ge_10_percent": at_least_half_ten,
        "tail_p99_improve_ge_10_percent_run_count": ten_percent_count,
        "tail_p99_improve_ge_10_percent_required_count": required_ten_percent,
        "all_tempo_plan_deadlines_met": all_deadlines,
        "all_tempo_zero_post_foreground_drain": all_zero_drain,
        "aggregate_4ms_slo_goodput_improves": aggregate_4ms_improves,
    }
    if promising:
        verdict = "promising"
        reasons = ["all conservative continuation gates passed"]
    elif not all_valid:
        verdict = "inconclusive"
        reasons = ["one or more runs failed schema, correctness, execution, deadline, or drain validation"]
    else:
        verdict = "kill"
        reasons = [name for name, passed in gates.items() if type(passed) is bool and not passed]

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "evidence_scope": {
            "measurement": (
                "same-allocation synthetic two-node scheduler screen using the official "
                "LMCache NixlChannel component with the recorded compatibility shim"
            ),
            "same_allocation_status": (
                "required for paired interpretation but not independently encoded in this result schema"
            ),
            "goodput_definition": (
                "successful global token steps divided by summed maximum-rank token-step latency"
            ),
            "not_claimed": [
                "end-to-end production serving performance",
                "SOTA superiority",
                "independent NIXL transport attribution",
                "cross-allocation generalization",
            ],
        },
        "slo_thresholds_ms": list(SLO_THRESHOLDS_MS),
        "input_result_paths": [str(path) for path in paths],
        "run_count": run_count,
        "valid_run_count": valid_count,
        "runs": public_runs,
        "aggregate": aggregate,
        "gates": gates,
        "verdict": verdict,
        "verdict_reasons": reasons,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", nargs="+", type=Path, help="explicit result.json path(s)")
    parser.add_argument("--output", type=Path, help="optional JSON report path; parent must exist")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = analyze_paths(args.result_json)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        output_absolute = str(args.output.absolute())
        input_absolute = {str(path.absolute()) for path in args.result_json}
        if output_absolute in input_absolute:
            raise SystemExit("--output must not overwrite an input result.json")
        if not args.output.parent.is_dir():
            raise SystemExit("--output parent directory must already exist")
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"output": str(args.output), "verdict": report["verdict"]}, sort_keys=True))
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
