#!/usr/bin/env python3
"""Fail-closed analysis for three same-allocation 4-node TEMPO campaigns.

The analyzer deliberately does not bootstrap nine observations as if they were
independent allocations.  It reports robust descriptive statistics and paired
same-prompt-occurrence comparisons only.  A campaign is never promotion
evidence: all three runs share one allocation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


OUTPUT_SCHEMA_VERSION = "tempo-vllm-lmcache-4node-campaign-analysis-1"
INPUT_RUNS = 3
NODES = 4
WORLD_SIZE = 16
MODES = ("fg_only", "lmcache_greedy", "tempo_group2")
SAMPLES_PER_MODE_PER_RUN = 3
POOLED_SAMPLES_PER_MODE = INPUT_RUNS * SAMPLES_PER_MODE_PER_RUN
BACKGROUND_BYTES = 128 * 1024 * 1024
BACKGROUND_SOURCE_CALLS = 8
METRICS = (
    "ttft_ms",
    "tpot_p50_ms",
    "request_e2e_ms",
    "background_finish_from_request_start_ms",
)
BACKGROUND_MODES = frozenset(("lmcache_greedy", "tempo_group2"))
SOURCE_CALL_FIELDS = (
    "background_source_calls",
    "source_calls",
    "coalesced_source_calls",
)
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ContractError(ValueError):
    """Raised when an input cannot support the claimed comparison."""


@dataclass(frozen=True)
class RunInput:
    label: str
    path: str
    payload: Mapping[str, Any]


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    return value


def _require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{where} must be boolean")
    return value


def _require_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{where} must be an integer")
    return value


def _require_number(value: Any, where: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where} must be finite")
    if nonnegative and result < 0.0:
        raise ContractError(f"{where} must be nonnegative")
    return result


def _allocation_id(payload: Mapping[str, Any], label: str) -> str:
    config = payload.get("config")
    config_map = config if isinstance(config, Mapping) else {}
    candidates = (
        payload.get("allocation_id"),
        payload.get("slurm_job_id"),
        payload.get("job_id"),
        config_map.get("allocation_id"),
        config_map.get("slurm_job_id"),
        config_map.get("job_id"),
    )
    values = {str(value).strip() for value in candidates if value is not None and str(value).strip()}
    if not values:
        raise ContractError(f"run {label}: explicit allocation_id/slurm_job_id is required")
    if len(values) != 1:
        raise ContractError(f"run {label}: conflicting allocation identifiers: {sorted(values)}")
    return values.pop()


def _source_calls(block: Mapping[str, Any], where: str) -> int:
    present = [key for key in SOURCE_CALL_FIELDS if key in block]
    if not present:
        raise ContractError(
            f"{where}: one explicit source-call field is required: {', '.join(SOURCE_CALL_FIELDS)}"
        )
    values = {_require_int(block[key], f"{where}.{key}") for key in present}
    if len(values) != 1:
        raise ContractError(f"{where}: conflicting source-call counts: {sorted(values)}")
    return values.pop()


def robust_summary(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ContractError("cannot summarize an empty sample")
    center = float(statistics.median(samples))
    return {
        "samples": len(samples),
        "median": center,
        "min": min(samples),
        "max": max(samples),
        "mad": float(statistics.median(abs(value - center) for value in samples)),
    }


def _p50_max(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ContractError("cannot summarize an empty run")
    return {
        "samples": len(samples),
        "p50": float(statistics.median(samples)),
        "max": max(samples),
    }


def _validate_top_level_booleans(payload: Mapping[str, Any], label: str) -> None:
    for key in ("overall_correctness_met", "output_equivalence_met", "promotion_valid"):
        if key not in payload:
            raise ContractError(f"run {label}: missing top-level {key}")
    if not _require_bool(payload["overall_correctness_met"], f"run {label}.overall_correctness_met"):
        raise ContractError(f"run {label}: overall correctness failed")
    if not _require_bool(payload["output_equivalence_met"], f"run {label}.output_equivalence_met"):
        raise ContractError(f"run {label}: output equivalence failed")
    if _require_bool(payload["promotion_valid"], f"run {label}.promotion_valid"):
        raise ContractError(f"run {label}: component campaign must not claim promotion validity")


def _validate_contract_header(payload: Mapping[str, Any], label: str) -> None:
    if _require_int(payload.get("nodes"), f"run {label}.nodes") != NODES:
        raise ContractError(f"run {label}: expected {NODES} nodes")
    if _require_int(payload.get("world_size"), f"run {label}.world_size") != WORLD_SIZE:
        raise ContractError(f"run {label}: expected world_size={WORLD_SIZE}")

    contract = _require_mapping(payload.get("coalesced_contract"), f"run {label}.coalesced_contract")
    if _require_int(contract.get("global_bytes"), f"run {label}.coalesced_contract.global_bytes") != BACKGROUND_BYTES:
        raise ContractError(f"run {label}: coalesced contract must move exactly {BACKGROUND_BYTES} bytes")
    active_sources = contract.get("active_sources", contract.get("active_pairs"))
    if not isinstance(active_sources, list) or len(active_sources) != BACKGROUND_SOURCE_CALLS:
        raise ContractError(f"run {label}: coalesced contract must name exactly {BACKGROUND_SOURCE_CALLS} sources")
    if len(set(active_sources)) != BACKGROUND_SOURCE_CALLS:
        raise ContractError(f"run {label}: coalesced contract source identifiers must be unique")
    if _require_int(contract.get("calls_per_source"), f"run {label}.coalesced_contract.calls_per_source") != 1:
        raise ContractError(f"run {label}: expected one call per source")


def _validate_run(run: RunInput) -> tuple[dict[str, Any], dict[str, list[Mapping[str, Any]]], dict[str, dict[str, Mapping[str, Any]]]]:
    if not run.label or not LABEL_RE.fullmatch(run.label):
        raise ContractError(f"invalid run label {run.label!r}")
    payload = _require_mapping(run.payload, f"run {run.label}")
    allocation_id = _allocation_id(payload, run.label)
    _validate_top_level_booleans(payload, run.label)
    _validate_contract_header(payload, run.label)

    blocks_value = payload.get("blocks")
    if not isinstance(blocks_value, list):
        raise ContractError(f"run {run.label}.blocks must be an array")
    if len(blocks_value) != len(MODES) * SAMPLES_PER_MODE_PER_RUN:
        raise ContractError(f"run {run.label}: expected exactly 9 blocks")

    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_occurrence: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    output_hashes: dict[str, set[str]] = defaultdict(set)
    mode_gate_details: dict[str, dict[str, Any]] = {}

    for index, raw_block in enumerate(blocks_value):
        where = f"run {run.label}.blocks[{index}]"
        block = _require_mapping(raw_block, where)
        mode = block.get("mode")
        if mode not in MODES:
            raise ContractError(f"{where}.mode must be one of {MODES}")
        if not _require_bool(block.get("correctness_met"), f"{where}.correctness_met"):
            raise ContractError(f"{where}: correctness failed")
        errors = block.get("transfer_errors")
        if errors not in (None, []):
            raise ContractError(f"{where}: transfer_errors is not empty")

        occurrence_value = block.get("prompt_occurrence", block.get("prompt_index"))
        if occurrence_value is None or isinstance(occurrence_value, (list, dict)):
            raise ContractError(f"{where}: prompt_occurrence or prompt_index is required")
        occurrence = str(occurrence_value)
        if mode in by_occurrence[occurrence]:
            raise ContractError(f"run {run.label}: duplicate {mode} for prompt occurrence {occurrence}")

        digest = block.get("output_token_sha256")
        if not isinstance(digest, str) or not digest:
            raise ContractError(f"{where}.output_token_sha256 must be nonempty")
        output_hashes[occurrence].add(digest)

        expected_bytes = 0 if mode == "fg_only" else BACKGROUND_BYTES
        expected_calls = 0 if mode == "fg_only" else BACKGROUND_SOURCE_CALLS
        for key in (
            "expected_background_bytes",
            "background_completed_bytes",
            "receiver_verified_bytes",
        ):
            actual = _require_int(block.get(key), f"{where}.{key}")
            if actual != expected_bytes:
                raise ContractError(f"{where}.{key}: expected {expected_bytes}, got {actual}")
        actual_calls = _source_calls(block, where)
        if actual_calls != expected_calls:
            raise ContractError(f"{where}: expected {expected_calls} source calls, got {actual_calls}")

        for metric in METRICS:
            _require_number(block.get(metric), f"{where}.{metric}")
        _require_number(block.get("post_foreground_drain_ms"), f"{where}.post_foreground_drain_ms")
        _require_bool(block.get("schedule_start_adherence_met"), f"{where}.schedule_start_adherence_met")
        _require_bool(block.get("absolute_service_deadline_met"), f"{where}.absolute_service_deadline_met")

        by_mode[mode].append(block)
        by_occurrence[occurrence][mode] = block

    counts = Counter({mode: len(blocks) for mode, blocks in by_mode.items()})
    if any(counts.get(mode, 0) != SAMPLES_PER_MODE_PER_RUN for mode in MODES):
        raise ContractError(f"run {run.label}: expected three blocks per mode, got {dict(counts)}")
    if len(by_occurrence) != SAMPLES_PER_MODE_PER_RUN:
        raise ContractError(f"run {run.label}: expected three prompt occurrences")
    for occurrence, mode_blocks in by_occurrence.items():
        if set(mode_blocks) != set(MODES):
            raise ContractError(f"run {run.label}: prompt occurrence {occurrence} lacks a mode")
        if len(output_hashes[occurrence]) != 1:
            raise ContractError(f"run {run.label}: output mismatch for prompt occurrence {occurrence}")

    declared_equivalence = payload.get("output_equivalence_by_prompt")
    if not isinstance(declared_equivalence, Mapping) or not declared_equivalence:
        raise ContractError(f"run {run.label}: output_equivalence_by_prompt is required")
    if set(map(str, declared_equivalence.keys())) != set(by_occurrence):
        raise ContractError(f"run {run.label}: output-equivalence prompt keys do not match blocks")
    if not all(_require_bool(value, f"run {run.label}.output_equivalence_by_prompt") for value in declared_equivalence.values()):
        raise ContractError(f"run {run.label}: declared prompt equivalence failed")

    mode_summaries: dict[str, Any] = {}
    for mode in MODES:
        blocks = by_mode[mode]
        adherence = all(bool(block["schedule_start_adherence_met"]) for block in blocks)
        deadline = all(bool(block["absolute_service_deadline_met"]) for block in blocks)
        drains = [float(block["post_foreground_drain_ms"]) for block in blocks]
        mode_gate_details[mode] = {
            "schedule_adherence_met": adherence,
            "absolute_deadline_met": deadline,
            "no_post_foreground_drain_met": all(value == 0.0 for value in drains),
            "post_foreground_drain_p50_ms": float(statistics.median(drains)),
            "post_foreground_drain_max_ms": max(drains),
        }
        mode_summaries[mode] = {
            "samples": len(blocks),
            "metrics": {
                metric: _p50_max(float(block[metric]) for block in blocks)
                for metric in METRICS
            },
            "gates": mode_gate_details[mode],
        }

    tempo_gates = mode_gate_details["tempo_group2"]
    declared_gate_fields = {
        "candidate_schedule_adherence_met": tempo_gates["schedule_adherence_met"],
        "candidate_absolute_deadline_met": tempo_gates["absolute_deadline_met"],
        "candidate_no_post_foreground_drain_met": tempo_gates["no_post_foreground_drain_met"],
    }
    for key, computed in declared_gate_fields.items():
        if key not in payload:
            raise ContractError(f"run {run.label}: missing top-level {key}")
        declared = _require_bool(payload[key], f"run {run.label}.{key}")
        if declared != computed:
            raise ContractError(f"run {run.label}: {key} disagrees with block-level evidence")

    report = {
        "label": run.label,
        "source_path": run.path,
        "allocation_id": allocation_id,
        "checks": {
            "correctness_met": True,
            "output_equivalence_met": True,
            "background_bytes_per_block": BACKGROUND_BYTES,
            "background_bytes_met": True,
            "source_calls_per_background_block": BACKGROUND_SOURCE_CALLS,
            "source_calls_met": True,
            "tempo_schedule_adherence_met": tempo_gates["schedule_adherence_met"],
            "tempo_absolute_deadline_met": tempo_gates["absolute_deadline_met"],
            "tempo_no_post_foreground_drain_met": tempo_gates["no_post_foreground_drain_met"],
        },
        "mode_summaries": mode_summaries,
    }
    return report, dict(by_mode), dict(by_occurrence)


def _paired_comparison(
    normalized: Sequence[tuple[RunInput, dict[str, dict[str, Mapping[str, Any]]]]],
    baseline: str,
    metric: str,
) -> dict[str, Any]:
    deltas: list[float] = []
    improvements: list[float] = []
    wins = ties = losses = 0
    pairs: list[dict[str, Any]] = []
    for run, occurrences in normalized:
        for occurrence in sorted(occurrences):
            tempo = float(occurrences[occurrence]["tempo_group2"][metric])
            base = float(occurrences[occurrence][baseline][metric])
            delta = tempo - base
            deltas.append(delta)
            if math.isclose(tempo, base, rel_tol=0.0, abs_tol=1e-12):
                outcome = "tie"
                ties += 1
            elif tempo < base:
                outcome = "win"
                wins += 1
            else:
                outcome = "loss"
                losses += 1
            improvement = None if base == 0.0 else 100.0 * (base - tempo) / base
            if improvement is not None:
                improvements.append(improvement)
            pairs.append(
                {
                    "run": run.label,
                    "prompt_occurrence": occurrence,
                    "tempo": tempo,
                    "baseline": base,
                    "tempo_minus_baseline": delta,
                    "improvement_pct": improvement,
                    "outcome": outcome,
                }
            )

    foreground_background_finish = baseline == "fg_only" and metric == "background_finish_from_request_start_ms"
    result: dict[str, Any] = {
        "paired_samples": len(deltas),
        "lower_is_better": True,
        "tempo_minus_baseline": robust_summary(deltas),
        "win_rate": None if foreground_background_finish else wins / len(deltas),
        "wins": None if foreground_background_finish else wins,
        "ties": None if foreground_background_finish else ties,
        "losses": None if foreground_background_finish else losses,
        "improvement_pct": robust_summary(improvements) if improvements else None,
        "pairs": pairs,
    }
    if foreground_background_finish:
        result["comparison_interpretation"] = (
            "not_comparable: fg_only performs no background transfer; absolute paired deltas are retained"
        )
    return result


def analyze_runs(runs: Sequence[RunInput]) -> dict[str, Any]:
    if len(runs) != INPUT_RUNS:
        raise ContractError(f"exactly {INPUT_RUNS} --run inputs are required")
    labels = [run.label for run in runs]
    if len(set(labels)) != INPUT_RUNS:
        raise ContractError("run labels must be unique")
    paths = [str(Path(run.path).resolve()) for run in runs]
    if len(set(paths)) != INPUT_RUNS:
        raise ContractError("run paths must be unique")

    reports: list[dict[str, Any]] = []
    normalized_modes: list[tuple[RunInput, dict[str, list[Mapping[str, Any]]]]] = []
    normalized_pairs: list[tuple[RunInput, dict[str, dict[str, Mapping[str, Any]]]]] = []
    for run in runs:
        report, modes, occurrences = _validate_run(run)
        reports.append(report)
        normalized_modes.append((run, modes))
        normalized_pairs.append((run, occurrences))

    allocation_ids = {report["allocation_id"] for report in reports}
    if len(allocation_ids) != 1:
        raise ContractError(f"the three runs are not from one allocation: {sorted(allocation_ids)}")
    allocation_id = allocation_ids.pop()

    pooled_modes: dict[str, Any] = {}
    for mode in MODES:
        blocks = [block for _, mode_map in normalized_modes for block in mode_map[mode]]
        if len(blocks) != POOLED_SAMPLES_PER_MODE:
            raise ContractError(f"mode {mode}: expected {POOLED_SAMPLES_PER_MODE} pooled samples")
        pooled_modes[mode] = {
            "samples": len(blocks),
            "metrics": {
                metric: robust_summary(float(block[metric]) for block in blocks)
                for metric in METRICS
            },
        }

    comparisons: dict[str, Any] = {}
    for baseline in ("lmcache_greedy", "fg_only"):
        key = f"tempo_group2_vs_{baseline}"
        comparisons[key] = {
            "pairing": "same run label and prompt occurrence",
            "metrics": {
                metric: _paired_comparison(normalized_pairs, baseline, metric)
                for metric in METRICS
            },
        }

    tempo_checks = [report["checks"] for report in reports]
    all_runtime_gates_met = all(
        checks["tempo_schedule_adherence_met"]
        and checks["tempo_absolute_deadline_met"]
        and checks["tempo_no_post_foreground_drain_met"]
        for checks in tempo_checks
    )
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_method": "descriptive_robust_no_bootstrap",
        "claim_scope": "same_allocation_three_campaigns_research_prototype_only",
        "promotion_valid": False,
        "experimental_design": {
            "same_allocation_three_campaigns": True,
            "allocation_id": allocation_id,
            "allocation_independence": False,
            "independence_note": (
                "three temporal repeats on one allocation; pooled samples are not treated as independent allocations"
            ),
            "run_count": INPUT_RUNS,
            "samples_per_mode_per_run": SAMPLES_PER_MODE_PER_RUN,
            "pooled_samples_per_mode": POOLED_SAMPLES_PER_MODE,
            "paired_by": ["run_label", "prompt_occurrence"],
        },
        "contract": {
            "nodes": NODES,
            "world_size": WORLD_SIZE,
            "background_bytes_per_block": BACKGROUND_BYTES,
            "source_calls_per_background_block": BACKGROUND_SOURCE_CALLS,
            "modes": list(MODES),
        },
        "validation": {
            "all_three_run_contracts_met": True,
            "all_tempo_runtime_gates_met": all_runtime_gates_met,
            "all_tempo_schedule_adherence_met": all(
                checks["tempo_schedule_adherence_met"] for checks in tempo_checks
            ),
            "all_tempo_absolute_deadlines_met": all(
                checks["tempo_absolute_deadline_met"] for checks in tempo_checks
            ),
            "all_tempo_no_post_foreground_drain_met": all(
                checks["tempo_no_post_foreground_drain_met"] for checks in tempo_checks
            ),
        },
        "runs": reports,
        "pooled_modes": pooled_modes,
        "paired_comparisons": comparisons,
    }


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ContractError("--run must be LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    if not label or not LABEL_RE.fullmatch(label):
        raise ContractError(f"invalid --run label {label!r}")
    if not raw_path:
        raise ContractError(f"--run {label}: path is empty")
    return label, Path(raw_path)


def load_runs(specs: Sequence[str]) -> list[RunInput]:
    if len(specs) != INPUT_RUNS:
        raise ContractError(f"exactly {INPUT_RUNS} --run inputs are required")
    loaded: list[RunInput] = []
    for spec in specs:
        label, path = parse_run_spec(spec)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read run {label} from {path}: {exc}") from exc
        loaded.append(RunInput(label=label, path=str(path), payload=_require_mapping(payload, str(path))))
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=RESULT_JSON",
        help="one campaign result; provide exactly three distinct labels and paths",
    )
    parser.add_argument("--output", type=Path, required=True, help="analysis JSON to create")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = analyze_runs(load_runs(args.run))
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except ContractError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
