#!/usr/bin/env python3
"""Minimal single-run screen for a live v4-open versus C0 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "tempo-c0-live-screen-1"
RANKS = frozenset(range(4))
MIN_COLLECTIVE_TAIL_IMPROVEMENT = 0.03
MAX_SKEW_REGRESSION = 0.10


def _percentile(values: Iterable[float], quantile: float = 99.0) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile from no values")
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc


def _integer(row: Mapping[str, Any], field: str, source: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid or missing {field}") from exc


def _number(row: Mapping[str, Any], field: str, source: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid or missing {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}: non-finite {field}")
    return value


def _enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_arm(arm_dir: Path, *, expect_c0: bool) -> dict[str, Any]:
    collective_groups: dict[
        tuple[int, int, str, int], dict[int, tuple[dict[str, str], Path]]
    ] = defaultdict(dict)
    step_groups: dict[int, dict[int, tuple[dict[str, str], Path]]] = defaultdict(dict)
    events: list[tuple[dict[str, Any], Path]] = []
    summaries: list[dict[str, Any]] = []

    for rank in sorted(RANKS):
        collective_path = arm_dir / f"collectives_rank{rank}.csv"
        for row in _read_csv(collective_path):
            row_rank = _integer(row, "rank", collective_path)
            if row_rank != rank:
                raise ValueError(
                    f"{collective_path}: row rank {row_rank} does not match file rank {rank}"
                )
            step = _integer(row, "step", collective_path)
            if step < 0:
                continue
            key = (
                step,
                _integer(row, "phase_index", collective_path),
                str(row.get("phase_signature", "")),
                _integer(row, "sequence", collective_path),
            )
            if rank in collective_groups[key]:
                raise ValueError(f"{collective_path}: duplicate collective group {key}")
            collective_groups[key][rank] = (row, collective_path)

        step_path = arm_dir / f"steps_rank{rank}.csv"
        for row in _read_csv(step_path):
            row_rank = _integer(row, "rank", step_path)
            if row_rank != rank:
                raise ValueError(
                    f"{step_path}: row rank {row_rank} does not match file rank {rank}"
                )
            step = _integer(row, "step", step_path)
            if step < 0:
                continue
            if rank in step_groups[step]:
                raise ValueError(f"{step_path}: duplicate step {step}")
            step_groups[step][rank] = (row, step_path)

        event_path = arm_dir / f"checkpoint_events_rank{rank}.json"
        rank_events = _read_json(event_path)
        if not isinstance(rank_events, list) or not rank_events:
            raise ValueError(f"{event_path}: expected a non-empty JSON array")
        for event in rank_events:
            if not isinstance(event, dict):
                raise ValueError(f"{event_path}: checkpoint event is not an object")
            events.append((event, event_path))

        summary_path = arm_dir / f"summary_rank{rank}.json"
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            raise ValueError(f"{summary_path}: expected a JSON object")
        if _integer(summary, "rank", summary_path) != rank:
            raise ValueError(f"{summary_path}: rank does not match filename")
        if _integer(summary, "world_size", summary_path) != len(RANKS):
            raise ValueError(f"{summary_path}: expected world_size={len(RANKS)}")
        if str(summary.get("policy", "")) != "v4_open":
            raise ValueError(f"{summary_path}: expected policy=v4_open")
        if _enabled(summary.get("c0_enabled", False)) != expect_c0:
            raise ValueError(
                f"{summary_path}: c0_enabled does not match the {arm_dir.name} arm"
            )
        summaries.append(summary)

    complete_collective_groups = [
        rows for rows in collective_groups.values() if frozenset(rows) == RANKS
    ]
    active_groups = [
        rows
        for rows in complete_collective_groups
        if all(
            _enabled(row["checkpoint_active_at_ready"])
            and not _enabled(row.get("finalize_at_ready", False))
            for row, _source in rows.values()
        )
    ]
    if not active_groups:
        raise ValueError(f"{arm_dir}: no complete active collective groups")

    collective_tails: list[float] = []
    arrival_skews_ns: list[float] = []
    for rows in active_groups:
        collective_tails.append(
            max(_number(row, "gpu_ms", source) for row, source in rows.values())
        )
        corrected_arrivals = [
            _integer(row, "ready_corrected_ns", source)
            for row, source in rows.values()
        ]
        arrival_skews_ns.append(float(max(corrected_arrivals) - min(corrected_arrivals)))

    complete_step_groups = [
        rows for rows in step_groups.values() if frozenset(rows) == RANKS
    ]
    window_groups = [
        rows
        for rows in complete_step_groups
        if all(_enabled(row["checkpoint_window"]) for row, _source in rows.values())
    ]
    if not window_groups:
        raise ValueError(f"{arm_dir}: no complete checkpoint-window steps")
    window_step_tails = [
        max(_number(row, "step_ms", source) for row, source in rows.values())
        for rows in window_groups
    ]

    durable_values = [
        _number(event, "durable_ms", source) for event, source in events
    ]
    deadline_all_met = all(
        event.get("deadline_met") is True for event, _source in events
    )
    c0_rates = sorted(
        {
            int(summary["c0_d2h_rate_bps"])
            for summary in summaries
            if summary.get("c0_d2h_rate_bps") not in (None, 0, "", "0")
        }
    )

    return {
        "complete_collective_groups": len(complete_collective_groups),
        "active_complete_collective_groups": len(active_groups),
        "collective_slowest_rank_gpu_ms_p99": _percentile(collective_tails),
        "corrected_arrival_skew_ns_p99": _percentile(arrival_skews_ns),
        "complete_window_steps": len(window_groups),
        "window_slowest_rank_step_ms_p99": _percentile(window_step_tails),
        "checkpoint_events": len(events),
        "max_durable_ms": max(durable_values),
        "deadline_all_met": deadline_all_met,
        "c0_enabled": expect_c0,
        "c0_d2h_rate_bps": c0_rates[0] if len(c0_rates) == 1 else None,
    }


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0 if candidate >= baseline else math.inf
    return (baseline - candidate) / baseline


def analyze_c0_live(result_root: Path | str) -> dict[str, Any]:
    """Analyze one open/C0 pair; this is a screen, not causal evidence."""

    root = Path(result_root)
    open_metrics = _load_arm(root / "open", expect_c0=False)
    c0_metrics = _load_arm(root / "c0", expect_c0=True)

    tail_improvement = _relative_improvement(
        float(open_metrics["collective_slowest_rank_gpu_ms_p99"]),
        float(c0_metrics["collective_slowest_rank_gpu_ms_p99"]),
    )
    open_skew = float(open_metrics["corrected_arrival_skew_ns_p99"])
    c0_skew = float(c0_metrics["corrected_arrival_skew_ns_p99"])
    skew_limit = open_skew * (1.0 + MAX_SKEW_REGRESSION)
    tail_gate = tail_improvement >= MIN_COLLECTIVE_TAIL_IMPROVEMENT
    skew_gate = c0_skew <= skew_limit if open_skew > 0.0 else c0_skew == 0.0
    deadline_gate = bool(c0_metrics["deadline_all_met"])
    promising = tail_gate and skew_gate and deadline_gate

    reasons: list[str] = []
    if not tail_gate:
        reasons.append("C0 collective tail improvement is below 3%")
    if not skew_gate:
        reasons.append("C0 corrected arrival-skew regresses by more than 10%")
    if not deadline_gate:
        reasons.append("at least one C0 checkpoint missed its deadline")

    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_scope": {
            "kind": "single-run screen",
            "causal_claim": False,
            "promotion_decision": False,
            "note": "A promising result only justifies replication; it is not causal evidence or a promotion result.",
        },
        "thresholds": {
            "minimum_collective_tail_improvement_fraction": MIN_COLLECTIVE_TAIL_IMPROVEMENT,
            "maximum_corrected_arrival_skew_regression_fraction": MAX_SKEW_REGRESSION,
            "require_all_c0_deadlines": True,
        },
        "arms": {"open": open_metrics, "c0": c0_metrics},
        "comparison": {
            "collective_tail_improvement_fraction": tail_improvement,
            "collective_tail_improvement_percent": tail_improvement * 100.0,
            "corrected_arrival_skew_ratio": (
                c0_skew / open_skew if open_skew > 0.0 else None
            ),
            "window_step_tail_improvement_fraction": _relative_improvement(
                float(open_metrics["window_slowest_rank_step_ms_p99"]),
                float(c0_metrics["window_slowest_rank_step_ms_p99"]),
            ),
        },
        "gates": {
            "collective_tail_improves_at_least_3_percent": tail_gate,
            "corrected_arrival_skew_no_worse_than_10_percent": skew_gate,
            "all_c0_deadlines_met": deadline_gate,
        },
        "decision": {
            "promising": promising,
            "verdict": "promising" if promising else "kill/no-go",
            "next_step": "replicate" if promising else "stop C0",
            "reasons": reasons,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze_c0_live(args.result_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
