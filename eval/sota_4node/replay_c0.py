#!/usr/bin/env python3
"""Offline feasibility replay for the minimal C0 D2H controller.

This replay can answer whether a fixed D2H rate still leaves enough time for
the observed checkpoint bytes. It intentionally does not claim to predict the
collective benefit of max_inflight_bytes: the retained traces do not contain
per-request completion timestamps or an interference response model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "tempo-c0-offline-replay-1"
DEFAULT_JOBS = ("56859316", "56860098", "56861820", "56861979")
CALIBRATION_JOBS = frozenset(("56859316", "56860098"))


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def simulate_tandem(
    *,
    demand_batches: Iterable[tuple[int, int]],
    state_bytes: int,
    file_bytes: int,
    d2h_rate_bps: int,
    pfs_rate_bps: int,
    d2h_request_bytes: int,
    pfs_request_bytes: int,
    finalization_reserve_ns: int,
) -> dict[str, int]:
    """Replay D2H then PFS as two work-conserving constant-rate servers."""

    values = (
        state_bytes,
        file_bytes,
        d2h_rate_bps,
        pfs_rate_bps,
        d2h_request_bytes,
        pfs_request_bytes,
    )
    if any(value <= 0 for value in values):
        raise ValueError("bytes, rates, and request sizes must be positive")
    if file_bytes < state_bytes:
        raise ValueError("file_bytes must cover state_bytes")
    if finalization_reserve_ns < 0:
        raise ValueError("finalization_reserve_ns must be nonnegative")

    d2h_departures: list[tuple[int, int]] = []
    d2h_free_ns = 0
    cumulative = 0
    for arrival_ns, batch_bytes in sorted(demand_batches):
        if arrival_ns < 0 or batch_bytes <= 0:
            raise ValueError("demand batches need nonnegative time and positive bytes")
        remaining = batch_bytes
        while remaining:
            size = min(d2h_request_bytes, remaining)
            start_ns = max(arrival_ns, d2h_free_ns)
            d2h_free_ns = start_ns + _ceil_div(
                size * 1_000_000_000,
                d2h_rate_bps,
            )
            cumulative += size
            d2h_departures.append((cumulative, d2h_free_ns))
            remaining -= size

    if cumulative != state_bytes:
        raise ValueError(
            f"D2H demand sums to {cumulative} bytes, expected {state_bytes}"
        )

    pfs_free_ns = 0
    written = 0
    departure_index = 0
    while written < file_bytes:
        size = min(pfs_request_bytes, file_bytes - written)
        required_state = min(state_bytes, written + size)
        while (
            departure_index < len(d2h_departures)
            and d2h_departures[departure_index][0] < required_state
        ):
            departure_index += 1
        if departure_index >= len(d2h_departures):
            available_ns = d2h_free_ns
        else:
            available_ns = d2h_departures[departure_index][1]
        start_ns = max(available_ns, pfs_free_ns)
        pfs_free_ns = start_ns + _ceil_div(size * 1_000_000_000, pfs_rate_bps)
        written += size

    return {
        "d2h_finish_ns": d2h_free_ns,
        "pfs_finish_ns": pfs_free_ns,
        "predicted_completion_ns": pfs_free_ns + finalization_reserve_ns,
        "d2h_requests": len(d2h_departures),
        "pfs_requests": _ceil_div(file_bytes, pfs_request_bytes),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def _index_records(
    records: Iterable[dict[str, Any]],
    record_type: str,
) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("record_type") == record_type:
            indexed.setdefault(str(record["checkpoint_id"]), []).append(record)
    for values in indexed.values():
        values.sort(key=lambda item: int(item["monotonic_ns"]))
    return indexed


def _replay_rank(
    *,
    job_dir: Path,
    rank: int,
    selection: dict[str, Any],
    max_inflight_bytes: int,
    rate_bps: int,
) -> list[dict[str, Any]]:
    policy_dir = job_dir / "v4_open"
    records = _read_jsonl(policy_dir / f"tempo_v4_telemetry_rank{rank}.jsonl")
    finishes = _index_records(records, "finish")
    plans = _index_records(records, "plan")
    starts = _index_records(records, "start")
    events = json.loads(
        (policy_dir / f"checkpoint_events_rank{rank}.json").read_text(
            encoding="utf-8"
        )
    )

    ordered_finishes = sorted(
        (values[-1] for values in finishes.values()),
        key=lambda item: int(item["event_step"]),
    )
    if len(ordered_finishes) != len(events):
        raise ValueError(
            f"rank {rank}: {len(ordered_finishes)} finish records but "
            f"{len(events)} checkpoint events"
        )

    geometry = selection["geometry"]
    deadline = selection["deadline_feasibility"]
    d2h_request_bytes = int(geometry["d2h_request_bytes"])
    if max_inflight_bytes < d2h_request_bytes:
        raise ValueError("max_inflight_bytes must fit one physical D2H request")

    results: list[dict[str, Any]] = []
    for finish, event in zip(ordered_finishes, events):
        checkpoint_id = str(finish["checkpoint_id"])
        if checkpoint_id not in plans or checkpoint_id not in starts:
            raise ValueError(f"rank {rank} {checkpoint_id}: missing start or plan")
        plan = plans[checkpoint_id][0]
        start = starts[checkpoint_id][0]
        trace = finish["durability_evidence"]["admission_trace"]
        if trace["schema_version"] != "tempo-v4-admission-trace-2":
            raise ValueError(f"rank {rank} {checkpoint_id}: unsupported trace schema")

        batches = [
            (int(entry["activation_monotonic_ns"]), int(entry["d2h_bytes"]))
            for entry in trace["entries"]
            if int(entry["d2h_bytes"]) > 0
        ]
        if not batches:
            raise ValueError(f"rank {rank} {checkpoint_id}: empty D2H demand")
        first_activation_ns = min(time_ns for time_ns, _ in batches)
        trigger_corrected_ns = int(plan["trigger_corrected_ns"])
        first_issue_corrected_ns = int(event["d2h_first_issue_corrected_ns"])
        pre_d2h_ns = max(0, first_issue_corrected_ns - trigger_corrected_ns)
        relative_batches = [
            (pre_d2h_ns + time_ns - first_activation_ns, size)
            for time_ns, size in batches
        ]

        state_bytes = int(event["state_bytes_local"])
        file_bytes = int(
            finish["durability_evidence"].get(
                "checkpoint_file_bytes",
                event["checkpoint_file_bytes"],
            )
        )
        d2h_stats = finish["event_relative_stats"]["d2h"]
        demand_bytes = sum(size for _, size in batches)
        integrity_ok = (
            demand_bytes
            == state_bytes
            == int(d2h_stats["total_bytes"])
            == int(d2h_stats["completed_bytes"])
            and bool(finish["commit_validated"])
            and bool(finish["fsync_evidence_valid"])
        )
        if not integrity_ok:
            raise ValueError(f"rank {rank} {checkpoint_id}: byte/integrity mismatch")

        structural = start["structural"]
        if int(structural["d2h_chunk_bytes"]) != d2h_request_bytes:
            raise ValueError(f"rank {rank} {checkpoint_id}: D2H geometry drift")

        simulation = simulate_tandem(
            demand_batches=relative_batches,
            state_bytes=state_bytes,
            file_bytes=file_bytes,
            d2h_rate_bps=rate_bps,
            pfs_rate_bps=int(selection["down_only_selection"]["selected_pfs_bps"]),
            d2h_request_bytes=d2h_request_bytes,
            pfs_request_bytes=int(geometry["pfs_request_bytes"]),
            finalization_reserve_ns=int(deadline["finalization_reserve_ns"]),
        )
        horizon_ns = int(plan["deadline_corrected_ns"]) - trigger_corrected_ns
        usable_horizon_ns = horizon_ns - int(deadline["deadline_margin_ns"])
        no_drain_budget_ns = int(deadline["no_drain_bulk_budget_ns"])
        static_capacity = rate_bps * no_drain_budget_ns // 1_000_000_000
        static_shortfall = max(0, state_bytes - static_capacity)
        predicted_pass = (
            simulation["predicted_completion_ns"] <= usable_horizon_ns
            and static_shortfall == 0
        )
        actual_busy_ns = int(
            event["tier_stage_stats_end"]["logical_stage"]["d2h"]["busy_ns"]
        )
        actual_open_rate_bps = (
            state_bytes * 1_000_000_000 // actual_busy_ns if actual_busy_ns else 0
        )
        predicted_absolute_ns = (
            trigger_corrected_ns + simulation["predicted_completion_ns"]
        )

        results.append(
            {
                "checkpoint_id": checkpoint_id,
                "event_step": int(finish["event_step"]),
                "rank": rank,
                "demand_batches": len(batches),
                "state_bytes": state_bytes,
                "file_bytes": file_bytes,
                "candidate_d2h_rate_bps": rate_bps,
                "actual_open_d2h_rate_bps": actual_open_rate_bps,
                "predicted_d2h_ms": round(simulation["d2h_finish_ns"] / 1e6, 3),
                "predicted_durable_ms": round(
                    simulation["predicted_completion_ns"] / 1e6,
                    3,
                ),
                "actual_open_durable_ms": float(finish["durable_ms"]),
                "deadline_budget_ms": round(usable_horizon_ns / 1e6, 3),
                "static_capacity_shortfall_bytes": static_shortfall,
                "predicted_deadline_pass": predicted_pass,
                "actual_open_deadline_pass": bool(finish["deadline_met"]),
                "predicted_completion_corrected_ns": predicted_absolute_ns,
                "integrity_ok": True,
            }
        )
    return results


def replay_job(
    *,
    job_dir: Path,
    max_inflight_bytes: int,
    rate_bps: int | None,
) -> dict[str, Any]:
    selection = json.loads(
        (job_dir / "stage_service_selection.json").read_text(encoding="utf-8")
    )
    job_id = job_dir.name.rsplit("_", 1)[-1]
    selected_rate = int(selection["down_only_selection"]["selected_d2h_bps"])
    candidate_rate = selected_rate if rate_bps is None else rate_bps
    expected_world_size = int(selection["expected_world_size"])

    rank_results: list[dict[str, Any]] = []
    for rank in range(expected_world_size):
        rank_results.extend(
            _replay_rank(
                job_dir=job_dir,
                rank=rank,
                selection=selection,
                max_inflight_bytes=max_inflight_bytes,
                rate_bps=candidate_rate,
            )
        )

    groups: list[dict[str, Any]] = []
    for event_step in sorted({item["event_step"] for item in rank_results}):
        members = [item for item in rank_results if item["event_step"] == event_step]
        if len(members) != expected_world_size:
            raise ValueError(f"job {job_id} step {event_step}: incomplete rank group")
        completions = [
            int(item["predicted_completion_corrected_ns"]) for item in members
        ]
        groups.append(
            {
                "event_step": event_step,
                "rank_count": len(members),
                "predicted_group_durable_ms": max(
                    item["predicted_durable_ms"] for item in members
                ),
                "predicted_completion_skew_ms": round(
                    (max(completions) - min(completions)) / 1e6,
                    3,
                ),
                "predicted_deadline_pass": all(
                    item["predicted_deadline_pass"] for item in members
                ),
            }
        )

    return {
        "job_id": job_id,
        "split": "calibration" if job_id in CALIBRATION_JOBS else "held_out",
        "candidate": {
            "max_inflight_bytes": max_inflight_bytes,
            "rate_bytes_per_second": candidate_rate,
            "rate_source": "trace_calibration" if rate_bps is None else "cli",
        },
        "groups": groups,
        "ranks": rank_results,
    }


def replay(
    *,
    trace_root: Path,
    jobs: Iterable[str] = DEFAULT_JOBS,
    max_inflight_bytes: int = 1 << 20,
    rate_bps: int | None = None,
) -> dict[str, Any]:
    if max_inflight_bytes <= 0:
        raise ValueError("max_inflight_bytes must be positive")
    if rate_bps is not None and rate_bps <= 0:
        raise ValueError("rate_bps must be positive")

    job_results = [
        replay_job(
            job_dir=trace_root / f"staged_g1_job_{job_id}",
            max_inflight_bytes=max_inflight_bytes,
            rate_bps=rate_bps,
        )
        for job_id in jobs
    ]
    deadline_pass = all(
        group["predicted_deadline_pass"]
        for job in job_results
        for group in job["groups"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "d2h_rate_deadline": "identifiable_constant_rate_proxy",
            "max_inflight_performance_effect": "not_identifiable",
            "collective_tail_effect": "not_identifiable",
            "pfs_policy": "optimized_open_unchanged",
        },
        "jobs": job_results,
        "summary": {
            "trace_integrity": "pass",
            "predicted_deadline_pass_all": deadline_pass,
            "decision": "one_node_probe" if deadline_pass else "kill_c0",
            "reason": (
                "Rate feasibility passes; one short 1-node probe is needed to "
                "measure collective impact and the max-inflight effect."
                if deadline_pass
                else "The fixed-rate candidate misses the replayed deadline."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--jobs", nargs="+", default=list(DEFAULT_JOBS))
    parser.add_argument("--max-inflight-bytes", type=int, default=1 << 20)
    parser.add_argument("--rate-bps", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = replay(
        trace_root=args.trace_root,
        jobs=args.jobs,
        max_inflight_bytes=args.max_inflight_bytes,
        rate_bps=args.rate_bps,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
