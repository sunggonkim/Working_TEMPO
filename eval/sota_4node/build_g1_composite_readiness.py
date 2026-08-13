#!/usr/bin/env python3
"""Build a conservative G1 composite-stage readiness record.

Perlmutter does not expose all GPU-local/PCIe/NUMA/Lustre byte counters used by
the fine-grained tier contract.  The tier runner nevertheless records exact
logical bytes and active service intervals for its isolated D2H and persistent
stages.  This tool makes that narrower evidence machine-readable without
renaming it as a hardware counter or promoting a topology claim.

The output is intentionally ``observed_composite`` rather than ``ready``:
it can authorize a *fabric follow-up* only when the raw five-mode matrix and
the isolated stage intervention are complete, but it can never authorize a
fine-domain scheduler or a SOTA claim by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "tempo-rd-g1-composite-readiness-1"
MODES = ("fg_only", "open_combined", "d2h_only", "persist_only", "combined")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive int")
    return value


def _logical_stage(root: Path, mode: str, rank: int) -> dict[str, Any]:
    events_path = root / mode / f"checkpoint_events_rank{rank}.json"
    events = _load(events_path)
    if not isinstance(events, list) or not events:
        raise ValueError(f"{mode}/checkpoint_events_rank{rank}.json is empty")
    start = events[0].get("tier_stage_stats_start")
    end = events[-1].get("tier_stage_stats_end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise ValueError(f"{mode}/rank{rank}: stage snapshots are missing")
    start_logical = start.get("logical_stage")
    end_logical = end.get("logical_stage")
    if not isinstance(start_logical, dict) or not isinstance(end_logical, dict):
        raise ValueError(f"{mode}/rank{rank}: logical stage snapshots are missing")
    result: dict[str, Any] = {}
    for stage in ("d2h", "pfs"):
        before = start_logical.get(stage)
        after = end_logical.get(stage)
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(f"{mode}/rank{rank}: {stage} logical stage is missing")
        bytes_before = int(before.get("completed_bytes", 0))
        bytes_after = int(after.get("completed_bytes", 0))
        busy_before = int(before.get("busy_ns", 0))
        busy_after = int(after.get("busy_ns", 0))
        if min(bytes_before, bytes_after, busy_before, busy_after) < 0:
            raise ValueError(f"{mode}/rank{rank}: negative {stage} logical counter")
        if bytes_after < bytes_before or busy_after < busy_before:
            raise ValueError(f"{mode}/rank{rank}: {stage} logical counter regressed")
        result[stage] = {
            "bytes": bytes_after - bytes_before,
            "busy_ns": busy_after - busy_before,
            "requests": max(0, int(after.get("completed_requests", 0)) - int(before.get("completed_requests", 0))),
            "hardware_counter": False,
            "counter_semantics": "logical_bytes_and_wait_interval",
        }
    return result


def _mode_metric(root: Path, mode: str) -> dict[str, float]:
    step: list[float] = []
    window: list[float] = []
    for rank in range(4):
        summary = _load(root / mode / f"summary_rank{rank}.json")
        for key, target in (("step_p99_ms", step), ("window_step_p99_ms", window)):
            value = summary.get(key)
            if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{mode}/rank{rank}: invalid {key}")
            target.append(float(value))
    return {"rank_max_step_p99_ms": max(step), "rank_max_window_p99_ms": max(window)}


def build_composite_readiness(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _load(root / "g1_tier_runtime_manifest.json")
    if manifest.get("world_size") != 4 or manifest.get("nodes") != 1:
        raise ValueError("composite G1 requires one node and four ranks")
    if manifest.get("modes") != list(MODES):
        raise ValueError("G1 mode order is not exact")
    mode_metrics = {mode: _mode_metric(root, mode) for mode in MODES}
    stage_by_mode: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        if mode == "fg_only":
            stage_by_mode[mode] = {"d2h": {"bytes": 0, "busy_ns": 0, "requests": 0}, "pfs": {"bytes": 0, "busy_ns": 0, "requests": 0}}
            continue
        rank_records = [_logical_stage(root, mode, rank) for rank in range(4)]
        stage_by_mode[mode] = {}
        for stage in ("d2h", "pfs"):
            values = [record[stage] for record in rank_records]
            stage_by_mode[mode][stage] = {
                "group_min_bytes": min(item["bytes"] for item in values),
                "group_min_busy_ns": min(item["busy_ns"] for item in values),
                "group_max_bytes": max(item["bytes"] for item in values),
                "group_max_busy_ns": max(item["busy_ns"] for item in values),
                "group_min_requests": min(item["requests"] for item in values),
                "hardware_counter": False,
                "counter_semantics": "logical_bytes_and_wait_interval",
            }
    d2h = stage_by_mode["d2h_only"]["d2h"]
    pfs = stage_by_mode["persist_only"]["pfs"]
    if d2h["group_min_bytes"] <= 0 or d2h["group_min_busy_ns"] <= 0:
        raise ValueError("d2h-only intervention did not expose a positive logical stage")
    if pfs["group_min_bytes"] <= 0 or pfs["group_min_busy_ns"] <= 0:
        raise ValueError("persist-only intervention did not expose a positive logical stage")
    if stage_by_mode["d2h_only"]["pfs"]["group_max_bytes"] != 0:
        raise ValueError("d2h-only unexpectedly used persistent bytes")
    if stage_by_mode["persist_only"]["d2h"]["group_max_bytes"] != 0:
        raise ValueError("persist-only unexpectedly used D2H bytes")
    fg = mode_metrics["fg_only"]
    signals = {
        "d2h_composite": {
            "intervention_mode": "d2h_only",
            "tail_delta_ms": mode_metrics["d2h_only"]["rank_max_step_p99_ms"] - fg["rank_max_step_p99_ms"],
            "window_delta_ms": mode_metrics["d2h_only"]["rank_max_window_p99_ms"] - fg["rank_max_window_p99_ms"],
            "stage_bytes": d2h["group_min_bytes"],
            "stage_busy_ns": d2h["group_min_busy_ns"],
        },
        "persistent_composite": {
            "intervention_mode": "persist_only",
            "tail_delta_ms": mode_metrics["persist_only"]["rank_max_step_p99_ms"] - fg["rank_max_step_p99_ms"],
            "window_delta_ms": mode_metrics["persist_only"]["rank_max_window_p99_ms"] - fg["rank_max_window_p99_ms"],
            "stage_bytes": pfs["group_min_bytes"],
            "stage_busy_ns": pfs["group_min_busy_ns"],
        },
    }
    source_files = [
        "g1_tier_runtime_manifest.json",
        "train_executed.py",
        "tier_attribution_runner_executed.py",
        "validate_g1_tier_raw_executed.py",
    ]
    return {
        "schema_version": SCHEMA,
        "status": "observed_composite",
        "promotion_ready": False,
        "fabric_followup_eligible": True,
        "fine_domain_promotion": False,
        "causal_scope": "composite_stage_intervention_only",
        "hardware_counter_claim": False,
        "world_size": 4,
        "nodes": 1,
        "modes": list(MODES),
        "mode_metrics": mode_metrics,
        "logical_stage_by_mode": stage_by_mode,
        "signals": signals,
        "missing_fine_domains": ["gpu_local", "pcie_host", "host_numa", "nic_fabric", "persistent_endpoint"],
        "deferred_domains": ["slingshot_fabric"],
        "source_sha256": {name: _sha256(root / name) for name in source_files},
        "reasons": [
            "logical stage bytes/active intervals are observed for isolated interventions",
            "hardware counter families remain unavailable or unbound",
            "this record cannot promote a fine-grained topology or scheduler claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_composite_readiness(args.root), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
