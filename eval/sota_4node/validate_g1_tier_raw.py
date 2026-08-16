#!/usr/bin/env python3
"""Fail-closed structural validator for the five-mode G1 raw artifact.

This validator intentionally does not compute a causal promotion.  It checks
that a completed one-node tier allocation produced the exact rank/step/source
evidence required by the later ``tempo-rd-g1-result-5`` composition step.
Missing hardware counters or missing path attribution therefore remain an
explicit ``promotion_ready=false`` outcome rather than being inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


MODES = ("fg_only", "open_combined", "d2h_only", "persist_only", "combined")
POLICY_BY_MODE = {
    "fg_only": "none",
    "open_combined": "datastates",
    "d2h_only": "datastates",
    "persist_only": "datastates",
    "combined": "datastates",
}
PERSISTENT_MODES = frozenset({"open_combined", "persist_only", "combined"})
STAGE_COUNTER_FIELDS = (
    "total_bytes",
    "queued_bytes",
    "ready_bytes",
    "admitted_bytes",
    "completed_bytes",
    "inflight_bytes",
    "inflight_requests",
    "admitted_requests",
    "max_request_bytes",
    "peak_inflight_bytes",
    "peak_inflight_requests",
    "last_progress_monotonic_ns",
    "last_completion_monotonic_ns",
)
MONOTONIC_STAGE_FIELDS = (
    "total_bytes",
    "admitted_bytes",
    "completed_bytes",
    "admitted_requests",
    "peak_inflight_bytes",
    "peak_inflight_requests",
)
RANK_RE = re.compile(r"_(?:rank|r)(\d+)\.(?:json|csv)$")
STEP_RE = re.compile(r"(?:step[-_])(\d+)")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc


def _exact_rank_files(directory: Path, pattern: str, *, required: bool = True) -> dict[int, Path]:
    paths = sorted(directory.glob(pattern))
    result: dict[int, Path] = {}
    for path in paths:
        match = RANK_RE.search(path.name)
        if match is None:
            raise ValueError(f"{directory.name}: unparseable rank file {path.name}")
        rank = int(match.group(1))
        if rank in result:
            raise ValueError(f"{directory.name}: duplicate rank {rank}")
        result[rank] = path
    if required and set(result) != {0, 1, 2, 3}:
        raise ValueError(f"{directory.name}: expected exact rank files 0..3")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_csv_rows(path: Path, label: str) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{label}: cannot read {path}") from exc
    if len(lines) < 2 or not lines[0].strip():
        raise ValueError(f"{label}: CSV has no data rows")


def _validate_stage_snapshot(raw: object, label: str) -> dict[str, object]:
    if type(raw) is not dict:
        raise ValueError(f"{label}: stage snapshot is not an object")
    for field in STAGE_COUNTER_FIELDS:
        value = raw.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"{label}: {field} must be a non-negative integer")
    return raw


def _validate_stage_pair(start: object, end: object, label: str) -> None:
    start_map = _validate_stage_snapshot(start, f"{label}/start")
    end_map = _validate_stage_snapshot(end, f"{label}/end")
    for field in MONOTONIC_STAGE_FIELDS:
        if end_map[field] < start_map[field]:
            raise ValueError(f"{label}: {field} regresses across the event")


def _validate_logical_stage(raw: object, label: str) -> None:
    """Validate optional native-path logical stage timing evidence.

    Native DataStates may expose zero TEMPO admission counters while still
    moving exact state/file bytes.  This record is useful for a stage ledger,
    but it must never self-attest as a PCIe/NIC/OST hardware counter.
    """

    if raw is None:
        return
    if type(raw) is not dict or set(raw) != {
        "schema", "counter_semantics", "hardware_counter", "d2h", "pfs"
    }:
        raise ValueError(f"{label}: logical stage schema is not exact")
    if raw["schema"] != "tempo-rd-logical-stage-timing-1":
        raise ValueError(f"{label}: unsupported logical stage schema")
    if raw["counter_semantics"] != "logical_bytes_and_wait_interval":
        raise ValueError(f"{label}: logical stage semantics are not explicit")
    if raw["hardware_counter"] is not False:
        raise ValueError(f"{label}: logical stage must not self-attest hardware counters")
    for stage_name in ("d2h", "pfs"):
        stage = raw[stage_name]
        _validate_stage_snapshot(stage, f"{label}/{stage_name}")
        busy_ns = stage.get("busy_ns")
        if type(busy_ns) is not int or busy_ns < 0:
            raise ValueError(f"{label}/{stage_name}: busy_ns must be non-negative")


def _validate_engine_snapshot_pair(
    start: object, end: object, label: str, *, persistent: bool
) -> None:
    if type(start) is not dict or type(end) is not dict:
        raise ValueError(f"{label}: engine snapshots must be objects")
    for stage in ("d2h", "pfs"):
        _validate_stage_pair(start.get(stage), end.get(stage), f"{label}/{stage}")
    _validate_logical_stage(start.get("logical_stage"), f"{label}/logical_stage/start")
    _validate_logical_stage(end.get("logical_stage"), f"{label}/logical_stage/end")
    if persistent:
        # Persistent attribution must prove the actual O_DIRECT/fsync endpoint,
        # rather than merely reporting a PFS-shaped dictionary.
        if end.get("pfs_odirect_verified") is not True:
            raise ValueError(f"{label}: pfs O_DIRECT evidence is absent")
        if end.get("pfs_fsync_complete") is not True:
            raise ValueError(f"{label}: fresh pfs fsync evidence is absent")
        start_fsync = start.get("pfs_fsync_monotonic_ns")
        end_fsync = end.get("pfs_fsync_monotonic_ns")
        if (
            type(start_fsync) is not int
            or type(end_fsync) is not int
            or start_fsync < 0
            or end_fsync <= start_fsync
        ):
            raise ValueError(f"{label}: pfs fsync timestamp is not fresh")


def _validate_host_pressure_record(path: Path, rank: int, manifest: dict[str, object]) -> None:
    record = _load_json(path)
    if type(record) is not dict or record.get("schema_version") != "tempo-rd-host-pressure-run-1":
        raise ValueError(f"fg_only/host-pressure rank {rank}: schema mismatch")
    spec = record.get("spec")
    if type(spec) is not dict or spec.get("rank") != rank or spec.get("world_size") != 4:
        raise ValueError(f"fg_only/host-pressure rank {rank}: rank/world mismatch")
    expected = manifest["host_pressure_placebo"]
    if spec.get("buffer_bytes") != expected["buffer_bytes"] or spec.get("source") != expected["source"]:
        raise ValueError(f"fg_only/host-pressure rank {rank}: spec mismatch")
    samples = record.get("samples")
    if type(samples) is not list or len(samples) < 2:
        raise ValueError(f"fg_only/host-pressure rank {rank}: insufficient samples")
    previous = None
    for sample in samples:
        if type(sample) is not dict or set(sample) != {
            "sample_id", "timestamp_ns", "cumulative_touched_bytes",
            "cumulative_busy_ns", "numa_node_bytes",
        }:
            raise ValueError(f"fg_only/host-pressure rank {rank}: sample schema mismatch")
        if previous is not None:
            if sample["timestamp_ns"] <= previous["timestamp_ns"]:
                raise ValueError(f"fg_only/host-pressure rank {rank}: timestamp regressed")
            for field in ("cumulative_touched_bytes", "cumulative_busy_ns", "numa_node_bytes"):
                if sample[field] < previous[field]:
                    raise ValueError(f"fg_only/host-pressure rank {rank}: {field} regressed")
        previous = sample
    if previous["cumulative_touched_bytes"] < expected["buffer_bytes"]:
        raise ValueError(f"fg_only/host-pressure rank {rank}: declared buffer was not touched")
    if previous["cumulative_busy_ns"] <= 0:
        raise ValueError(f"fg_only/host-pressure rank {rank}: busy interval is empty")
    digest = record.get("output_sha256")
    if type(digest) is not str or len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise ValueError(f"fg_only/host-pressure rank {rank}: output digest is invalid")
    unsigned = dict(record)
    unsigned["output_sha256"] = ""
    expected_digest = hashlib.sha256(
        (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    if digest != expected_digest:
        raise ValueError(f"fg_only/host-pressure rank {rank}: output digest mismatch")


def _validate_checkpoint_metrics(
    record: object,
    label: str,
    *,
    manifest: dict[str, object],
    persistent: bool,
) -> dict[str, object]:
    """Bind the saved metrics to the frozen G1 payload/commit contract.

    Stage snapshots alone do not prove that the measured mode moved the
    intended state.  The checkpoint record must therefore carry the exact
    per-rank state size and, for persistent modes, matching logical/physical
    extents plus fresh fsync/global-commit evidence.
    """

    if type(record) is not dict:
        raise ValueError(f"{label}: checkpoint metrics are not an object")
    state_bytes = record.get("state_bytes_local")
    expected_state = manifest["state_bytes_per_rank"]
    if type(state_bytes) is not int or state_bytes != expected_state:
        raise ValueError(
            f"{label}: state_bytes_local does not match the frozen per-rank state"
        )
    checkpoint_path = record.get("checkpoint_path")
    if type(checkpoint_path) is not str or not checkpoint_path:
        raise ValueError(f"{label}: checkpoint_path is missing")
    durable_ms = record.get("durable_ms")
    if (
        type(durable_ms) not in (int, float)
        or isinstance(durable_ms, bool)
        or not math.isfinite(float(durable_ms))
        or durable_ms < 0
    ):
        raise ValueError(f"{label}: durable_ms is not a finite non-negative number")
    if type(record.get("deadline_met")) is not bool:
        raise ValueError(f"{label}: deadline_met must be a boolean")

    if persistent:
        logical = record.get("logical_file_extent_bytes")
        physical = record.get("checkpoint_file_bytes")
        allocated = record.get("checkpoint_allocated_bytes")
        expected_extent = manifest["logical_file_extent_bytes"]
        if type(logical) is not int or logical != expected_extent:
            raise ValueError(f"{label}: logical extent does not match the G1 manifest")
        if type(physical) is not int or physical != logical:
            raise ValueError(f"{label}: physical checkpoint extent mismatches logical extent")
        if type(allocated) is not int or allocated < physical:
            raise ValueError(f"{label}: allocated checkpoint bytes are insufficient")
        if record.get("commit_validated") is not True:
            raise ValueError(f"{label}: global commit evidence is absent")
        if record.get("fsync_evidence_valid") is not True:
            raise ValueError(f"{label}: fsync evidence is absent")
        marker = record.get("commit_marker_path")
        digest = record.get("commit_manifest_sha256")
        if type(marker) is not str or not marker:
            raise ValueError(f"{label}: commit marker path is missing")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"{label}: commit manifest digest is not lowercase SHA-256")
    else:
        for field in (
            "checkpoint_file_bytes",
            "checkpoint_allocated_bytes",
            "logical_file_extent_bytes",
        ):
            value = record.get(field, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"{label}: {field} must be a non-negative integer")
        for field in ("commit_validated", "fsync_evidence_valid"):
            if field in record and type(record[field]) is not bool:
                raise ValueError(f"{label}: {field} must be a boolean")
    return record


def validate_g1_tier_raw(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest_path = root / "g1_tier_runtime_manifest.json"
    plan_path = root / "g1_command_plan.json"
    status_path = root / "execution_status.env"
    placement_path = root / "placement.txt"
    if not manifest_path.is_file() or not plan_path.is_file() or not status_path.is_file():
        raise ValueError("G1 tier runtime manifest, command plan, or execution status is missing")
    manifest = _load_json(manifest_path)
    if type(manifest) is not dict or manifest.get("schema_version") != "tempo-rd-g1-tier-runtime-1":
        raise ValueError("unsupported G1 tier runtime manifest")
    if manifest.get("stage") != "g1_tier" or manifest.get("nodes") != 1 or manifest.get("world_size") != 4:
        raise ValueError("G1 tier geometry is not one node/four ranks")
    if manifest.get("slurm_submitted") is not True or manifest.get("no_retry") is not True:
        raise ValueError("G1 tier raw artifact must record one submitted, no-retry allocation")
    if manifest.get("modes") != list(MODES):
        raise ValueError("G1 tier mode order is not exact")
    steps = manifest.get("checkpoint_steps")
    if type(steps) is not list or not steps or any(type(step) is not int for step in steps):
        raise ValueError("G1 checkpoint steps must be an integer list")
    if steps != sorted(set(steps)):
        raise ValueError("G1 checkpoint steps must be sorted and unique")
    if manifest.get("steps") != 72 or manifest.get("warmup_steps") != 12:
        raise ValueError("G1 runtime step schedule is not the frozen 72/12 contract")
    source_sha = manifest.get("source_sha256")
    legacy_contract = not isinstance(source_sha, dict) or "build_g1_causal_readiness_executed.py" not in source_sha
    expected_caps = (60, 30, 390, 150) if legacy_contract else (35, 20, 235, 65)
    if (manifest.get("mode_outer_seconds"), manifest.get("restore_outer_seconds")) != expected_caps[:2]:
        raise ValueError("G1 per-mode timeout caps are not the frozen contract")
    if (manifest.get("phase_budget_seconds"), manifest.get("cleanup_reserve_seconds")) != expected_caps[2:]:
        raise ValueError("G1 phase budget does not leave the required cleanup reserve")
    for name in ("state_bytes_per_rank", "logical_file_extent_bytes"):
        value = manifest.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"G1 {name} must be a positive integer")
    if manifest["logical_file_extent_bytes"] < manifest["state_bytes_per_rank"]:
        raise ValueError("G1 logical extent is smaller than state bytes")
    placebo = manifest.get("host_pressure_placebo")
    if type(placebo) is not dict or set(placebo) != {"mode", "buffer_bytes", "source", "rank_files"}:
        raise ValueError("G1 host-pressure placebo contract is missing")
    if placebo["mode"] != "fg_only" or placebo["buffer_bytes"] != 64 * 1024 * 1024:
        raise ValueError("G1 host-pressure placebo geometry is not frozen")
    if placebo["source"] != "proc_self_numa_maps_plus_touch_loop":
        raise ValueError("G1 host-pressure placebo source is not explicit")
    if placebo["rank_files"] != [f"fg_only/host_pressure_rank_{rank}.json" for rank in range(4)]:
        raise ValueError("G1 host-pressure placebo rank file set is not exact")
    geometry = manifest.get("geometry")
    if type(geometry) is not dict or set(geometry) != {
        "layers", "hidden_size", "ffn_size", "heads", "sequence_length", "batch_size"
    }:
        raise ValueError("G1 geometry record is not exact")
    if geometry != {
        "layers": 2,
        "hidden_size": 2048,
        "ffn_size": 8192,
        "heads": 16,
        "sequence_length": 64,
        "batch_size": 1,
    }:
        raise ValueError("G1 geometry does not match the frozen tier-attribution contract")
    sources = manifest.get("source_sha256")
    source_keys = {
        "train_executed.py",
        "tier_attribution_runner_executed.py",
        "validate_g1_tier_raw_executed.py",
        "validate_g1_result_executed.py",
        "compose_g1_result_executed.py",
        "host_pressure_train_wrapper_executed.py",
        "host_pressure_placebo.py",
    }
    extended_source_keys = source_keys | {
        "build_g1_causal_readiness_executed.py",
        "capture_g1_domain_counters_executed.py",
    }
    foreground_source_keys = extended_source_keys | {
        "prepare_foreground_path_executed.py",
    }
    # The NVML/PCIe probe is an optional foreground-path extension.  New
    # allocations bind it in the runtime manifest; older artifacts legitimately
    # omit it, so keep both exact contracts rather than silently ignoring an
    # executed source file.
    pcie_source_keys = extended_source_keys | {
        "capture_nvml_pcie_observation_executed.py",
    }
    foreground_pcie_source_keys = foreground_source_keys | {
        "capture_nvml_pcie_observation_executed.py",
    }
    lustre_source_keys = extended_source_keys | {
        "capture_lustre_rpc_observation_executed.py",
    }
    pcie_lustre_source_keys = pcie_source_keys | {
        "capture_lustre_rpc_observation_executed.py",
    }
    foreground_lustre_source_keys = foreground_source_keys | {
        "capture_lustre_rpc_observation_executed.py",
    }
    foreground_pcie_lustre_source_keys = foreground_pcie_source_keys | {
        "capture_lustre_rpc_observation_executed.py",
    }
    if type(sources) is not dict or set(sources) not in (
        source_keys,
        extended_source_keys,
        foreground_source_keys,
        pcie_source_keys,
        foreground_pcie_source_keys,
        lustre_source_keys,
        pcie_lustre_source_keys,
        foreground_lustre_source_keys,
        foreground_pcie_lustre_source_keys,
    ):
        raise ValueError("G1 source hash record is not exact")
    for name, expected in sources.items():
        path = root / name
        if not path.is_file() or type(expected) is not str or len(expected) != 64 or _sha256(path) != expected:
            raise ValueError(f"G1 source hash mismatch: {name}")
    plan = _load_json(plan_path)
    if type(plan) is not dict or plan.get("schema_version") != "tempo-rd-g1-command-plan-1":
        raise ValueError("G1 command plan schema mismatch")
    if plan.get("submitting") is not False or plan.get("world_size") != 4:
        raise ValueError("G1 command plan must remain non-submitting and world-size four")
    commands = plan.get("commands")
    if type(commands) is not list or [item.get("mode") for item in commands] != list(MODES):
        raise ValueError("G1 command plan mode order is not exact")
    if not placement_path.is_file():
        raise ValueError("G1 placement evidence is missing")
    placement_lines = [line for line in placement_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # srun may interleave stdout from the four ranks.  Validate the complete
    # rank/local-rank set, not the incidental line order, so a valid allocation
    # is not rejected merely because output buffering reordered these lines.
    placement_records: dict[int, int] = {}
    for line in placement_lines:
        rank_match = re.search(r"\brank=(\d+)\b", line)
        local_match = re.search(r"\blocal_rank=(\d+)\b", line)
        if rank_match is None or local_match is None:
            raise ValueError("G1 placement line lacks rank/local_rank evidence")
        rank = int(rank_match.group(1))
        local_rank = int(local_match.group(1))
        if rank in placement_records:
            raise ValueError(f"G1 placement evidence duplicates rank {rank}")
        if local_rank != rank:
            raise ValueError(f"G1 placement rank/local-rank mismatch for rank {rank}")
        placement_records[rank] = local_rank
    if set(placement_records) != {0, 1, 2, 3}:
        raise ValueError("G1 placement evidence does not contain exact ranks 0..3")

    mode_summary: dict[str, dict[str, object]] = {}
    for mode in MODES:
        mode_dir = root / mode
        if not mode_dir.is_dir():
            raise ValueError(f"G1 mode directory missing: {mode}")
        summary_files = _exact_rank_files(mode_dir, "summary_rank*.json")
        collective_files = _exact_rank_files(mode_dir, "collectives_rank*.csv")
        step_files = _exact_rank_files(mode_dir, "steps_rank*.csv")
        for rank in range(4):
            summary = _load_json(summary_files[rank])
            if type(summary) is not dict:
                raise ValueError(f"{mode}/summary rank {rank}: not an object")
            if summary.get("rank") != rank or summary.get("world_size") != 4:
                raise ValueError(f"{mode}/summary rank {rank}: rank/world mismatch")
            if summary.get("policy") != POLICY_BY_MODE[mode] or summary.get("tier_mode") != mode:
                raise ValueError(f"{mode}/summary rank {rank}: policy/tier mismatch")
            if summary.get("source_sha256") != sources["train_executed.py"]:
                raise ValueError(f"{mode}/summary rank {rank}: train source mismatch")
            expected_endpoint = {
                "fg_only": "",
                "d2h_only": "node_local_sink",
                "open_combined": "persistent_endpoint",
                "persist_only": "persistent_endpoint",
                "combined": "persistent_endpoint",
            }[mode]
            if summary.get("tier_endpoint") != expected_endpoint:
                raise ValueError(f"{mode}/summary rank {rank}: endpoint mismatch")
            expected_preloaded = mode == "persist_only"
            expected_gpu_transfer = mode in {"d2h_only", "open_combined", "combined"}
            if summary.get("tier_host_preloaded") is not expected_preloaded:
                raise ValueError(f"{mode}/summary rank {rank}: host-preload marker mismatch")
            if summary.get("tier_gpu_transfer") is not expected_gpu_transfer:
                raise ValueError(f"{mode}/summary rank {rank}: GPU-transfer marker mismatch")
            _require_csv_rows(collective_files[rank], f"{mode}/collectives rank {rank}")
            _require_csv_rows(step_files[rank], f"{mode}/steps rank {rank}")
        if mode == "fg_only":
            for rank in range(4):
                _validate_host_pressure_record(
                    root / f"fg_only/host_pressure_rank_{rank}.json", rank, manifest
                )
        if mode != "fg_only":
            event_files = _exact_rank_files(mode_dir, "checkpoint_events_rank*.json")
            checkpoint_files = _exact_rank_files(mode_dir, "checkpoint_rank*.json")
            for rank in range(4):
                events = _load_json(event_files[rank])
                metrics = _load_json(checkpoint_files[rank])
                if type(events) is not list or len(events) != len(steps):
                    raise ValueError(f"{mode}/rank {rank}: checkpoint event count mismatch")
                if type(metrics) is not dict:
                    raise ValueError(f"{mode}/rank {rank}: checkpoint metrics are not an object")
                _validate_engine_snapshot_pair(
                    metrics.get("tier_stage_stats_start"),
                    metrics.get("tier_stage_stats_end"),
                    f"{mode}/rank {rank}",
                    persistent=mode in PERSISTENT_MODES,
                )
                _validate_checkpoint_metrics(
                    metrics,
                    f"{mode}/rank {rank}/final metrics",
                    manifest=manifest,
                    persistent=mode in PERSISTENT_MODES,
                )
                event_steps = []
                for event in events:
                    _validate_checkpoint_metrics(
                        event,
                        f"{mode}/rank {rank}/event",
                        manifest=manifest,
                        persistent=mode in PERSISTENT_MODES,
                    )
                    match = STEP_RE.search(str(event.get("checkpoint_path", "")))
                    if match is None:
                        raise ValueError(f"{mode}/rank {rank}: event path has no step")
                    event_steps.append(int(match.group(1)))
                if event_steps != steps:
                    raise ValueError(f"{mode}/rank {rank}: event step order mismatch")
            if mode in PERSISTENT_MODES:
                restore = _exact_rank_files(mode_dir, "fresh_restore_rank*.json", required=False)
                if set(restore) != {0, 1, 2, 3}:
                    restore = _exact_rank_files(mode_dir, "restore_rank*.json", required=False)
                if set(restore) != {0, 1, 2, 3}:
                    raise ValueError(f"{mode}: persistent mode lacks exact four-rank restore evidence")
                for rank, path in restore.items():
                    record = _load_json(path)
                    if type(record) is not dict or record.get("passed") is not True:
                        raise ValueError(f"{mode}/restore rank {rank}: restore did not pass")
                    if "rank" in record and record.get("rank") != rank:
                        raise ValueError(f"{mode}/restore rank {rank}: embedded rank mismatches filename")
        mode_summary[mode] = {"ranks": 4, "checkpoint_events": len(steps) if mode != "fg_only" else 0}

    status_text = status_path.read_text(encoding="utf-8")
    if "status=raw_complete" not in status_text:
        raise ValueError("G1 execution status is not raw_complete")
    return {
        "schema_version": "tempo-rd-g1-tier-raw-evaluation-1",
        "status": "pass",
        "live_external_execution": True,
        "promotion_ready": False,
        "reason": "raw structure is complete; compose metrics/path counters and run validate_g1_result.py",
        "modes": mode_summary,
        "checkpoint_steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_g1_tier_raw(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
