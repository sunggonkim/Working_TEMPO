#!/usr/bin/env python3
"""Compose a live G1 causal result only from a validated raw artifact.

The tier runner intentionally emits structural evidence separately from the
domain metric/counter records.  This tool is the boundary between those two
artifacts: it validates the raw five-mode tree, checks the source/geometry
binding, loads a separately written metric sidecar, and then delegates the
causal decision to ``validate_g1_result.py``.  It never submits work and never
turns a design manifest into live evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

try:
    from eval.sota_4node.validate_g1_result import validate_g1_result
    from eval.sota_4node.validate_g1_tier_raw import validate_g1_tier_raw
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eval.sota_4node.validate_g1_result import validate_g1_result
    from eval.sota_4node.validate_g1_tier_raw import validate_g1_tier_raw


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _host_pressure_raw_digest(root: Path, manifest: dict[str, Any]) -> str:
    """Digest the validated raw placebo records, including rank-file names.

    The metric sidecar is written by a separate producer from the raw tier
    runner.  Binding only both artifacts to the same source bundle is not
    enough: a sidecar could otherwise self-attest a different host-NUMA
    counter series.  The canonical path+record projection makes the causal
    composition reproducible while remaining insensitive to JSON whitespace.
    """

    placebo = manifest.get("host_pressure_placebo")
    if type(placebo) is not dict or type(placebo.get("rank_files")) is not list:
        raise ValueError("G1 raw manifest host-pressure rank files are missing")
    records: list[dict[str, object]] = []
    for relative in placebo["rank_files"]:
        if type(relative) is not str:
            raise ValueError("G1 raw manifest host-pressure rank file is not a string")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("G1 host-pressure rank file escapes raw root") from exc
        if not path.is_file():
            raise ValueError(f"G1 host-pressure rank file is missing: {relative}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if type(record) is not dict:
            raise ValueError(f"G1 host-pressure rank file is not an object: {relative}")
        records.append({"path": relative, "record": record})
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_bundle(root: Path, manifest: dict[str, Any]) -> str:
    raw = manifest.get("source_sha256")
    base_keys = {
        "train_executed.py",
        "tier_attribution_runner_executed.py",
        "validate_g1_tier_raw_executed.py",
        "validate_g1_result_executed.py",
        "compose_g1_result_executed.py",
        "build_g1_causal_readiness_executed.py",
        "host_pressure_train_wrapper_executed.py",
        "host_pressure_placebo.py",
        "capture_g1_domain_counters_executed.py",
    }
    allowed_keys = (
        base_keys,
        base_keys | {"prepare_foreground_path_executed.py"},
        base_keys | {"capture_nvml_pcie_observation_executed.py"},
        base_keys | {"prepare_foreground_path_executed.py", "capture_nvml_pcie_observation_executed.py"},
        base_keys | {"capture_lustre_rpc_observation_executed.py"},
        base_keys | {"capture_nvml_pcie_observation_executed.py", "capture_lustre_rpc_observation_executed.py"},
        base_keys | {"prepare_foreground_path_executed.py", "capture_lustre_rpc_observation_executed.py"},
        base_keys | {"prepare_foreground_path_executed.py", "capture_nvml_pcie_observation_executed.py", "capture_lustre_rpc_observation_executed.py"},
    )
    if type(raw) is not dict or set(raw) not in allowed_keys:
        raise ValueError("G1 raw manifest source_sha256 keys are not exact")
    pieces: list[str] = []
    for name in sorted(raw):
        path = root / name
        if not path.is_file() or raw[name] != _sha256(path):
            raise ValueError(f"G1 raw source hash mismatch: {name}")
        pieces.append(f"{name}:{raw[name]}")
    return hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()


def _assert_analysis_snapshots(root: Path) -> None:
    """Refuse to compose with analysis code different from the allocation snapshot."""

    current = {
        "compose_g1_result_executed.py": Path(__file__).resolve(),
        "validate_g1_tier_raw_executed.py": Path(
            inspect.getsourcefile(validate_g1_tier_raw) or ""
        ).resolve(),
        "validate_g1_result_executed.py": Path(
            inspect.getsourcefile(validate_g1_result) or ""
        ).resolve(),
    }
    for name, path in current.items():
        snapshot = root / name
        if not path.is_file() or not snapshot.is_file() or _sha256(path) != _sha256(snapshot):
            raise ValueError(f"analysis snapshot mismatch: {name}")


def compose_g1_result(raw_root: Path, metrics_path: Path) -> dict[str, object]:
    """Return the validated causal evaluation plus raw-structure provenance."""

    raw_root = raw_root.resolve()
    metrics_path = metrics_path.resolve()
    raw_evaluation = validate_g1_tier_raw(raw_root)
    _assert_analysis_snapshots(raw_root)
    manifest_path = raw_root / "g1_tier_runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_bundle = _source_bundle(raw_root, manifest)
    host_pressure_raw_digest = _host_pressure_raw_digest(raw_root, manifest)
    if not metrics_path.is_file():
        raise ValueError(f"G1 metric sidecar is missing: {metrics_path}")
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    if type(result) is not dict:
        raise ValueError("G1 metric sidecar must be a JSON object")
    if result.get("source_bundle_sha256") != source_bundle:
        raise ValueError("G1 metric sidecar is not bound to the raw source bundle")
    if result.get("host_pressure_raw_digest") != host_pressure_raw_digest:
        raise ValueError("G1 metric sidecar is not bound to raw host-pressure records")
    for key in ("world_size", "nodes", "state_bytes_per_rank", "logical_file_extent_bytes", "checkpoint_steps"):
        if result.get(key) != manifest.get(key):
            raise ValueError(f"G1 metric sidecar disagrees with raw manifest: {key}")
    validated = validate_g1_result(result)
    return {
        "schema_version": "tempo-rd-g1-composed-evaluation-1",
        "raw_structure": raw_evaluation,
        "causal_evaluation": validated,
        "source_bundle_sha256": source_bundle,
        "host_pressure_raw_digest": host_pressure_raw_digest,
        "live_external_execution": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("metrics_sidecar", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compose_g1_result(args.raw_root, args.metrics_sidecar)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
