#!/usr/bin/env python3
"""Manifest and result parser for the official Mooncake TE microbenchmark."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


EXPECTED_WHEEL_VERSION = "0.3.12.post1"
OFFICIAL_REPOSITORY = "https://github.com/kvcache-ai/Mooncake.git"
MANIFEST_SCHEMA = "tempo.mooncake-official-manifest.v1"
RESULT_SCHEMA = "tempo.mooncake-official-result.v1"
SCOPE = "official-transfer-engine-only"
TRANSPORT = "tcp"
MEASUREMENT_SCOPE = "aggregate-throughput-only"

_RATE_UNIT_BYTES_PER_SECOND = {
    "GB": 1_000_000_000.0,
    "GiB": float(1 << 30),
    "Gb": 1_000_000_000.0 / 8.0,
    "MB": 1_000_000.0,
    "MiB": float(1 << 20),
    "Mb": 1_000_000.0 / 8.0,
    "KB": 1_000.0,
    "KiB": float(1 << 10),
    "Kb": 1_000.0 / 8.0,
}

_COMPLETION_RE = re.compile(
    r"Test completed:\s*duration\s+"
    r"(?P<duration>[0-9]+(?:\.[0-9]+)?),\s*batch count\s+"
    r"(?P<batch_count>[0-9]+),\s*throughput\s+"
    r"(?P<throughput>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<rate_unit>GB|GiB|Gb|MB|MiB|Mb|KB|KiB|Kb)/s"
)


class MooncakeParseError(ValueError):
    """Raised when official benchmark output or provenance is incomplete."""


def parse_benchmark_output(text: str) -> dict[str, Any]:
    """Parse the final aggregate-throughput line emitted by Mooncake TE."""

    matches = list(_COMPLETION_RE.finditer(text))
    if not matches:
        raise MooncakeParseError("Mooncake completion/throughput line not found")

    match = matches[-1]
    duration = float(match.group("duration"))
    batch_count = int(match.group("batch_count"))
    throughput = float(match.group("throughput"))
    rate_unit = match.group("rate_unit")
    if not math.isfinite(duration) or duration <= 0:
        raise MooncakeParseError("Mooncake reported a non-positive duration")
    if batch_count <= 0:
        raise MooncakeParseError("Mooncake completed no transfer batches")
    if not math.isfinite(throughput) or throughput <= 0:
        raise MooncakeParseError("Mooncake reported non-positive throughput")

    return {
        "duration_seconds": duration,
        "batch_count": batch_count,
        "throughput": {
            "value": throughput,
            "unit": f"{rate_unit}/s",
            "bytes_per_second": throughput
            * _RATE_UNIT_BYTES_PER_SECOND[rate_unit],
        },
    }


def make_manifest(
    *,
    wheel_version: str,
    git_commit: str,
    git_repository: str,
    binary: str,
    job_id: str,
    node_list: str,
) -> dict[str, Any]:
    """Create the fixed two-node official-code experiment contract."""

    if wheel_version != EXPECTED_WHEEL_VERSION:
        raise MooncakeParseError(
            f"expected Mooncake wheel {EXPECTED_WHEEL_VERSION}, got {wheel_version}"
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_commit):
        raise MooncakeParseError("Mooncake source commit must be a full 40-hex SHA")
    if git_repository != OFFICIAL_REPOSITORY:
        raise MooncakeParseError(
            f"expected official repository {OFFICIAL_REPOSITORY}, got {git_repository}"
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "baseline": "Mooncake Transfer Engine",
        "scope": SCOPE,
        "transport": TRANSPORT,
        "measurement_scope": MEASUREMENT_SCOPE,
        "artifact": {
            "distribution": "mooncake-transfer-engine",
            "wheel_version": wheel_version,
            "binary": str(Path(binary).resolve()),
            "source_repository": git_repository,
            "source_git_commit": git_commit.lower(),
        },
        "allocation": {
            "job_id": job_id,
            "node_list": node_list,
            "nodes": 2,
            "tasks": 2,
            "gpus_per_node": 4,
        },
        "workload": {
            "roles": {"target_rank": 0, "initiator_rank": 1},
            "metadata_server": "P2PHANDSHAKE",
            "backend": "classic",
            "operation": "read",
            "memory": "vram",
            "gpu_aware": True,
            "gpu_id": -1,
            "block_size_bytes": 33_554_432,
            "batch_size": 1,
            "threads": 4,
            "duration_seconds": 5,
            "buffer_size_bytes_per_gpu": 134_217_728,
            "report_unit": "GB",
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject labels that could overstate what this baseline measures."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise MooncakeParseError("unexpected Mooncake manifest schema")
    if manifest.get("scope") != SCOPE:
        raise MooncakeParseError(f"scope must be {SCOPE}")
    if manifest.get("transport") != TRANSPORT:
        raise MooncakeParseError(f"transport must be {TRANSPORT}")
    if manifest.get("measurement_scope") != MEASUREMENT_SCOPE:
        raise MooncakeParseError(
            f"measurement_scope must be {MEASUREMENT_SCOPE}"
        )

    artifact = manifest.get("artifact", {})
    if artifact.get("wheel_version") != EXPECTED_WHEEL_VERSION:
        raise MooncakeParseError("manifest does not identify the pinned wheel")
    if artifact.get("source_repository") != OFFICIAL_REPOSITORY:
        raise MooncakeParseError("manifest does not identify the official repository")
    if not re.fullmatch(
        r"[0-9a-f]{40}", str(artifact.get("source_git_commit", ""))
    ):
        raise MooncakeParseError("manifest source commit is not exact")

    workload = manifest.get("workload", {})
    expected = {
        "metadata_server": "P2PHANDSHAKE",
        "backend": "classic",
        "memory": "vram",
        "gpu_aware": True,
        "gpu_id": -1,
        "block_size_bytes": 33_554_432,
        "batch_size": 1,
        "threads": 4,
        "duration_seconds": 5,
    }
    for key, value in expected.items():
        if workload.get(key) != value:
            raise MooncakeParseError(f"unexpected workload value for {key}")


def build_result(
    manifest: dict[str, Any],
    initiator_logs: Sequence[Path],
    target_logs: Sequence[Path],
) -> dict[str, Any]:
    """Build a throughput-only result from explicitly named rank logs."""

    validate_manifest(manifest)
    if not initiator_logs:
        raise MooncakeParseError("at least one initiator log is required")
    initiator_text = "\n".join(path.read_text(errors="replace") for path in initiator_logs)
    for path in target_logs:
        path.read_text(errors="replace")

    return {
        "schema_version": RESULT_SCHEMA,
        "baseline": manifest["baseline"],
        "scope": SCOPE,
        "transport": TRANSPORT,
        "measurement_scope": MEASUREMENT_SCOPE,
        "manifest": manifest,
        "measurement": parse_benchmark_output(initiator_text),
        "logs": {
            "initiator": [str(path.resolve()) for path in initiator_logs],
            "target": [str(path.resolve()) for path in target_logs],
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise MooncakeParseError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="write the run manifest")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--wheel-version", required=True)
    manifest.add_argument("--git-commit", required=True)
    manifest.add_argument("--git-repository", required=True)
    manifest.add_argument("--binary", required=True)
    manifest.add_argument("--job-id", required=True)
    manifest.add_argument("--node-list", required=True)

    result = commands.add_parser("result", help="parse aggregate throughput")
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--initiator-log", type=Path, action="append", required=True)
    result.add_argument("--target-log", type=Path, action="append", default=[])
    result.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.command == "manifest":
        value = make_manifest(
            wheel_version=args.wheel_version,
            git_commit=args.git_commit,
            git_repository=args.git_repository,
            binary=args.binary,
            job_id=args.job_id,
            node_list=args.node_list,
        )
    else:
        value = build_result(
            _read_json(args.manifest), args.initiator_log, args.target_log
        )
    _write_json(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
