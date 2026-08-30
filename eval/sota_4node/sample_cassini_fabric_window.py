"""Collect one bounded Cassini window synchronized with a CXI co-job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time

from tempo.cassini_endpoint import CassiniEndpointSampler
from tempo.pd_endpoint_evidence import PDEndpointIdentity, PDEndpointRole


SCHEMA = "tempo-cassini-fabric-window-v1"
_PATTERNS = {
    "pairwise-bidir",
    "pd-2p2d-incast",
    "pd-3p1d-incast",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-seconds", type=float, required=True)
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--ready-prefix", type=Path, required=True)
    parser.add_argument("--pattern", choices=sorted(_PATTERNS), required=True)
    parser.add_argument("--start-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if not 0.1 <= args.sample_seconds <= 9.0:
        parser.error("--sample-seconds must be in [0.1, 9.0]")
    if not 1.0 <= args.start_timeout_seconds <= 60.0:
        parser.error("--start-timeout-seconds must be in [1, 60]")
    for name in ("start_file", "ready_prefix"):
        value = getattr(args, name)
        if not value.is_absolute():
            parser.error(f"--{name.replace('_', '-')} must be absolute")
    if args.start_file.parent != args.ready_prefix.parent:
        parser.error("start and ready markers must share one result directory")
    if not args.start_file.parent.is_dir():
        parser.error("marker parent must already exist")
    return args


def _role(pattern: str, node_index: int) -> PDEndpointRole:
    if pattern == "pd-3p1d-incast":
        return (
            PDEndpointRole.DECODER
            if node_index == 3
            else PDEndpointRole.PREFILL
        )
    return (
        PDEndpointRole.PREFILL
        if node_index in {0, 2}
        else PDEndpointRole.DECODER
    )


def main() -> int:
    args = _arguments()
    raw_node_index = os.environ.get("SLURM_PROCID")
    try:
        node_index = int(raw_node_index) if raw_node_index is not None else -1
    except ValueError as exc:
        raise SystemExit("SLURM_PROCID must be an integer") from exc
    if not 0 <= node_index < 4:
        raise SystemExit("sampler requires exactly one ordered task per node")

    identity = PDEndpointIdentity(
        endpoint_id=f"{socket.gethostname()}-{_role(args.pattern, node_index).value}",
        role=_role(args.pattern, node_index),
        pair_index=node_index,
    )
    sampler = None
    baseline = None
    initialization_error = None
    try:
        sampler = CassiniEndpointSampler(
            identity,
            min_interval_ms=0.0,
            max_window_ms=10_000.0,
        )
        baseline = sampler.sample(force=True)
    except (OSError, TypeError, ValueError) as exc:
        initialization_error = f"{type(exc).__name__}:{exc}"

    ready_file = Path(f"{args.ready_prefix}.node{node_index}")
    if ready_file.exists():
        raise SystemExit(f"ready marker already exists: {ready_file}")
    ready_file.write_text("ready\n", encoding="ascii")

    deadline = time.monotonic() + args.start_timeout_seconds
    while not args.start_file.is_file():
        if time.monotonic() >= deadline:
            raise SystemExit("timed out waiting for synchronized start")
        time.sleep(0.01)
    started_ns = time.perf_counter_ns()
    synchronized_baseline = (
        sampler.sample(force=True) if sampler is not None else None)
    time.sleep(args.sample_seconds)
    sample = sampler.sample(force=True) if sampler is not None else None
    finished_ns = time.perf_counter_ns()

    print(json.dumps({
        "schema": SCHEMA,
        "node_index": node_index,
        "hostname": socket.gethostname(),
        "pattern": args.pattern,
        "role": identity.role.value,
        "sample_seconds": args.sample_seconds,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "initialization_error": initialization_error,
        "baseline_valid": (
            baseline.get("valid") if isinstance(baseline, dict) else None),
        "baseline_reason": (
            baseline.get("invalid_reason")
            if isinstance(baseline, dict) else None),
        "synchronized_baseline_valid": (
            synchronized_baseline.get("valid")
            if isinstance(synchronized_baseline, dict) else None),
        "synchronized_baseline_reason": (
            synchronized_baseline.get("invalid_reason")
            if isinstance(synchronized_baseline, dict) else None),
        "sample": sample,
    }, sort_keys=True, separators=(",", ":")), flush=True)
    return 0 if sample is not None and sample.get("valid") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
