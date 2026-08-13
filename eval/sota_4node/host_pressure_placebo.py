#!/usr/bin/env python3
"""Run one rank of the host-NUMA pressure placebo.

This helper is intentionally separate from the checkpoint/KV runner.  It is
used by a future approved G1 allocation to create the ``placebo`` evidence
required by ``validate_g1_result.py``: a foreground-only process can run while
this process repeatedly touches a declared host buffer, and the helper emits
monotonic process NUMA-map samples.  It never submits Slurm work, chooses a
NUMA node, or converts topology labels into causal evidence.

The parent-pid guard is important for the existing training harness: if a
rank exits through ``os._exit`` or is killed by a timeout, the worker stops
instead of becoming an orphan.  A partial output is still written, but the
validator will reject it unless the declared buffer and positive busy interval
were observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Iterable

try:
    from tempo.host_pressure import HostPressureSample, HostPressureSpec, validate_host_pressure_series
except ModuleNotFoundError:  # direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tempo.host_pressure import HostPressureSample, HostPressureSpec, validate_host_pressure_series


SCHEMA_VERSION = "tempo-rd-host-pressure-run-1"
PAGE_SIZE = int(os.sysconf("SC_PAGESIZE"))
_NUMA_RE = re.compile(r"(?:^|\s)N(?P<node>[0-9]+)=(?P<pages>[0-9]+)")


def _record_digest(record: dict[str, object]) -> str:
    """Hash the canonical record with its digest field blanked."""

    import copy

    canonical = copy.deepcopy(record)
    canonical["output_sha256"] = ""
    encoded = (json.dumps(canonical, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def process_numa_bytes(numa_node: int, *, proc_maps: Path = Path("/proc/self/numa_maps")) -> int:
    """Return bytes currently attributed to ``numa_node`` by this process.

    ``/proc/<pid>/numa_maps`` reports pages, not bytes.  The helper keeps this
    conversion explicit and fails closed when the kernel file is unavailable
    or malformed; a guessed value must never become causal evidence.
    """

    if type(numa_node) is not int or numa_node < 0:
        raise ValueError("numa_node must be a non-negative int")
    try:
        text = proc_maps.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read process NUMA map {proc_maps}: {exc}") from exc
    pages = 0
    found = False
    for line in text.splitlines():
        for match in _NUMA_RE.finditer(line):
            if int(match.group("node")) == numa_node:
                pages += int(match.group("pages"))
                found = True
    if not found:
        raise RuntimeError(f"NUMA node {numa_node} is absent from {proc_maps}")
    return pages * PAGE_SIZE


def _touch_pages(buffer: bytearray) -> tuple[int, int]:
    """Touch every page and return ``(bytes, busy_ns)``."""

    start = time.monotonic_ns()
    touched = 0
    for offset in range(0, len(buffer), PAGE_SIZE):
        buffer[offset] = (buffer[offset] + 1) & 0xFF
        touched += PAGE_SIZE
    # The final page may be shorter than PAGE_SIZE; the declared buffer, not
    # the page-rounded count, is the contract-visible amount.
    touched = len(buffer)
    return touched, time.monotonic_ns() - start


def run_pressure(
    spec: HostPressureSpec,
    *,
    output: Path,
    parent_pid: int | None = None,
    stop_file: Path | None = None,
    proc_maps: Path = Path("/proc/self/numa_maps"),
    clock_ns=time.monotonic_ns,
    sleep=time.sleep,
) -> dict[str, object]:
    """Touch the host buffer and atomically write a rank-local JSON record."""

    if not isinstance(spec, HostPressureSpec):
        raise TypeError("spec must be a HostPressureSpec")
    if parent_pid is not None and (type(parent_pid) is not int or parent_pid <= 0):
        raise ValueError("parent_pid must be a positive int")
    if stop_file is not None:
        stop_file = Path(stop_file).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = bytearray(spec.buffer_bytes)
    baseline = process_numa_bytes(spec.numa_node, proc_maps=proc_maps)
    start_ns = clock_ns()
    next_sample_ns = start_ns
    touched_total = 0
    busy_total = 0
    samples: list[HostPressureSample] = []

    def sample(sample_id: str, now_ns: int) -> None:
        samples.append(
            HostPressureSample(
                sample_id=sample_id,
                timestamp_ns=now_ns,
                cumulative_touched_bytes=touched_total,
                cumulative_busy_ns=busy_total,
                numa_node_bytes=max(0, process_numa_bytes(spec.numa_node, proc_maps=proc_maps) - baseline),
            )
        )

    sample("start", start_ns)
    while clock_ns() - start_ns < spec.duration_ns:
        if stop_file is not None and stop_file.exists():
            break
        if parent_pid is not None:
            try:
                observed_parent = os.getppid()
            except OSError:
                observed_parent = -1
            if observed_parent != parent_pid:
                break
        touched, busy = _touch_pages(buffer)
        touched_total += touched
        busy_total += busy
        now_ns = clock_ns()
        if now_ns >= next_sample_ns:
            sample(f"sample-{len(samples):06d}", now_ns)
            next_sample_ns = now_ns + spec.sample_period_ns
        else:
            sleep(min(spec.sample_period_ns / 1_000_000_000, 0.001))

    end_ns = clock_ns()
    if not samples or samples[-1].timestamp_ns < end_ns:
        sample("finish", end_ns)
    # This validation intentionally happens before publication.  If NUMA
    # accounting was unavailable, the process raises rather than publishing a
    # plausible-looking placebo record.
    validated = validate_host_pressure_series(spec, samples)
    record = {
        "schema_version": SCHEMA_VERSION,
        "spec": {
            "rank": spec.rank,
            "world_size": spec.world_size,
            "numa_node": spec.numa_node,
            "buffer_bytes": spec.buffer_bytes,
            "duration_ns": spec.duration_ns,
            "sample_period_ns": spec.sample_period_ns,
            "source": spec.source,
        },
        "samples": [
            {
                "sample_id": item.sample_id,
                "timestamp_ns": item.timestamp_ns,
                "cumulative_touched_bytes": item.cumulative_touched_bytes,
                "cumulative_busy_ns": item.cumulative_busy_ns,
                "numa_node_bytes": item.numa_node_bytes,
            }
            for item in validated
        ],
        "output_sha256": "",
    }
    record["output_sha256"] = _record_digest(record)
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, output)
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--numa-node", type=int, required=True)
    parser.add_argument("--buffer-mib", type=int, default=64)
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--stop-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = HostPressureSpec(
        rank=args.rank,
        world_size=args.world_size,
        numa_node=args.numa_node,
        buffer_bytes=args.buffer_mib * 1024 * 1024,
        duration_ns=args.duration_ms * 1_000_000,
        sample_period_ns=args.sample_ms * 1_000_000,
    )
    run_pressure(spec, output=args.output, parent_pid=args.parent_pid, stop_file=args.stop_file)


if __name__ == "__main__":
    main()
