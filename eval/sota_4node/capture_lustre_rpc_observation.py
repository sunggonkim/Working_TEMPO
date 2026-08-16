#!/usr/bin/env python3
"""Capture source-bound Lustre OSC RPC page histograms.

Per-client ``/proc/fs/lustre/osc/*/rpc_stats`` is available on some
Perlmutter images, but its ``pages per rpc`` table is not a byte counter.  This
helper preserves the raw page histogram and derives only page counts.  It is
therefore diagnostic evidence for the persistent endpoint, never a substitute
for the strict ``lustre_ost_bytes`` contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any


SCHEMA = "tempo-rd-lustre-rpc-page-observation-1"
_PAGE_ROW = re.compile(r"^\s*(\d+):\s*(\d+)\s+\d+\s+\d+\s*\|\s*(\d+)\s+\d+\s+\d+\s*$")


def parse_rpc_stats(text: str) -> dict[str, Any]:
    """Parse only the stable pages-per-RPC read/write histogram."""

    if type(text) is not str or "pages per rpc" not in text:
        raise ValueError("Lustre rpc_stats pages-per-rpc table is missing")
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "pages per rpc" in line)
    rows: list[dict[str, int]] = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        match = _PAGE_ROW.match(line)
        if match is None:
            raise ValueError("malformed Lustre pages-per-rpc row")
        pages, read_rpcs, write_rpcs = (int(value) for value in match.groups())
        if pages <= 0 or read_rpcs < 0 or write_rpcs < 0:
            raise ValueError("invalid Lustre pages-per-rpc value")
        rows.append({
            "pages": pages,
            "read_rpcs": read_rpcs,
            "write_rpcs": write_rpcs,
            "read_pages": pages * read_rpcs,
            "write_pages": pages * write_rpcs,
        })
    if not rows:
        raise ValueError("Lustre pages-per-rpc table has no rows")
    return {
        "read_rpcs": sum(row["read_rpcs"] for row in rows),
        "write_rpcs": sum(row["write_rpcs"] for row in rows),
        "read_pages": sum(row["read_pages"] for row in rows),
        "write_pages": sum(row["write_pages"] for row in rows),
        "rows": rows,
    }


def snapshot(*, mode: str, phase: str, proc_root: Path = Path("/proc/fs/lustre/osc")) -> dict[str, Any]:
    if not mode or phase not in {"start", "end"}:
        raise ValueError("mode/phase is invalid")
    proc_root = Path(proc_root)
    records: list[dict[str, Any]] = []
    for rpc_path in sorted(proc_root.glob("*/rpc_stats")):
        try:
            text = rpc_path.read_text(encoding="utf-8")
        except OSError:
            continue
        parsed = parse_rpc_stats(text)
        records.append({
            "endpoint": rpc_path.parent.name,
            "source": str(rpc_path),
            "counter_semantics": "cumulative_rpc_page_histogram",
            **parsed,
        })
    if not records:
        raise ValueError("no readable Lustre OSC rpc_stats records")
    return {
        "schema": SCHEMA,
        "mode": mode,
        "phase": phase,
        "scope": "endpoint",
        "scope_id": "lustre_client_osc",
        "timestamp_ns": time.monotonic_ns(),
        "page_size_bytes": 4096,
        "records": records,
        "hardware_counter": False,
        "causal_ready": False,
    }


def write_snapshot(root: Path, *, mode: str, phase: str, proc_root: Path = Path("/proc/fs/lustre/osc")) -> Path:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"lustre_rpc_{phase}.json"
    path.write_text(json.dumps(snapshot(mode=mode, phase=phase, proc_root=proc_root), sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--phase", choices=("start", "end"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc/fs/lustre/osc"))
    args = parser.parse_args()
    print(write_snapshot(args.output_root, mode=args.mode, phase=args.phase, proc_root=args.proc_root))


if __name__ == "__main__":
    main()
