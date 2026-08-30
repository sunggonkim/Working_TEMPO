#!/usr/bin/env python3
"""Derive the frozen mixed-prompt workload with exactly 128 output tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    rows = [json.loads(line) for line in
            args.source.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 24:
        raise ValueError("exactly 24 frozen mixed-prompt rows required")
    expected = {"mix512": 8, "mix1230": 8, "mix2048": 8}
    observed = {key: 0 for key in expected}
    for row in rows:
        request_id = str(row["request_id"])
        bucket = next((key for key in expected if request_id.startswith(key + "-")), None)
        if bucket is None:
            raise ValueError(f"unexpected request id: {request_id}")
        observed[bucket] += 1
        if row.get("max_tokens") != 16:
            raise ValueError("source must be the frozen output16 workload")
        row["max_tokens"] = 128
    if observed != expected:
        raise ValueError(f"bucket geometry mismatch: {observed}")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "buckets": observed,
                      "output_tokens": 128}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
