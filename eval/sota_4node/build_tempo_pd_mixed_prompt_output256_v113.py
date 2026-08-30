#!/usr/bin/env python3
"""Derive the frozen mixed-prompt workload with exactly 256 output tokens."""

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
    if len(rows) != 24 or any(row.get("max_tokens") != 16 for row in rows):
        raise ValueError("frozen 24-row output16 source required")
    for row in rows:
        row["max_tokens"] = 256
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8")
    print(json.dumps({"rows": 24, "output_tokens": 256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
