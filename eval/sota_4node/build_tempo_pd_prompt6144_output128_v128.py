#!/usr/bin/env python3
"""Convert the frozen prompt6144 diagnostic rows to 24 output128 rows."""

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
    source = [json.loads(line) for line in
              args.source.read_text(encoding="utf-8").splitlines()]
    if len(source) != 24:
        raise ValueError("exact 24 source rows required")
    rows = []
    for index, source_row in enumerate(source):
        row = dict(source_row)
        row["max_tokens"] = 128
        row["request_id"] = f"long6144-prod128-r{index}"
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": 24, "output_tokens": 128}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
