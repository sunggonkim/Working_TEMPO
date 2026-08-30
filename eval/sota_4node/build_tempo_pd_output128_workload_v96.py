#!/usr/bin/env python3
"""Clone a 24-row explicit workload with exactly 128 output tokens."""

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
        raise ValueError("exactly 24 source rows required")
    for row in rows:
        row["max_tokens"] = 128
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output_tokens": 128}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
