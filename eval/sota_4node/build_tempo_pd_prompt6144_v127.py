#!/usr/bin/env python3
"""Build 24 prompt6144-class rows split between output16 and output128."""

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
    long_source = [row for row in source
                   if str(row["request_id"]).startswith("mix2048-")]
    if len(long_source) != 8:
        raise ValueError("exact eight prompt2048 source rows required")
    rows = []
    for index in range(24):
        base = dict(long_source[index % 8])
        first, separator, suffix = base["prompt"].partition("\n")
        if not separator or "nonce " not in first:
            raise ValueError("source nonce/header contract mismatch")
        base["prompt"] = (
            f"{first}\n{suffix}\n{suffix}\n{suffix}\nLong6144 variant {index:02d}."
        )
        output_tokens = 16 if index < 12 else 128
        base["max_tokens"] = output_tokens
        base["request_id"] = f"long6144-o{output_tokens}-r{index % 12}"
        rows.append(base)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": 24, "outputs": {"16": 12, "128": 12}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
