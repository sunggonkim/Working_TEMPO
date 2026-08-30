#!/usr/bin/env python3
"""Build 24 rows spanning 3 prompt buckets and 4 validated output lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_OUTPUTS = (16, 16, 32, 32, 64, 64, 128, 128)


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
        raise ValueError("exact frozen 24-row mixed source required")
    counts = {}
    for bucket_index, bucket in enumerate((512, 1230, 2048)):
        for index, output_tokens in enumerate(_OUTPUTS):
            row = rows[bucket_index * 8 + index]
            if not str(row["request_id"]).startswith(f"mix{bucket}-"):
                raise ValueError("prompt bucket order mismatch")
            row["request_id"] = f"het-p{bucket}-o{output_tokens}-r{index % 2}"
            row["max_tokens"] = output_tokens
            counts[(bucket, output_tokens)] = counts.get((bucket, output_tokens), 0) + 1
    if set(counts.values()) != {2} or len(counts) != 12:
        raise ValueError("heterogeneous cell geometry mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8")
    print(json.dumps({"rows": 24, "cells": len(counts),
                      "outputs": [16, 32, 64, 128]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
