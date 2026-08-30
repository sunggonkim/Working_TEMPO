#!/usr/bin/env python3
"""Build a 24-row mixed 512/1230/2048-prompt, output16 workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


_NONCE = re.compile(r"nonce [0-9]{3}\.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.source) != 3 or args.label != ["512", "1230", "2048"]:
        raise ValueError("exact sources/labels 512,1230,2048 required")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite: {args.output}")
    mixed = []
    for bucket, source in zip(args.label, args.source, strict=True):
        rows = [json.loads(line) for line in
                source.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 24:
            raise ValueError(f"{bucket}: exactly 24 rows required")
        for index, row in enumerate(rows[:8]):
            value = dict(row)
            global_index = len(mixed)
            if len(_NONCE.findall(value["prompt"])) != 1:
                raise ValueError(f"{bucket}/{index}: nonce contract mismatch")
            value["prompt"] = _NONCE.sub(
                f"nonce {global_index:03d}.", value["prompt"])
            value["request_id"] = f"mix{bucket}-{index}"
            value["max_tokens"] = 16
            mixed.append(value)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in mixed),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(mixed), "buckets": args.label,
                      "output_tokens": 16}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
