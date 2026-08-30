#!/usr/bin/env python3
"""Build one 24-row workload spanning output16-256 and prompt4094."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path, output_tokens: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 24 or any(row.get("max_tokens") != output_tokens for row in rows):
        raise ValueError(f"{path}: expected 24 output{output_tokens} rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise ValueError("refusing existing cross-geometry output")
    selected = []
    for output_tokens, root in (
        (16, "tempo_pd_mixed_prompt_output16_input_v98"),
        (32, "tempo_pd_mixed_prompt_output32_input_v114"),
        (64, "tempo_pd_mixed_prompt_output64_input_v114"),
        (128, "tempo_pd_mixed_prompt_output128_input_v109"),
        (256, "tempo_pd_mixed_prompt_output256_input_v113"),
    ):
        source = _rows(args.results_root / root / "workloads/validation.jsonl",
                       output_tokens)
        selected.extend(dict(source[index]) for index in (0, 8, 16, 17))
    long_rows = _rows(
        args.results_root / "tempo_pd_prompt4096_input_v123/workloads/validation.jsonl",
        16)
    # That file is mixed output16/output128; validate it explicitly instead.
    if sorted(row["max_tokens"] for row in long_rows) != [16] * 12 + [128] * 12:
        raise ValueError("prompt4096 output16/output128 geometry mismatch")
    selected.extend(dict(long_rows[index]) for index in (0, 1, 12, 13))
    if len(selected) != 24:
        raise AssertionError("exact 24 cross-geometry rows required")
    for index, row in enumerate(selected):
        row["request_id"] = f"cross-geometry-{index:02d}"
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8")
    print(json.dumps({"rows": 24,
                      "outputs": {str(token): sum(row["max_tokens"] == token
                                                   for row in selected)
                                  for token in (16, 32, 64, 128, 256)}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
