#!/usr/bin/env python3
"""Create paired TEMPO-PD workloads in a three-row Latin bucket order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import make_tempo_pd_workloads_v1 as base


LATIN_BUCKET_ORDER = (0, 1, 2, 1, 2, 0, 2, 0, 1)


def _balanced(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {0: [], 1: [], 2: []}
    for row in rows:
        request_id = str(row["request_id"])
        bucket = int(request_id.split("-b", 1)[1].split("-", 1)[0])
        grouped[bucket].append(row)
    base._require(all(len(grouped[index]) == 3 for index in range(3)),
                  "balanced workload requires exactly three rows per bucket")
    positions = {index: 0 for index in range(3)}
    ordered: list[dict[str, object]] = []
    for bucket in LATIN_BUCKET_ORDER:
        ordered.append(grouped[bucket][positions[bucket]])
        positions[bucket] += 1
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", default="64,192,384")
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=32)
    args = parser.parse_args()
    base._require(args.model.is_absolute() and (args.model / "config.json").is_file(),
                  "model must be an absolute local directory")
    base._require(not args.output_dir.exists(), "output directory already exists")
    repetitions = tuple(int(value) for value in args.repetitions.split(","))
    base._require(len(repetitions) == 3 and args.samples_per_bucket == 3,
                  "v11 freezes three buckets and three samples")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    calibration, validation, buckets = base.build_workloads(
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        repetitions=repetitions,
        samples_per_bucket=args.samples_per_bucket,
        output_tokens=args.output_tokens,
    )
    calibration = _balanced(calibration)
    validation = _balanced(validation)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    calibration_path = args.output_dir / "calibration.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    base._write_jsonl(calibration_path, calibration)
    base._write_jsonl(validation_path, validation)
    (args.output_dir / "workload_manifest.json").write_text(json.dumps({
        "schema": "tempo-pd-paired-workloads-balanced-11",
        "model": str(args.model),
        "calibration_path": str(calibration_path),
        "validation_path": str(validation_path),
        "buckets": buckets,
        "request_count_per_workload": len(calibration),
        "dispatch_bucket_order": list(LATIN_BUCKET_ORDER),
        "calibration_and_validation_prompts_are_distinct": True,
        "calibration_and_validation_token_buckets_are_equal": True,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
