#!/usr/bin/env python3
"""Create paired calibration/validation workloads with equal token buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence


SCHEMA = "tempo-pd-paired-workloads-1"
PROMPT_UNIT = (
    "Measured admission must preserve output correctness, decode latency, and "
    "the exact live KV routing contract. "
)
MARKERS = (
    "Alpha route. ", "Bravo route. ", "Charlie route. ", "Delta route. ",
    "Echo route. ", "Foxtrot route. ", "Golf route. ", "Hotel route. ",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_workloads(
    encode: Callable[[str], Sequence[int]],
    *,
    repetitions: Sequence[int],
    samples_per_bucket: int,
    output_tokens: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    _require(bool(repetitions) and all(type(value) is int and value > 0 for value in repetitions),
             "repetitions must be positive ints")
    _require(type(samples_per_bucket) is int and samples_per_bucket >= 3,
             "samples_per_bucket must be at least three")
    _require(type(output_tokens) is int and output_tokens >= 2,
             "output_tokens must be at least two")
    calibration: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    buckets: list[dict[str, object]] = []
    global_index = 0
    for bucket, count in enumerate(repetitions):
        candidates = [(marker + PROMPT_UNIT * count, marker) for marker in MARKERS]
        grouped: dict[int, list[tuple[str, str]]] = {}
        for prompt, marker in candidates:
            grouped.setdefault(len(encode(prompt)), []).append((prompt, marker))
        choices = [(tokens, values) for tokens, values in grouped.items() if len(values) >= 2]
        _require(bool(choices), f"bucket {bucket}: no distinct equal-token prompt pair")
        prompt_tokens, values = sorted(choices, key=lambda item: item[0])[0]
        calibration_prompt, calibration_marker = values[0]
        validation_prompt, validation_marker = values[1]
        _require(calibration_prompt != validation_prompt, "prompt pair must be distinct")
        for sample in range(samples_per_bucket):
            calibration.append({
                "request_id": f"cal-b{bucket}-s{sample}-r{global_index}",
                "prompt": calibration_prompt,
                "max_tokens": output_tokens,
            })
            validation.append({
                "request_id": f"val-b{bucket}-s{sample}-r{global_index}",
                "prompt": validation_prompt,
                "max_tokens": output_tokens,
            })
            global_index += 1
        buckets.append({
            "bucket": bucket,
            "repetitions": count,
            "prompt_tokens": prompt_tokens,
            "calibration_marker": calibration_marker,
            "validation_marker": validation_marker,
            "samples_per_route": samples_per_bucket,
            "output_tokens": output_tokens,
        })
    return calibration, validation, buckets


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    _require(not path.exists(), f"refusing to overwrite: {path}")
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", default="64,256,512")
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=2)
    args = parser.parse_args()
    _require(args.model.is_absolute() and (args.model / "config.json").is_file(),
             "model must be an absolute local directory")
    _require(not args.output_dir.exists(), "output directory already exists")
    repetitions = tuple(int(value) for value in args.repetitions.split(","))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    calibration, validation, buckets = build_workloads(
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        repetitions=repetitions,
        samples_per_bucket=args.samples_per_bucket,
        output_tokens=args.output_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    calibration_path = args.output_dir / "calibration.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    _write_jsonl(calibration_path, calibration)
    _write_jsonl(validation_path, validation)
    (args.output_dir / "workload_manifest.json").write_text(json.dumps({
        "schema": SCHEMA,
        "model": str(args.model),
        "calibration_path": str(calibration_path),
        "validation_path": str(validation_path),
        "buckets": buckets,
        "request_count_per_workload": len(calibration),
        "calibration_and_validation_prompts_are_distinct": True,
        "calibration_and_validation_token_buckets_are_equal": True,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
