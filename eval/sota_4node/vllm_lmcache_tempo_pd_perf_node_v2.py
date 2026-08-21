#!/usr/bin/env python3
"""Context-safe workload wrapper for the TEMPO-PD performance node."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base


PROMPT_REPETITIONS = (64, 192, 384)
MAX_MODEL_LEN = 8192


def _prepare_workloads(args, model: Path, python: Path) -> tuple[Path, Path]:
    workload_dir = args.result_dir / "workloads"
    manifest_path = workload_dir / "workload_manifest.json"
    if args.node_index == 0:
        subprocess.run([
            str(python), "-m", "eval.sota_4node.make_tempo_pd_workloads_v1",
            "--model", str(model),
            "--output-dir", str(workload_dir),
            "--repetitions", ",".join(map(str, PROMPT_REPETITIONS)),
            "--samples-per-bucket", str(args.samples_per_bucket),
            "--output-tokens", str(args.output_tokens),
        ], cwd=args.repo_root, check=True, timeout=120.0)
    else:
        base.common._wait_file(manifest_path, [])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    buckets = payload.get("buckets")
    base._require(isinstance(buckets, list) and len(buckets) == 3,
                  "three workload buckets are required")
    base._require(
        [row.get("repetitions") for row in buckets] == list(PROMPT_REPETITIONS),
        "workload repetitions drifted",
    )
    base._require(
        all(
            type(row.get("prompt_tokens")) is int
            and row["prompt_tokens"] + args.output_tokens <= MAX_MODEL_LEN
            for row in buckets
        ),
        "prompt plus output exceeds max model length",
    )
    return workload_dir / "calibration.jsonl", workload_dir / "validation.jsonl"


def main() -> int:
    base._prepare_workloads = _prepare_workloads
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
