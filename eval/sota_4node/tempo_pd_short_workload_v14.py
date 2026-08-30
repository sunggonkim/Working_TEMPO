"""Node-safe nine-request equal-short-context workload preparation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base


REPETITIONS = (64, 64, 64)


def prepare(args, model: Path, python: Path) -> tuple[Path, Path]:
    workload_dir = args.result_dir / "workloads"
    manifest_path = workload_dir / "workload_manifest.json"
    if args.node_index == 0:
        subprocess.run([
            str(python), "-m", "eval.sota_4node.make_tempo_pd_workloads_v1",
            "--model", str(model), "--output-dir", str(workload_dir),
            "--repetitions", ",".join(map(str, REPETITIONS)),
            "--samples-per-bucket", str(args.samples_per_bucket),
            "--output-tokens", str(args.output_tokens),
        ], cwd=args.repo_root, check=True, timeout=120.0)
    else:
        base.common._wait_file(manifest_path, [])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    buckets = payload.get("buckets")
    base._require(isinstance(buckets, list) and len(buckets) == 3,
                  "three workload rows required")
    base._require([row.get("repetitions") for row in buckets] == list(REPETITIONS),
                  "short repetitions drifted")
    prompt_tokens = {row.get("prompt_tokens") for row in buckets}
    base._require(len(prompt_tokens) == 1, "all short rows must have equal tokens")
    return workload_dir / "calibration.jsonl", workload_dir / "validation.jsonl"
