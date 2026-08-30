"""Node-safe balanced workload preparation for the v11 crossover."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node.make_tempo_pd_workloads_balanced_v11 import LATIN_BUCKET_ORDER


def prepare(args, model: Path, python: Path) -> tuple[Path, Path]:
    workload_dir = args.result_dir / "workloads"
    manifest_path = workload_dir / "workload_manifest.json"
    if args.node_index == 0:
        subprocess.run([
            str(python), "-m", "eval.sota_4node.make_tempo_pd_workloads_balanced_v11",
            "--model", str(model), "--output-dir", str(workload_dir),
            "--repetitions", "64,192,384",
            "--samples-per-bucket", str(args.samples_per_bucket),
            "--output-tokens", str(args.output_tokens),
        ], cwd=args.repo_root, check=True, timeout=120.0)
    else:
        base.common._wait_file(manifest_path, [])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base._require(payload.get("dispatch_bucket_order") == list(LATIN_BUCKET_ORDER),
                  "Latin bucket order drifted")
    return workload_dir / "calibration.jsonl", workload_dir / "validation.jsonl"
