"""Nine equal-size short requests with unique cold-cache prompt identities."""

from __future__ import annotations

import json
from pathlib import Path

from eval.sota_4node import tempo_pd_short_workload_v14 as short
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base


READY = "unique-short-v21-ready"


def _rewrite(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    base._require(len(rows) == 9, "unique short workload requires nine rows")
    prompts: set[str] = set()
    with path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            row["prompt"] += f"\nCold-cache request nonce {index:02d}."
            prompts.add(row["prompt"])
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    base._require(len(prompts) == 9, "cold-cache prompts must be unique")


def prepare(args, model: Path, python: Path) -> tuple[Path, Path]:
    calibration, validation = short.prepare(args, model, python)
    marker = args.result_dir / "workloads" / READY
    if args.node_index == 0:
        _rewrite(calibration)
        _rewrite(validation)
        marker.write_text("ready\n", encoding="utf-8")
    else:
        base.common._wait_file(marker, [])
    return calibration, validation
