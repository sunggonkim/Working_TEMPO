"""Many-request short workload with unique first-chunk identities."""

from __future__ import annotations

import json
from pathlib import Path

from eval.sota_4node import tempo_pd_short_workload_v14 as short
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base


def _rewrite(path: Path, expected: int) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    base._require(len(rows) == expected, f"expected {expected} workload rows")
    with path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            row["prompt"] = f"Cold-cache high-load nonce {index:03d}.\n" + row["prompt"]
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def prepare(args, model: Path, python: Path):
    calibration, validation = short.prepare(args, model, python)
    marker = args.result_dir / "workloads" / "unique-head-many-v37-ready"
    expected = 3 * args.samples_per_bucket
    if args.node_index == 0:
        _rewrite(calibration, expected)
        _rewrite(validation, expected)
        marker.write_text("ready\n", encoding="utf-8")
    else:
        base.common._wait_file(marker, [])
    return calibration, validation
