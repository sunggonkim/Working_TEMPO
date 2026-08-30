#!/usr/bin/env python3
"""Collision-free paired workload with explicit burst/idle arrivals."""

import json
from pathlib import Path
import re
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v308 as unique


ITEM = re.compile(r".*-item-([0-9]{2})$")
PAIRS_PER_BURST = 4
PAIR_GAP_MS = 14.0
INTER_BURST_GAP_MS = 220.0
BURST_STRIDE_MS = (PAIRS_PER_BURST - 1) * PAIR_GAP_MS + INTER_BURST_GAP_MS


def _argument(name):
    return sys.argv[sys.argv.index(name) + 1]


def _rows(source: Path, phase: str):
    rows = unique._rows(source, phase)
    for row in rows:
        match = ITEM.fullmatch(row["request_id"])
        if match is None:
            raise ValueError("item identity missing")
        item = int(match.group(1))
        burst, slot = divmod(item, PAIRS_PER_BURST)
        row["arrival_offset_ms"] = burst * BURST_STRIDE_MS + slot * PAIR_GAP_MS
    return rows


def _run_phase(root: Path, source: Path, phase: str, workers: str):
    rows = _rows(source, phase)
    workload = root / f"{phase}.jsonl"
    raw = root / f"{phase}.raw.json"
    unique.base._write(workload, rows)
    command = [
        sys.executable, "-m",
        "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_salted_v296",
        "--base-url", _argument("--base-url"),
        "--model", _argument("--model"),
        "--served-model-name", _argument("--served-model-name"),
        "--workload", str(workload), "--output", str(raw),
        "--mode", "tempo_auto", "--run-id", f"bursty-paired-{phase}",
        "--default-max-tokens", "32", "--max-workers", workers,
        "--timeout-s", "120", "--seed", "20260816",
    ]
    if "--api-key-env" in sys.argv:
        command.extend(("--api-key-env", _argument("--api-key-env")))
    subprocess.run(command, check=True, timeout=300.0)
    value = json.loads(raw.read_text())
    span_ms = max(row["arrival_offset_ms"] for row in rows) + PAIR_GAP_MS
    value["workload"]["request_rate_per_s"] = len(rows) * 1000.0 / span_ms
    value["mixed_crossover_contract"] = {
        "schema": "tempo-pd-mixed-request-crossover-260",
        "phase": phase,
        "base_items": 24,
        "requests": 48,
        "tempo_requests": 24,
        "lmcache_remote_requests": 24,
        "same_client_window": True,
        "paired_by_geometry_and_base_item": True,
        "variant_assignment_counterbalanced_by_item_parity": True,
        "warm_max_workers": 1,
        "measured_max_workers": int(_argument("--max-workers")),
        "cache_isolation": "vllm_cache_salt_plus_unique_18_token_regions_v305",
        "arrival_trace": "six_bursts_four_pairs_14ms_with_220ms_idle_v322",
        "explicit_arrivals": True,
        "pairs_per_burst": PAIRS_PER_BURST,
        "pair_gap_ms": PAIR_GAP_MS,
        "inter_burst_gap_ms": INTER_BURST_GAP_MS,
        "scheduled_span_ms": span_ms,
    }
    raw.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return raw


def main():
    if _argument("--run-id").endswith("-warmup"):
        return unique.base.base.serial.main()
    output = Path(_argument("--output")).resolve()
    source = Path(_argument("--workload")).resolve()
    root = output.parent / "bursty_paired_v322"
    root.mkdir()
    _run_phase(root, source, "warm", "1")
    measured = _run_phase(root, source, "measured", _argument("--max-workers"))
    output.write_text(measured.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
