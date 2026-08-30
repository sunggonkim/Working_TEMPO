#!/usr/bin/env python3
"""Run the mixed crossover with isolated, deterministic cold-cache keys."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_v265 as base


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _run_phase(root: Path, source: Path, phase: str, workers: str) -> Path:
    workload = root / f"{phase}.jsonl"
    raw = root / f"{phase}.raw.json"
    base._write(workload, base._rows(source, phase))
    command = [
        sys.executable, "-m",
        "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_salted_v296",
        "--base-url", _argument("--base-url"),
        "--model", _argument("--model"),
        "--served-model-name", _argument("--served-model-name"),
        "--workload", str(workload), "--output", str(raw),
        "--mode", "tempo_auto", "--run-id", f"mixed-only-salted-{phase}",
        "--default-max-tokens", "32", "--max-workers", workers,
        "--request-rate", _argument("--request-rate"),
        "--timeout-s", "120", "--seed", "20260815",
    ]
    if "--api-key-env" in sys.argv:
        command.extend(("--api-key-env", _argument("--api-key-env")))
    subprocess.run(command, check=True, timeout=300.0)
    value = json.loads(raw.read_text())
    value["mixed_crossover_contract"] = {
        "schema": "tempo-pd-mixed-request-crossover-260",
        "phase": phase,
        "base_items": 24,
        "requests": 48,
        "tempo_requests": 24,
        "lmcache_remote_requests": 24,
        "same_client_window": True,
        "paired_by_geometry_and_base_item": True,
        "nonce_offsets": [930, 960],
        "variant_assignment_counterbalanced_by_item_parity": True,
        "warm_max_workers": 1,
        "measured_max_workers": int(_argument("--max-workers")),
        "cache_isolation": "sha256_request_id_vllm_cache_salt",
        "cache_salt_unique_per_request": True,
    }
    raw.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return raw


def main() -> int:
    run_id = _argument("--run-id")
    if run_id.endswith("-warmup"):
        return base.serial.main()
    output = Path(_argument("--output")).resolve()
    source = Path(_argument("--workload")).resolve()
    root = output.parent / "mixed_request_crossover_salted_v297"
    root.mkdir()
    _run_phase(root, source, "warm", "1")
    measured = _run_phase(root, source, "measured", _argument("--max-workers"))
    output.write_text(measured.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
