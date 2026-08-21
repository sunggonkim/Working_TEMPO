#!/usr/bin/env python3
"""Run only the counterbalanced same-window crossover after standard warm seed."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_serial_lm_warm_v230 as serial
from eval.sota_4node import run_tempo_pd_same_server_mixed_crossover_client_v260 as mixed


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _rows(source: Path, phase: str) -> list[dict]:
    base = [json.loads(line) for line in source.read_text().splitlines()]
    if len(base) != 24:
        raise ValueError("mixed-only requires 24 base items")
    rows = []
    for item, row in enumerate(base):
        assignments = (("tempo", "A", 930), ("remote", "B", 960))
        if item % 2:
            assignments = (("remote", "A", 930), ("tempo", "B", 960))
        for arm, variant, offset in assignments:
            rows.append(mixed._variant(
                row, arm=arm, phase=phase, variant=variant,
                offset=offset, item=item))
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                            for row in rows))


def _run_phase(root: Path, source: Path, phase: str, workers: str) -> Path:
    workload = root / f"{phase}.jsonl"
    raw = root / f"{phase}.raw.json"
    _write(workload, _rows(source, phase))
    command = [
        sys.executable, "-m",
        "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
        "--base-url", _argument("--base-url"),
        "--model", _argument("--model"),
        "--served-model-name", _argument("--served-model-name"),
        "--workload", str(workload), "--output", str(raw),
        "--mode", "tempo_auto", "--run-id", f"mixed-only-{phase}",
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
    }
    raw.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return raw


def main() -> int:
    run_id = _argument("--run-id")
    if run_id.endswith("-warmup"):
        return serial.main()
    output = Path(_argument("--output")).resolve()
    source = Path(_argument("--workload")).resolve()
    root = output.parent / "mixed_request_crossover_v265"
    root.mkdir()
    _run_phase(root, source, "warm", "1")
    measured = _run_phase(root, source, "measured", _argument("--max-workers"))
    output.write_text(measured.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
