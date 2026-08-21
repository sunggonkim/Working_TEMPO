#!/usr/bin/env python3
"""Add a same-window, geometry-paired Tempo/LMCache crossover block."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_serial_lm_warm_v230 as serial


NONCE = re.compile(r"nonce ([0-9]{3})\.")


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _variant(row: dict, *, arm: str, phase: str, variant: str,
             offset: int, item: int) -> dict:
    value = dict(row)
    match = NONCE.search(value["prompt"])
    if match is None:
        raise ValueError("mixed crossover nonce missing")
    nonce = int(match.group(1)) + offset
    if nonce > 999:
        raise ValueError("mixed crossover nonce overflow")
    value["prompt"] = NONCE.sub(f"nonce {nonce:03d}.", value["prompt"])
    value["request_id"] = (
        f"ssb-{arm}-r0-{phase}-mix{variant}-cache-item-{item:02d}")
    return value


def _rows(source: Path, phase: str) -> list[dict]:
    base = [json.loads(line) for line in source.read_text().splitlines()]
    if len(base) != 24:
        raise ValueError("mixed crossover requires 24 base items")
    rows = []
    for item, row in enumerate(base):
        assignments = (("tempo", "A", 930), ("remote", "B", 960))
        if item % 2:
            assignments = tuple(reversed(assignments))
        for arm, variant, offset in assignments:
            rows.append(_variant(row, arm=arm, phase=phase, variant=variant,
                                 offset=offset, item=item))
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                            for row in rows))


def _run_mixed() -> None:
    source = Path(_argument("--workload")).resolve()
    stage = Path(_argument("--output")).resolve().parent
    root = stage / "mixed_request_crossover_v260"
    root.mkdir()
    common = [
        sys.executable, "-m",
        "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
        "--base-url", _argument("--base-url"),
        "--model", _argument("--model"),
        "--served-model-name", _argument("--served-model-name"),
        "--mode", "tempo_auto",
        "--default-max-tokens", (_argument("--default-max-tokens")
                                  if "--default-max-tokens" in sys.argv else "32"),
        "--max-workers", _argument("--max-workers"),
        "--request-rate", _argument("--request-rate"),
        "--timeout-s", _argument("--timeout-s"),
        "--seed", _argument("--seed") if "--seed" in sys.argv else "20260815",
    ]
    if "--api-key-env" in sys.argv:
        common.extend(("--api-key-env", _argument("--api-key-env")))
    for phase in ("warm", "measured"):
        workload = root / f"{phase}.jsonl"
        raw = root / f"{phase}.raw.json"
        _write(workload, _rows(source, phase))
        command = common + [
            "--workload", str(workload), "--output", str(raw),
            "--run-id", f"mixed-crossover-{phase}",
        ]
        subprocess.run(command, check=True, timeout=1200.0)
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
        }
        raw.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    status = serial.main()
    if not _argument("--run-id").endswith("-warmup"):
        _run_mixed()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
