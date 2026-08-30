#!/usr/bin/env python3
"""Add balanced unmeasured probes and publish an epoch route mode."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as balanced
from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_serial_lm_warm_v230 as serial
from eval.sota_4node.tempo_pd_same_server_epoch_guard_router_v248 import MODE_SCHEMA


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _rewrite_workload(source: Path, output: Path, old: str, new: str) -> None:
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        request_id = row["request_id"]
        if old not in request_id:
            raise ValueError("calibration source prefix changed")
        row["request_id"] = request_id.replace(old, new, 1)
    output.write_text("".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def _metrics(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text())
    if not value["validation"]["performance_claim_allowed"]:
        raise ValueError(f"invalid calibration artifact: {path}")
    rows = value["requests"]
    e2e = [
        (row["token_arrival_offsets_ns"][-1] - row["dispatch_offset_ns"]) / 1e6
        for row in rows
    ]
    return {
        "throughput_per_s": len(rows) / (value["run"]["client_window_ns"] / 1e9),
        "e2e_max_ms": max(e2e),
    }


def select_mode(local: list[dict[str, float]], tempo: list[dict[str, float]]) -> dict:
    if len(local) != 3 or len(tempo) != 3:
        raise ValueError("exactly three calibration replicates per candidate required")
    local_throughput = statistics.median(row["throughput_per_s"] for row in local)
    tempo_throughput = statistics.median(row["throughput_per_s"] for row in tempo)
    local_e2e = statistics.median(row["e2e_max_ms"] for row in local)
    tempo_e2e = statistics.median(row["e2e_max_ms"] for row in tempo)
    throughput_ratio = tempo_throughput / local_throughput
    e2e_ratio = tempo_e2e / local_e2e
    selected = (
        "policy8"
        if throughput_ratio >= 1.005 and e2e_ratio <= 1.0
        else "fixed_local"
    )
    return {
        "selected_mode": selected,
        "median_local_throughput_per_s": local_throughput,
        "median_tempo_throughput_per_s": tempo_throughput,
        "tempo_to_local_throughput_ratio": throughput_ratio,
        "median_local_e2e_max_ms": local_e2e,
        "median_tempo_e2e_max_ms": tempo_e2e,
        "tempo_to_local_e2e_ratio": e2e_ratio,
    }


def _run_extra_calibration() -> None:
    output = Path(_argument("--output")).resolve()
    stage = output.parent
    mode_path = stage / "epoch_mode.json"
    if mode_path.exists():
        raise ValueError("stale epoch mode")
    base_url = _argument("--base-url")
    model = _argument("--model")
    served = _argument("--served-model-name")
    request_rate = _argument("--request-rate")
    max_workers = _argument("--max-workers")
    timeout_s = _argument("--timeout-s")
    seed = _argument("--seed") if "--seed" in sys.argv else "20260815"
    warm_root = stage / "same_server_balanced_warm"
    workload_root = stage / "same_server_balanced_warm_workloads"
    calibration_root = stage / "epoch_guard_calibration"
    calibration_root.mkdir()
    sources = {
        "local": workload_root / "01_fixed_local_r0.jsonl",
        "tempo": workload_root / "02_tempo_r0.jsonl",
    }
    artifacts = {
        "local": [warm_root / "01_fixed_local_r0.raw.json"],
        "tempo": [warm_root / "02_tempo_r0.raw.json"],
    }
    order = (("tempo", 1), ("local", 1), ("local", 2), ("tempo", 2))
    for sequence, (arm, replicate) in enumerate(order):
        workload = calibration_root / f"{sequence:02d}_{arm}_r{replicate}.jsonl"
        raw = calibration_root / f"{sequence:02d}_{arm}_r{replicate}.raw.json"
        _rewrite_workload(
            sources[arm], workload,
            f"ssb-{arm}-r0-warm-", f"ssb-{arm}-r{replicate}-warm-")
        command = [
            sys.executable, "-m",
            "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
            "--base-url", base_url,
            "--model", model,
            "--served-model-name", served,
            "--workload", str(workload),
            "--output", str(raw),
            "--mode", "tempo_auto",
            "--run-id", f"epoch-guard-{arm}-r{replicate}",
            "--max-workers", max_workers,
            "--request-rate", request_rate,
            "--timeout-s", timeout_s,
            "--seed", seed,
        ]
        if "--api-key-env" in sys.argv:
            command.extend(("--api-key-env", _argument("--api-key-env")))
        subprocess.run(command, check=True, timeout=1200.0)
        artifacts[arm].append(raw)
    local = [_metrics(path) for path in artifacts["local"]]
    tempo = [_metrics(path) for path in artifacts["tempo"]]
    selection = select_mode(local, tempo)
    value = {
        "schema": MODE_SCHEMA,
        "calibration_replicates_per_candidate": 3,
        "selection_rule": "tempo throughput ratio >=1.005 and e2e-max ratio <=1.0",
        "local": local,
        "tempo": tempo,
        **selection,
    }
    temporary = mode_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(mode_path)


def main() -> int:
    run_id = _argument("--run-id")
    status = serial.main()
    if run_id.endswith("-warmup"):
        _run_extra_calibration()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
