#!/usr/bin/env python3
"""Calibrate local, policy8, and LMCache-compatible remote before measurement."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_epoch_guard_client_v249 as prior
from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_serial_lm_warm_v230 as serial
from eval.sota_4node.tempo_pd_same_server_tri_epoch_guard_router_v255 import MODE_SCHEMA


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _median(rows: list[dict[str, float]], key: str) -> float:
    return statistics.median(row[key] for row in rows)


def select_mode(arms: dict[str, list[dict[str, float]]]) -> dict:
    if set(arms) != {"local", "tempo", "remote"}:
        raise ValueError("exact tri-arm calibration required")
    if any(len(rows) != 3 for rows in arms.values()):
        raise ValueError("exactly three replicates per arm required")
    medians = {
        arm: {
            "throughput_per_s": _median(rows, "throughput_per_s"),
            "e2e_max_ms": _median(rows, "e2e_max_ms"),
        }
        for arm, rows in arms.items()
    }
    tempo = medians["tempo"]
    local = medians["local"]
    remote = medians["remote"]
    policy8_clear = (
        tempo["throughput_per_s"] >= 1.005 * max(
            local["throughput_per_s"], remote["throughput_per_s"])
        and tempo["e2e_max_ms"] <= min(
            local["e2e_max_ms"], remote["e2e_max_ms"])
    )
    local_clear = (
        local["throughput_per_s"] >= 1.005 * remote["throughput_per_s"]
        and local["e2e_max_ms"] <= remote["e2e_max_ms"]
    )
    selected = "policy8" if policy8_clear else (
        "fixed_local" if local_clear else "lmcache_remote")
    return {
        "selected_mode": selected,
        "median_metrics": medians,
        "policy8_clear_win": policy8_clear,
        "local_clear_win": local_clear,
    }


def _rewrite(source: Path, output: Path, old: str, new: str) -> None:
    prior._rewrite_workload(source, output, old, new)


def _run_extra_calibration() -> None:
    output = Path(_argument("--output")).resolve()
    stage = output.parent
    mode_path = stage / "epoch_mode.json"
    if mode_path.exists():
        raise ValueError("stale tri-epoch mode")
    calibration_root = stage / "tri_epoch_guard_calibration"
    calibration_root.mkdir()
    warm_root = stage / "same_server_balanced_warm"
    workload_root = stage / "same_server_balanced_warm_workloads"
    sources = {
        "local": workload_root / "01_fixed_local_r0.jsonl",
        "tempo": workload_root / "02_tempo_r0.jsonl",
        "remote": workload_root / "00_lmcache_remote_r0.jsonl",
    }
    artifacts = {
        "local": [warm_root / "01_fixed_local_r0.raw.json"],
        "tempo": [warm_root / "02_tempo_r0.raw.json"],
        "remote": [warm_root / "00_lmcache_remote_r0.raw.json"],
    }
    prefixes = {"local": "local", "tempo": "tempo", "remote": "remote"}
    order = (("tempo", 1), ("remote", 1), ("local", 1),
             ("local", 2), ("remote", 2), ("tempo", 2))
    for sequence, (arm, replicate) in enumerate(order):
        workload = calibration_root / f"{sequence:02d}_{arm}_r{replicate}.jsonl"
        raw = calibration_root / f"{sequence:02d}_{arm}_r{replicate}.raw.json"
        marker = prefixes[arm]
        _rewrite(sources[arm], workload,
                 f"ssb-{marker}-r0-warm-", f"ssb-{marker}-r{replicate}-warm-")
        command = [
            sys.executable, "-m",
            "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_v38",
            "--base-url", _argument("--base-url"),
            "--model", _argument("--model"),
            "--served-model-name", _argument("--served-model-name"),
            "--workload", str(workload), "--output", str(raw),
            "--mode", "tempo_auto", "--run-id", f"tri-epoch-{arm}-r{replicate}",
            "--max-workers", _argument("--max-workers"),
            "--request-rate", _argument("--request-rate"),
            "--timeout-s", _argument("--timeout-s"),
            "--seed", _argument("--seed") if "--seed" in sys.argv else "20260815",
        ]
        if "--api-key-env" in sys.argv:
            command.extend(("--api-key-env", _argument("--api-key-env")))
        subprocess.run(command, check=True, timeout=1200.0)
        artifacts[arm].append(raw)
    arms = {arm: [prior._metrics(path) for path in paths]
            for arm, paths in artifacts.items()}
    selection = select_mode(arms)
    value = {
        "schema": MODE_SCHEMA,
        "calibration_replicates_per_candidate": 3,
        "selection_rule": (
            "policy8 must beat both alternatives by >=0.5% throughput and no "
            "e2e-max regression; local must clear remote likewise; otherwise "
            "use lmcache-compatible remote"
        ),
        "arms": arms,
        **selection,
    }
    temporary = mode_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(mode_path)


def main() -> int:
    status = serial.main()
    if _argument("--run-id").endswith("-warmup"):
        _run_extra_calibration()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
