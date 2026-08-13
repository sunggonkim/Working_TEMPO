#!/usr/bin/env python3
"""Run a foreground training command beside the host-pressure placebo.

The wrapper is a command-level adapter for the future G1 allocation.  It
starts one rank-local pressure worker, runs the supplied foreground command
unchanged, then asks the worker to finish and publishes its counter record.
It never submits Slurm work and does not alter the foreground command's
arguments or checkpoint policy.  A nonzero worker result is fatal: a missing
or partial placebo must not be silently treated as a clean control.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def _infer_numa_node() -> int:
    explicit = os.environ.get("TEMPO_RD_NUMA_NODE")
    if explicit is not None:
        value = int(explicit)
        if value < 0:
            raise ValueError("TEMPO_RD_NUMA_NODE must be non-negative")
        return value
    mapping = os.environ.get("PERLMUTTER_CPU_LDOM_MAP", "")
    local_rank = os.environ.get("SLURM_LOCALID")
    if not mapping or local_rank is None:
        raise RuntimeError("NUMA node must be explicit or provided by Slurm local-rank mapping")
    values = [item.strip() for item in mapping.split(",")]
    index = int(local_rank)
    if index < 0 or index >= len(values) or not values[index].isdigit():
        raise RuntimeError("invalid local-rank NUMA mapping")
    return int(values[index])


def run_with_pressure(
    *,
    train_script: Path,
    pressure_output: Path,
    train_args: list[str],
    rank: int,
    world_size: int = 4,
    numa_node: int | None = None,
    buffer_mib: int = 64,
    pressure_duration_ms: int = 30_000,
    sample_ms: int = 20,
) -> int:
    """Run foreground training and return its exact exit status."""

    train_script = Path(train_script).resolve()
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    if type(rank) is not int or rank < 0 or rank >= world_size:
        raise ValueError("rank must be within world_size")
    if world_size != 4:
        raise ValueError("G1 host-pressure wrapper is fixed to four ranks")
    if type(buffer_mib) is not int or buffer_mib < 64:
        raise ValueError("buffer_mib must be at least 64")
    if type(pressure_duration_ms) is not int or pressure_duration_ms <= 0:
        raise ValueError("pressure_duration_ms must be positive")
    if type(sample_ms) is not int or sample_ms <= 0 or sample_ms > pressure_duration_ms:
        raise ValueError("sample_ms must be positive and within the pressure duration")
    if not train_args:
        raise ValueError("foreground train arguments must be non-empty")

    numa = _infer_numa_node() if numa_node is None else numa_node
    if type(numa) is not int or numa < 0:
        raise ValueError("numa_node must be a non-negative int")
    # ``%r`` is expanded by this rank, not by the submit/login shell.  This
    # keeps one deterministic command line while ensuring every rank writes a
    # distinct local evidence record.
    pressure_output = Path(str(pressure_output).replace("%r", str(rank))).resolve()
    pressure_output.parent.mkdir(parents=True, exist_ok=True)
    stop_file = pressure_output.with_name(f".{pressure_output.name}.stop-{os.getpid()}")
    helper = Path(__file__).with_name("host_pressure_placebo.py").resolve()
    worker = subprocess.Popen(
        [
            sys.executable,
            str(helper),
            "--output",
            str(pressure_output),
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--numa-node",
            str(numa),
            "--buffer-mib",
            str(buffer_mib),
            "--duration-ms",
            str(pressure_duration_ms),
            "--sample-ms",
            str(sample_ms),
            "--parent-pid",
            str(os.getpid()),
            "--stop-file",
            str(stop_file),
        ],
        env=dict(os.environ),
    )
    train_rc = 1
    worker_rc: int | None = None
    try:
        completed = subprocess.run([sys.executable, str(train_script), *train_args], env=dict(os.environ))
        train_rc = completed.returncode
    finally:
        try:
            stop_file.touch()
            worker_rc = worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker_rc = worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker_rc = worker.wait(timeout=3)
        finally:
            stop_file.unlink(missing_ok=True)
    if train_rc != 0:
        return train_rc
    if worker_rc != 0 or not pressure_output.is_file():
        return 3
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--pressure-output", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--numa-node", type=int)
    parser.add_argument("--buffer-mib", type=int, default=64)
    parser.add_argument("--pressure-duration-ms", type=int, default=30_000)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    if args.rank is None:
        local_rank = os.environ.get("SLURM_PROCID")
        if local_rank is None:
            parser.error("--rank is required outside a Slurm rank")
        args.rank = int(local_rank)
    return args


def main() -> None:
    args = _parse_args()
    raise SystemExit(
        run_with_pressure(
            train_script=args.train_script,
            pressure_output=args.pressure_output,
            train_args=args.train_args,
            rank=args.rank,
            world_size=args.world_size,
            numa_node=args.numa_node,
            buffer_mib=args.buffer_mib,
            pressure_duration_ms=args.pressure_duration_ms,
            sample_ms=args.sample_ms,
        )
    )


if __name__ == "__main__":
    main()
