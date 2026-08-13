#!/usr/bin/env python3
"""Execute the frozen two-node raw composite fabric matrix inside Slurm.

The wrapper owns subprocess ordering; it does not infer causal counters or
promote a result.  Every mode is source-bound and is observed by the strict
raw extractor after training and restore.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


MODES = (
    ("fg_only", "none", "foreground"),
    ("open_combined", "datastates", "persistent_endpoint"),
    ("d2h_only", "datastates", "local_sink"),
    ("persist_only", "datastates", "persistent_endpoint"),
    ("combined", "datastates", "persistent_endpoint"),
)


def _srun(*args: str) -> list[str]:
    return [
        "srun", "--overlap", "--time=00:01:00", "--ntasks=8",
        "--ntasks-per-node=4", "--distribution=block:block",
        "--gpus-per-node=4", "--gpu-bind=none",
        "--cpu-bind=verbose,map_ldom:3,2,1,0", "--kill-on-bad-exit=1",
        "--wait=3", "--export=ALL", *args,
    ]


def _run(command: list[str], *, log: Path | None = None, timeout: int | None = None) -> None:
    log_handle = log.open("w", encoding="utf-8") if log else subprocess.DEVNULL
    try:
        subprocess.run(command, check=True, stdout=log_handle, stderr=subprocess.STDOUT, timeout=timeout)
    finally:
        if log:
            log_handle.close()


def _capture(result: Path, mode: str, phase: str, capture_script: Path) -> None:
    mode_root = result / mode
    mode_root.joinpath("domain_counters").mkdir(parents=True, exist_ok=True)
    command = (
        "python " + shlex.quote(str(capture_script)) +
        " --mode " + shlex.quote(mode) + " --phase " + shlex.quote(phase) +
        " --output \"$RESULT_DIR/" + mode +
        "/domain_counters/nic_fabric.rank_${SLURM_PROCID}.json\""
    )
    _run(_srun("bash", "-euo", "pipefail", "-c", command), timeout=60)


def _placement(result: Path) -> None:
    command = (
        "printf 'rank=%s\\nlocal_rank=%s\\nhost=%s\\n' \"$SLURM_PROCID\" "
        "\"$SLURM_LOCALID\" \"$HOSTNAME\" > \"$RESULT_DIR/placement_rank${SLURM_PROCID}.env\""
    )
    _run(_srun("bash", "-c", command), timeout=60)


def _train_args(train: Path, *, policy: str, tier_mode: str, output: Path, checkpoint: Path, restore: bool = False) -> list[str]:
    args = [
        "python", str(train), "--policy", policy, "--tier-mode", tier_mode,
        "--output-dir", str(output), "--checkpoint-dir", str(checkpoint),
        "--steps", "80", "--warmup-steps", "8", "--checkpoint-steps", "16,52",
        "--window-steps", "16", "--layers", "4", "--hidden-size", "2048",
        "--ffn-size", "8192", "--heads", "16", "--sequence-length", "64",
        "--batch-size", "1", "--probe-mb", "64", "--deadline-seconds", "1.0",
        "--datastates-cache-gb", "1", "--seed", "20260811",
    ]
    if restore:
        args.insert(4, "--restore-only")
    return args


def execute(args: argparse.Namespace) -> None:
    result = Path(args.result_dir).resolve()
    result.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    train = Path(args.train).resolve()
    capture = Path(args.capture).resolve()
    observer = Path(args.observer).resolve()
    os.environ["RESULT_DIR"] = str(result)
    os.environ["WORLD_SIZE"] = "8"
    # Emit one node-scoped HSN counter interval per FSDP collective.  The
    # observer labels this as node-slice evidence; it is not promoted to a
    # per-rank or GPU-originated counter by the raw validator.
    os.environ["TEMPO_RD_PHASE_FABRIC_COUNTERS"] = "1"
    _placement(result)
    for mode, policy, endpoint in MODES:
        output = result / mode
        checkpoint = checkpoint_root / mode
        output.mkdir(parents=True, exist_ok=True)
        checkpoint.mkdir(parents=True, exist_ok=True)
        os.environ["TEMPO_RD_TIER_MODE"] = mode
        os.environ["TEMPO_RD_ENDPOINT"] = endpoint
        if mode == "d2h_only":
            sink = Path(os.environ.get("D2H_SINK_ROOT", "/tmp")) / mode
            sink.mkdir(parents=True, exist_ok=True)
            os.environ["TEMPO_RD_LOCAL_SINK_ROOT"] = str(sink)
        else:
            os.environ.pop("TEMPO_RD_LOCAL_SINK_ROOT", None)
        _capture(result, mode, "start", capture)
        _run(_srun(*_train_args(train, policy=policy, tier_mode=mode, output=output, checkpoint=checkpoint)), log=output / "train_srun.log", timeout=40)
        _capture(result, mode, "end", capture)
        (output / "raw_status.env").write_text("mode=%s\nstatus=complete\n" % mode, encoding="utf-8")
    for mode in ("open_combined", "persist_only", "combined"):
        output = result / mode
        checkpoint = checkpoint_root / mode
        _run(_srun(*_train_args(train, policy="datastates", tier_mode=mode, output=output, checkpoint=checkpoint, restore=True)), log=output / "restore_srun.log", timeout=30)
    for mode, _, _ in MODES:
        _run([sys.executable, str(observer), "--root", str(result), "--policy", mode, "--output", str(result / ("fabric_observation_%s.json" % mode))])
    (result / "execution_status.env").write_text("schema_version=tempo-rd-g2-fabric-raw-execution-1\nstatus=raw_complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--observer", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("tempo-rd-g2-fabric-raw; nodes=2; world_size=8; modes=" + ",".join(mode for mode, _, _ in MODES))
        return
    execute(args)


if __name__ == "__main__":
    main()
