#!/usr/bin/env python3
"""Capture an explicitly rate-based NVML PCIe observation.

NVML exposes PCIe throughput as a recent rate (KB/s), not as a monotonic
byte counter.  This helper therefore emits a separate observation schema and
never lets the result satisfy the strict G1 cumulative-counter gate.  It is
useful for checking whether a foreground/auxiliary intervention changes the
GPU endpoint rate on the same rank.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Sequence

import pynvml


SCHEMA = "tempo-rd-pcie-rate-observation-1"
NVML_WINDOW_NS = 20_000_000


def _device_index() -> int:
    local = os.environ.get("SLURM_LOCALID", os.environ.get("SLURM_PROCID", "0"))
    try:
        value = int(local)
    except ValueError as exc:
        raise ValueError("SLURM_LOCALID must be an integer") from exc
    if value < 0:
        raise ValueError("SLURM_LOCALID must be non-negative")
    return value


def snapshot(*, mode: str, phase: str) -> dict[str, object]:
    if not mode or phase not in {"start", "end", "stream"}:
        raise ValueError("mode/phase are invalid")
    pynvml.nvmlInit()
    try:
        index = _device_index()
        count = pynvml.nvmlDeviceGetCount()
        if index >= count:
            raise ValueError(f"NVML device index {index} is outside count {count}")
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        tx = int(pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES))
        rx = int(pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES))
        pci = pynvml.nvmlDeviceGetPciInfo(handle)
        bus_id = str(getattr(pci, "busId", ""))
        return {
            "schema": SCHEMA,
            "mode": mode,
            "phase": phase,
            "scope": "rank",
            "scope_id": f"rank {os.environ.get('SLURM_PROCID', index)}",
            "device_index": index,
            "pci_bus_id": bus_id,
            "source": "nvml:nvmlDeviceGetPcieThroughput",
            "unit": "KB/s",
            "timestamp_ns": time.monotonic_ns(),
            "tx_kb_per_s": tx,
            "rx_kb_per_s": rx,
            "counter_semantics": "instantaneous_rate_not_cumulative_bytes",
            "causal_ready": False,
        }
    finally:
        pynvml.nvmlShutdown()


def write_snapshot(root: Path, *, mode: str, phase: str) -> Path:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("SLURM_PROCID", os.environ.get("SLURM_LOCALID", "0"))
    path = root / f"pcie_rank_{rank}.{phase}.json"
    path.write_text(json.dumps(snapshot(mode=mode, phase=phase), sort_keys=True) + "\n", encoding="utf-8")
    return path


def integrate_rate_samples(
    samples: Sequence[dict[str, object]],
    *,
    interval_start_ns: int,
    interval_end_ns: int,
    max_gap_ns: int = NVML_WINDOW_NS,
) -> dict[str, object]:
    """Integrate a dense NVML rate trace without pretending it is a counter.

    ``nvmlDeviceGetPcieThroughput`` reports the byte rate over a recent 20ms
    window.  The returned estimate is therefore explicitly a sampled-rate
    integral, not a hardware cumulative byte counter.  We require coverage at
    both interval edges and bound the inter-sample gap; callers must carry the
    returned uncertainty into causal joins rather than feeding it to the
    strict cumulative-counter validator.
    """

    if type(interval_start_ns) is not int or type(interval_end_ns) is not int:
        raise ValueError("integration interval must use integer nanoseconds")
    if interval_start_ns < 0 or interval_end_ns <= interval_start_ns:
        raise ValueError("integration interval is not positive")
    if type(max_gap_ns) is not int or max_gap_ns <= 0:
        raise ValueError("max_gap_ns must be a positive integer")
    if len(samples) < 2:
        raise ValueError("at least two rate samples are required")
    ordered = sorted(samples, key=lambda item: item.get("timestamp_ns", -1))
    if list(ordered) != list(samples):
        raise ValueError("rate samples must already be monotonic")
    previous_ts = None
    max_tx = 0
    max_rx = 0
    tx_bytes = 0
    rx_bytes = 0
    for sample in samples:
        if type(sample) is not dict:
            raise ValueError("rate sample must be an object")
        timestamp = sample.get("timestamp_ns")
        tx = sample.get("tx_kb_per_s")
        rx = sample.get("rx_kb_per_s")
        if (
            type(timestamp) is not int
            or type(tx) is not int
            or type(rx) is not int
            or timestamp < 0
            or tx < 0
            or rx < 0
        ):
            raise ValueError("rate sample timestamp/rates must be non-negative integers")
        if previous_ts is not None:
            gap = timestamp - previous_ts
            if gap <= 0 or gap > max_gap_ns:
                raise ValueError("rate sample gap is outside the bounded interval")
            # Trapezoidal integral.  KB/s * ns * 1000 / 1e9 gives bytes.
            tx_bytes += ((previous_tx + tx) * gap * 1000) // 2_000_000_000
            rx_bytes += ((previous_rx + rx) * gap * 1000) // 2_000_000_000
        previous_ts = timestamp
        previous_tx = tx
        previous_rx = rx
        max_tx = max(max_tx, tx)
        max_rx = max(max_rx, rx)
    first_ts = samples[0]["timestamp_ns"]
    last_ts = samples[-1]["timestamp_ns"]
    if first_ts < interval_start_ns or first_ts - interval_start_ns > NVML_WINDOW_NS:
        raise ValueError("rate trace does not cover the interval start")
    if last_ts > interval_end_ns or interval_end_ns - last_ts > NVML_WINDOW_NS:
        raise ValueError("rate trace does not cover the interval end")
    # The rate API averages a 20ms window; conservatively charge two edge
    # windows plus one full window for each possible sample transition.
    edge_uncertainty = (
        max(max_tx, max_rx) * 1000 * (2 * NVML_WINDOW_NS + (len(samples) - 1) * max_gap_ns)
    ) // 1_000_000_000
    return {
        "tx_estimated_bytes": tx_bytes,
        "rx_estimated_bytes": rx_bytes,
        "max_tx_kb_per_s": max_tx,
        "max_rx_kb_per_s": max_rx,
        "uncertainty_bytes": edge_uncertainty,
        "interval_start_ns": interval_start_ns,
        "interval_end_ns": interval_end_ns,
        "sample_count": len(samples),
        "causal_ready": False,
    }


def write_stream(
    root: Path,
    *,
    mode: str,
    duration_ms: int,
    sample_interval_ms: int = 5,
) -> Path:
    """Capture a bounded JSONL rate trace for a future approved run."""

    if type(duration_ms) is not int or duration_ms <= 0 or duration_ms > 600_000:
        raise ValueError("duration_ms must be in 1..600000")
    if type(sample_interval_ms) is not int or sample_interval_ms <= 0:
        raise ValueError("sample_interval_ms must be positive")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("SLURM_PROCID", os.environ.get("SLURM_LOCALID", "0"))
    path = root / f"pcie_rank_{rank}.stream.jsonl"
    deadline = time.monotonic_ns() + duration_ms * 1_000_000
    with path.open("w", encoding="utf-8") as handle:
        while True:
            value = snapshot(mode=mode, phase="stream")
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()
            if time.monotonic_ns() >= deadline:
                break
            time.sleep(sample_interval_ms / 1000.0)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--phase", choices=("start", "end", "stream"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration-ms", type=int)
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    args = parser.parse_args()
    if args.phase == "stream":
        if args.duration_ms is None:
            parser.error("--duration-ms is required with --phase stream")
        print(write_stream(
            args.output_root,
            mode=args.mode,
            duration_ms=args.duration_ms,
            sample_interval_ms=args.sample_interval_ms,
        ))
    elif args.phase in {"start", "end"}:
        print(write_snapshot(args.output_root, mode=args.mode, phase=args.phase))
    else:
        parser.error("--phase is required")


if __name__ == "__main__":
    main()
