#!/usr/bin/env python3
"""Capture only counters that are actually exposed by a G1 compute node.

This helper is intentionally conservative.  Linux ``hsn*`` interface byte
statistics are cumulative device counters and can support the NIC/CXI domain
when sampled before and after one isolated mode.  ``nvidia-smi`` topology,
``numastat``, ``cxi_stat`` rates, and Lustre RPC histograms are recorded as
capability/path context only; they are never converted into byte counters.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


SCHEMA = "tempo-rd-domain-counter-record-2"
CAPABILITY_SCHEMA = "tempo-rd-domain-capability-1"


def _numeric_bytes(value: Any) -> int:
    """Parse the byte total emitted by a JSON Nsight report."""

    if type(value) is int and value >= 0:
        return value
    if type(value) is float and value >= 0 and value.is_integer():
        return int(value)
    if not isinstance(value, str):
        raise ValueError("Nsight byte total is not numeric")
    text = value.strip().replace(",", "")
    units = (("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2),
             ("KiB", 1024), ("TB", 1000**4), ("GB", 1000**3),
             ("MB", 1000**2), ("KB", 1000), ("B", 1))
    for suffix, scale in units:
        if text.endswith(suffix):
            number = text[:-len(suffix)].strip()
            parsed = float(number)
            if parsed < 0 or not parsed.is_integer() and scale == 1:
                raise ValueError("invalid Nsight byte total")
            return int(round(parsed * scale))
    parsed = float(text)
    if parsed < 0 or not parsed.is_integer():
        raise ValueError("invalid Nsight byte total")
    return int(parsed)


def _row_byte_total(row: dict[str, Any]) -> int:
    """Return one unambiguous total from an Nsight row.

    Nsight releases use either ``Total`` or ``Bytes``.  A row containing both
    is accepted only when they agree; silently preferring one would make a
    malformed/hand-edited report look like a GPU counter.
    """

    values = []
    for key in ("Total", "total", "Bytes", "bytes"):
        if key in row and row[key] is not None:
            values.append(_numeric_bytes(row[key]))
    # Nsight Systems 25.x emits the report column as ``Total (MB)`` rather
    # than a unit-bearing ``Total`` value.  Keep the unit conversion explicit
    # and reject disagreement with any parallel Total/Bytes column.
    for key, unit in (("Total (MB)", "MB"), ("total (MB)", "MB"),
                      ("Total (MiB)", "MiB"), ("total (MiB)", "MiB")):
        if key in row and row[key] is not None:
            raw = row[key]
            if type(raw) not in (int, float) or raw < 0:
                raise ValueError("invalid Nsight byte total")
            values.append(_numeric_bytes(f"{raw} {unit}"))
    if not values:
        raise ValueError("Nsight Device-to-Host row has no total bytes")
    if len(set(values)) != 1:
        raise ValueError("Nsight Device-to-Host row has ambiguous total bytes")
    return values[0]


def _row_time_total(row: dict[str, Any]) -> int:
    """Parse Nsight's CUDA memcpy duration column in nanoseconds."""

    values = []
    for key in ("Total Time (ns)", "total time (ns)", "Total Time", "total_time_ns"):
        if key in row and row[key] is not None:
            raw = row[key]
            if type(raw) not in (int, float) or raw < 0 or not float(raw).is_integer():
                raise ValueError("invalid Nsight busy time")
            values.append(int(raw))
    if not values:
        raise ValueError("Nsight D2H time row has no total duration")
    if len(set(values)) != 1:
        raise ValueError("Nsight D2H time row has ambiguous duration")
    return values[0]


def parse_nsys_gpu_mem_report(
    report: Path,
    *,
    mode: str,
    timestamp_ns: int | None = None,
    busy_report: Path | None = None,
) -> dict[str, Any]:
    """Convert an exact Nsight CUDA MemOps JSON report to a GPU counter.

    The report must contain an unambiguous Device-to-Host operation row.  A
    report containing only aggregate memory operations is rejected because it
    cannot identify the D2H stage.
    """

    raw = json.loads(report.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("rows", raw.get("data", [])) if isinstance(raw, dict) else []
    if not isinstance(rows, list):
        raise ValueError("Nsight GPU memory report rows are not a list")
    matches: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        operation = str(row.get("Operation", row.get("operation", ""))).lower()
        if any(token in operation for token in ("dtoh", "device to host", "device-to-host", "memcpy dtoh")):
            matches.append(row)
    if not matches:
        raise ValueError("Nsight report has no unambiguous Device-to-Host row")
    total = 0
    for row in matches:
        total += _row_byte_total(row)
    busy_ns = 0
    if busy_report is not None:
        busy_raw = json.loads(Path(busy_report).read_text(encoding="utf-8"))
        busy_rows = busy_raw if isinstance(busy_raw, list) else busy_raw.get("rows", busy_raw.get("data", [])) if isinstance(busy_raw, dict) else []
        busy_matches = [
            row for row in busy_rows if isinstance(row, dict)
            and any(token in str(row.get("Operation", row.get("operation", ""))).lower()
                    for token in ("dtoh", "device to host", "device-to-host", "memcpy dtoh"))
        ]
        if not busy_matches:
            raise ValueError("Nsight time report has no unambiguous Device-to-Host row")
        busy_ns = sum(_row_time_total(row) for row in busy_matches)
    stamp = int(timestamp_ns if timestamp_ns is not None else time.monotonic_ns())
    source = "nsys:cuda_gpu_mem_size_sum:Device-to-Host"
    scope_id = os.environ.get("SLURM_PROCID", "rank-unknown")
    if "_rank_" in mode:
        suffix = mode.rsplit("_rank_", 1)[1]
        if suffix.isdigit():
            scope_id = f"rank {int(suffix)}"
    record = _counter_record(mode, [
        {"sample_id": f"{mode}-start-{stamp}", "source": source,
         "timestamp_ns": max(0, stamp - 1), "cumulative_bytes": 0,
         "cumulative_busy_ns": 0, "support": "supported"},
        {"sample_id": f"{mode}-end-{stamp}", "source": source,
         "timestamp_ns": stamp, "cumulative_bytes": total,
         "cumulative_busy_ns": busy_ns, "support": "supported"},
    ], source, domain="gpu_local", path_evidence="gpu_hbm_copy_engine",
       counter_family="gpu_copy_engine_bytes", scope="rank",
       scope_id=scope_id)
    # Nsight is deliberately a separate diagnostic path: its profiling
    # overhead makes the record ineligible for the timed G1 causal matrix.
    # A size-only Nsight report is diagnostic-only and is not eligible for the
    # G1 causal validator.  When a matching busy-time report is supplied, the
    # exact domain-counter schema is emitted so the record can be checked as
    # a real rank-bound interval; the caller still keeps the Nsight source
    # provenance and timed-metrics eligibility boundary explicit.
    if busy_report is None:
        record["diagnostic_only"] = True
    return record


def _command(*argv: str) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=5)
        return {"argv": list(argv), "returncode": result.returncode,
                "stdout": result.stdout[-8192:], "stderr": result.stderr[-8192:]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"argv": list(argv), "error": str(exc)}


def _hsn_snapshot() -> tuple[int, dict[str, int], str]:
    values: dict[str, int] = {}
    paths = sorted(Path("/sys/class/net").glob("hsn*/statistics"))
    for directory in paths:
        name = directory.parent.name
        for field in ("rx_bytes", "tx_bytes", "rx_packets", "tx_packets"):
            path = directory / field
            try:
                values[f"{name}.{field}"] = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
    total_bytes = sum(value for key, value in values.items() if key.endswith((".rx_bytes", ".tx_bytes")))
    return time.monotonic_ns(), values, f"sysfs:/sys/class/net/hsn*/statistics;host_device_sum"


def _counter_record(
    mode: str,
    samples: list[dict[str, Any]],
    source: str,
    *,
    domain: str = "nic_fabric",
    path_evidence: str = "cxi_nic_injection",
    counter_family: str = "cxi_tx_rx_bytes",
    scope: str = "host",
    scope_id: str = "",
) -> dict[str, Any]:
    if not isinstance(scope, str) or not scope:
        raise ValueError("counter scope is required")
    if not isinstance(scope_id, str) or not scope_id:
        scope_id = os.environ.get("HOSTNAME", "host-unknown")
    return {
        "schema": SCHEMA,
        "mode": mode,
        "domain": domain,
        "scope": scope,
        "scope_id": scope_id,
        "intervention_id": mode,
        "path_evidence": path_evidence,
        "counter_family": counter_family,
        "path_status": "observed",
        "counter_support": "supported",
        "source": source,
        "hardware_counter": True,
        "samples": samples,
    }


def _capability_probe() -> dict[str, Any]:
    """Describe candidate counter interfaces without relabeling them usable.

    A path appearing here is only a probe result.  The G1 readiness builder
    still requires a source-bound monotonic byte/busy series and will reject
    gauges, rates, device-total values, or diagnostic-only records.
    """

    pcie_candidates: list[str] = []
    for path in sorted(Path("/sys/bus/pci/devices").glob("*/statistics/*")):
        if path.name in {"rx_bytes", "tx_bytes", "bytes", "read_bytes", "write_bytes"}:
            pcie_candidates.append(str(path))
    numa_candidates = sorted(
        str(path) for path in Path("/sys/devices/system/node").glob("node*/numastat")
    )
    lustre_candidates = sorted(
        str(path)
        for path in Path("/sys/fs/lustre/osc").glob("*")
        if path.is_dir()
        for name in ("stats", "rpc_stats", "bytes", "read_bytes", "write_bytes")
        if (path / name).is_file()
    )
    nsys_candidates = [
        candidate for candidate in (
            "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Systems/bin/nsys",
            "/usr/local/cuda/bin/nsys",
        ) if Path(candidate).is_file()
    ]
    return {
        "pcie_byte_counter_candidates": pcie_candidates,
        "host_numa_counter_candidates": numa_candidates,
        "lustre_cumulative_counter_candidates": lustre_candidates,
        "nsys_diagnostic_binaries": nsys_candidates,
        "interpretation": "capability_probe_only; no candidate is causal evidence",
    }


def capture_mode(mode: str, output: Path, phase: str) -> None:
    timestamp_ns, values, source = _hsn_snapshot()
    total_bytes = sum(value for key, value in values.items() if key.endswith((".rx_bytes", ".tx_bytes")))
    sample = {
        "sample_id": f"{mode}-{phase}-{timestamp_ns}",
        "source": source,
        "timestamp_ns": timestamp_ns,
        "cumulative_bytes": total_bytes,
        # The exposed interface counters have no device-busy clock.  Zero is
        # explicit and keeps the byte series honest; rate derivation is None.
        "cumulative_busy_ns": 0,
        "support": "supported",
        "interface_bytes": values,
    }
    if phase == "start":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"mode": mode, "samples": [sample]}, sort_keys=True) + "\n", encoding="utf-8")
        return
    prior = json.loads(output.read_text(encoding="utf-8"))
    prior_samples = prior.get("samples") if isinstance(prior, dict) else None
    if not isinstance(prior_samples, list) or len(prior_samples) != 1:
        raise ValueError("counter start record is missing or malformed")
    samples = []
    for item in prior_samples + [sample]:
        samples.append({key: item[key] for key in (
            "sample_id", "source", "timestamp_ns", "cumulative_bytes",
            "cumulative_busy_ns", "support")})
    output.write_text(
        json.dumps(
            _counter_record(
                mode,
                samples,
                source,
                scope="host",
                scope_id=os.environ.get("HOSTNAME", "host-unknown"),
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_capabilities(output: Path) -> None:
    timestamp_ns, values, source = _hsn_snapshot()
    records = {
        "nic_fabric": {
            "path_status": "observed" if values else "unknown",
            # The sysfs HSN total is a real device counter, but it is host
            # aggregate and therefore cannot support a rank/slice causal
            # claim.  Keep the distinction explicit instead of letting a
            # later validator reinterpret a generic "supported" label.
            "counter_support": "supported" if values else "not_supported",
            "causal_scope_support": "not_supported",
            "diagnostic_only": True,
            "scope": "host",
            "counter_family": "cxi_tx_rx_bytes",
            "source": source,
            "hardware_counter": bool(values),
            "sample_timestamp_ns": timestamp_ns,
            "interfaces": sorted({key.split(".", 1)[0] for key in values}),
        },
        "gpu_local": {"path_status": "observed", "counter_support": "not_collected",
                      "reason": "nvidia-smi exposes topology/utilization but no copy-byte series"},
        "pcie_host": {"path_status": "observed", "counter_support": "not_collected",
                      "reason": "PCIe link metadata is available but no byte counter was exposed"},
        "host_numa": {"path_status": "observed", "counter_support": "not_collected",
                      "reason": "numastat is placement/pressure context, not a traffic-byte counter"},
        "slingshot_fabric": {"path_status": "observed", "counter_support": "not_supported",
                             "reason": "cxi_stat reports rates/codewords, not a monotonic byte counter"},
        "persistent_endpoint": {"path_status": "observed", "counter_support": "not_supported",
                                 "reason": "Lustre OSC exposes RPC histograms here, not exact cumulative byte totals"},
    }
    payload = {
        "schema": CAPABILITY_SCHEMA,
        "host": os.environ.get("HOSTNAME", ""),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "captured_monotonic_ns": timestamp_ns,
        "domains": records,
        "context": {
            "nvidia_smi": _command("nvidia-smi", "--query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader"),
            "nvidia_smi_dmon_pcie": _command("nvidia-smi", "dmon", "-s", "p", "-c", "1"),
            "cxi_stat": _command("cxi_stat", "-l"),
            "fi_info": _command("fi_info", "-p", "cxi", "-t", "FI_EP_RDM"),
            "numactl": _command("numactl", "--show"),
            "counter_probe": _capability_probe(),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--phase", choices=("start", "end"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capabilities", type=Path)
    parser.add_argument("--nsys-report", type=Path)
    parser.add_argument("--nsys-time-report", type=Path)
    args = parser.parse_args()
    if args.capabilities is not None:
        write_capabilities(args.capabilities)
    if args.nsys_report is not None:
        if args.mode is None or args.output is None:
            raise SystemExit("--nsys-report requires --mode and --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            parse_nsys_gpu_mem_report(
                args.nsys_report, mode=args.mode, busy_report=args.nsys_time_report
            ),
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode is not None and args.nsys_report is None:
        if args.output is None:
            raise SystemExit("--mode requires --output")
        if args.phase is None:
            raise SystemExit("--mode requires --phase")
        capture_mode(args.mode, args.output, args.phase)


if __name__ == "__main__":
    main()
