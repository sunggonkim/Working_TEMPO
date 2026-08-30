"""Summarize achieved CXI rates and synchronized Cassini evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA = "tempo-cxi-fabric-ladder-summary-v2"
TRAFFIC_SCHEMA = "tempo-cxi-background-traffic-3"
WINDOW_SCHEMA = "tempo-cassini-fabric-window-v1"


def _json_record(path: Path, schema: str) -> dict[str, Any]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == schema:
            records.append(value)
    if len(records) != 1:
        raise ValueError(f"{path} has {len(records)} {schema} records")
    return records[0]


def _maximum(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _decoder_nodes(pattern: str) -> tuple[int, ...]:
    return (3,) if pattern == "pd-3p1d-incast" else (1, 3)


def _profile(path: Path) -> dict[str, Any]:
    traffic_path = path / "traffic.log"
    traffic_text = traffic_path.read_text(encoding="utf-8", errors="replace")
    traffic = _json_record(traffic_path, TRAFFIC_SCHEMA)
    windows = [
        _json_record(item, WINDOW_SCHEMA)
        for item in sorted(path.glob("cassini-*.jsonl"))
    ]
    if len(windows) != 4 or [item["node_index"] for item in windows] != list(range(4)):
        raise ValueError(f"{path} lacks four ordered Cassini receipts")
    samples = [item.get("sample") for item in windows]
    valid_samples = [
        item for item in samples
        if isinstance(item, dict) and item.get("valid") is True
    ]
    signals = [item.get("signals", {}) for item in valid_samples]
    decoder_nodes = _decoder_nodes(traffic["pattern"])
    node_received = traffic["node_received_gbps"]
    transport_faults = sum(
        int(signal.get(name) or 0)
        for signal in signals
        for name in ("resource_nacks", "retries", "timeouts")
    )
    timeout_matches = re.findall(
        r"MPICH Slingshot Network Summary: ([0-9]+) network timeouts?",
        traffic_text,
    )
    mpich_network_timeouts = (
        sum(int(value) for value in timeout_matches)
        if timeout_matches else None
    )
    maximum_node_ingress = max(node_received)
    maximum_node_egress = max(traffic["node_sent_gbps"])
    return {
        "profile": path.name,
        "pattern": traffic["pattern"],
        "message_bytes": traffic["message_bytes"],
        "inflight": traffic["inflight"],
        "duty_cycle": traffic["duty_cycle"],
        "elapsed_s": traffic["elapsed_s"],
        "correctness": traffic["correctness"],
        "aggregate_sent_gbps": traffic["aggregate_sent_gbps"],
        "aggregate_received_gbps": traffic["aggregate_received_gbps"],
        "node_sent_gbps": traffic["node_sent_gbps"],
        "node_received_gbps": node_received,
        "decoder_ingress_gbps": sum(node_received[index] for index in decoder_nodes),
        "maximum_decoder_node_ingress_gbps": max(
            node_received[index] for index in decoder_nodes),
        "maximum_node_payload_ingress_fraction_of_800gbps": (
            maximum_node_ingress / 800.0),
        "maximum_node_payload_egress_fraction_of_800gbps": (
            maximum_node_egress / 800.0),
        "cassini_valid_nodes": len(valid_samples),
        "cassini_support": (
            valid_samples[0].get("support") if valid_samples else None),
        "oxe_channel_active_fraction_max": _maximum([
            signal.get(
                "oxe_channel_active_fraction_max",
                signal.get("link_utilization_fraction_max"),
            )
            for signal in signals
        ]),
        "oxe_channel_active_fraction_mean": _mean([
            signal.get(
                "oxe_channel_active_fraction_mean",
                signal.get("link_utilization_fraction_mean"),
            )
            for signal in signals
        ]),
        "oxe_fraction_is_not_aggregate_link_utilization": True,
        "rx_pause_fraction_max": _maximum([
            signal.get("rx_pause_fraction_max") for signal in signals
        ]),
        "tx_pause_fraction_max": _maximum([
            signal.get("tx_pause_fraction_max") for signal in signals
        ]),
        "host_posted_cycles_per_packet_max": _maximum([
            signal.get("host_posted_cycles_per_packet_max")
            for signal in signals
        ]),
        "receive_overflow_fraction_max": _maximum([
            signal.get("receive_overflow_fraction_max") for signal in signals
        ]),
        "ecn_fraction_max": _maximum([
            signal.get("ecn_fraction_max") for signal in signals
        ]),
        "tx_packets_per_s": sum(
            float(signal.get("tx_packets_per_s") or 0.0)
            for signal in signals
        ),
        "rx_packets_per_s": sum(
            float(signal.get("rx_packets_per_s") or 0.0)
            for signal in signals
        ),
        "transport_faults": transport_faults,
        "mpich_network_timeouts": mpich_network_timeouts,
    }


def _scaling_receipts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    receipts = []
    for pattern in sorted({row["pattern"] for row in rows}):
        candidates = sorted(
            (row for row in rows if row["pattern"] == pattern),
            key=lambda row: row["message_bytes"] * row["inflight"],
        )
        for lower, upper in zip(candidates, candidates[1:]):
            offered_ratio = (
                upper["message_bytes"] * upper["inflight"]
                / (lower["message_bytes"] * lower["inflight"])
            )
            if offered_ratio <= 1.0:
                continue
            achieved_ratio = (
                upper["decoder_ingress_gbps"]
                / max(lower["decoder_ingress_gbps"], 1.0e-12)
            )
            receipts.append({
                "pattern": pattern,
                "lower_profile": lower["profile"],
                "upper_profile": upper["profile"],
                "offered_parallelism_ratio": offered_ratio,
                "achieved_ingress_ratio": achieved_ratio,
                "incremental_scaling_efficiency": (
                    (achieved_ratio - 1.0) / (offered_ratio - 1.0)
                ),
            })
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    if not root.is_dir():
        parser.error("result_dir must exist")
    profile_dirs = sorted(
        item for item in root.iterdir()
        if item.is_dir() and (item / "traffic.log").is_file()
    )
    if not profile_dirs:
        parser.error("no fabric ladder profiles found")
    rows = [_profile(path) for path in profile_dirs]
    scaling = _scaling_receipts(rows)
    eligible = [
        row for row in rows
        if row["correctness"] is True
        and row["cassini_valid_nodes"] == 4
        and row["transport_faults"] == 0
        and row["mpich_network_timeouts"] == 0
        and row["pattern"] != "pairwise-bidir"
    ]
    recommended = max(
        eligible,
        key=lambda row: (
            row["pattern"] == "pd-3p1d-incast",
            row["inflight"],
            max(
                row["rx_pause_fraction_max"] or 0.0,
                row["tx_pause_fraction_max"] or 0.0,
            ),
            row["maximum_decoder_node_ingress_gbps"],
        ),
        default=None,
    )
    low_efficiency = [
        item for item in scaling
        if item["incremental_scaling_efficiency"] < 0.25
    ]
    physical_link_saturation = any(
        max(
            row["maximum_node_payload_ingress_fraction_of_800gbps"],
            row["maximum_node_payload_egress_fraction_of_800gbps"],
        ) >= 0.80
        for row in rows
    )
    pause_backpressure = any(
        max(
            row["rx_pause_fraction_max"] or 0.0,
            row["tx_pause_fraction_max"] or 0.0,
        ) >= 0.10
        for row in rows
    )
    host_backpressure = any(
        (row["host_posted_cycles_per_packet_max"] or 0.0) > 3.0
        for row in rows
    )
    backpressure_bottleneck = bool(
        low_efficiency and pause_backpressure and host_backpressure)
    result = {
        "schema": SCHEMA,
        "profile_count": len(rows),
        "profiles": rows,
        "scaling_receipts": scaling,
        "fabric_bottleneck_confirmed": backpressure_bottleneck,
        "fabric_backpressure_bottleneck_confirmed": backpressure_bottleneck,
        "physical_link_saturation_confirmed": physical_link_saturation,
        "bottleneck_class": (
            "receiver_endpoint_and_cassini_flow_control"
            if backpressure_bottleneck and not physical_link_saturation
            else "physical_link_saturation"
            if physical_link_saturation
            else "not_confirmed"
        ),
        "fabric_bottleneck_gate": {
            "physical_payload_fraction_ge_0_80": physical_link_saturation,
            "sublinear_scaling_observed": bool(low_efficiency),
            "pause_fraction_ge_0_10": pause_backpressure,
            "host_blocked_cycles_per_packet_gt_3": host_backpressure,
            "criterion": (
                "incremental_scaling_efficiency<0.25 and pause>=0.10 and "
                "host posted blocked cycles/packet>3"),
            "physical_saturation_criterion": (
                "maximum measured application payload ingress or egress "
                ">=80% of four 200-Gb/s NICs"),
        },
        "recommended_profile": (
            recommended["profile"] if recommended is not None else None),
        "recommendation_basis": (
            "highest-concurrency fault-free 3P-to-1D incast at the measured "
            "throughput plateau; intended to expose global pair/routing value"
            if recommended is not None else "no eligible profile"),
    }
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for row in rows for value in row.values()
    ):
        raise ValueError("summary contains non-finite values")
    output = root / "fabric_ladder_summary_v2.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
