#!/usr/bin/env python3
"""Summarize fixed-arm P/D endpoint evidence without naming a switch bottleneck."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from eval.sota_4node import run_tempo_pd_contention_fixed_client as client
from eval.sota_4node.tempo_pd_endpoint_probe import (
    validate_vllm_endpoint_cumulative,
)


SCHEMA = "tempo-pd-endpoint-characterization-v1"
_FRACTION_SIGNALS = (
    "rx_pause_fraction_max",
    "tx_pause_fraction_max",
    "receive_overflow_fraction_max",
    "ecn_fraction_max",
)
_FAULT_SIGNALS = ("resource_nacks", "retries", "timeouts")
_PACKET_SIGNALS = ("tx_packets", "rx_packets")
_HOST_SIGNALS = (
    "host_posted_cycles_per_packet_max",
    "host_nonposted_cycles_per_packet_max",
)
_CLIENT_SCHEMAS = {
    "tempo-pd-contention-fixed-client-v5",
    "tempo-pd-contention-fixed-client-v6",
    "tempo-pd-contention-fixed-client-v7",
}
_BLOCK_SCHEMAS = {
    "tempo-pd-contention-fixed-block-v5",
    "tempo-pd-contention-fixed-block-v6",
    "tempo-pd-contention-fixed-block-v7",
}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _percentile(values: list[float], fraction: float) -> float:
    _require(bool(values), "percentile input is empty")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _endpoint_index(stage: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        row["probe"]["endpoint"]["endpoint_id"]: row["probe"]
        for row in stage["snapshots"]
    }


def _cassini_block(
    midpoint: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    result = {}
    for endpoint_id in sorted(midpoint):
        samples = [midpoint[endpoint_id]["cassini"],
                   after[endpoint_id]["cassini"]]
        _require(all(sample["valid"] is True for sample in samples),
                 "measured Cassini sample is invalid")
        signals = [sample["signals"] for sample in samples]
        supported_values = {
            name: [row[name] for row in signals if row[name] is not None]
            for name in (*_FRACTION_SIGNALS, *_FAULT_SIGNALS,
                         *_PACKET_SIGNALS, *_HOST_SIGNALS)
        }
        result[endpoint_id] = {
            "windows_ms": [sample["window_ms"] for sample in samples],
            "fraction_max": {
                name: (max(values) if values else None)
                for name, values in supported_values.items()
                if name in _FRACTION_SIGNALS
            },
            "fault_count": {
                name: (sum(values) if values else None)
                for name, values in supported_values.items()
                if name in _FAULT_SIGNALS
            },
            "packet_count": {
                name: (sum(values) if values else None)
                for name, values in supported_values.items()
                if name in _PACKET_SIGNALS
            },
            "host_cycles_per_packet_max": {
                name: (max(values) if values else None)
                for name, values in supported_values.items()
                if name in _HOST_SIGNALS
            },
        }
    return result


def _vllm_cumulative_block(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]] | None:
    if any("vllm_cumulative" not in probe for probe in (*before.values(),
                                                         *after.values())):
        return None
    result = {}
    for endpoint_id in sorted(before):
        start = before[endpoint_id]["vllm_cumulative"]
        end = after[endpoint_id]["vllm_cumulative"]
        validate_vllm_endpoint_cumulative(start)
        validate_vllm_endpoint_cumulative(end)
        _require(start["engine_indices"] == end["engine_indices"],
                 "vLLM engine set changed within a block")
        delta = {}
        for name, final in end["values"].items():
            initial = start["values"][name]
            _require(final >= initial, "cumulative vLLM metric regressed")
            delta[name] = final - initial

        def mean(prefix: str) -> float | None:
            count = delta[prefix + "_count"]
            return delta[prefix + "_sum"] / count if count else None

        result[endpoint_id] = {
            "engine_indices": end["engine_indices"],
            "delta": delta,
            "derived": {
                "mean_time_to_first_token_seconds": mean(
                    "vllm:time_to_first_token_seconds"),
                "mean_e2e_request_latency_seconds": mean(
                    "vllm:e2e_request_latency_seconds"),
                "mean_queue_time_seconds": mean(
                    "vllm:request_queue_time_seconds"),
                "mean_inference_time_seconds": mean(
                    "vllm:request_inference_time_seconds"),
                "mean_prefill_time_seconds": mean(
                    "vllm:request_prefill_time_seconds"),
                "mean_decode_time_seconds": mean(
                    "vllm:request_decode_time_seconds"),
                "mean_prefill_kv_computed_tokens": mean(
                    "vllm:request_prefill_kv_computed_tokens"),
            },
        }
    return result


def _block(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(raw.get("schema") == "tempo-pd-stream-metrics-raw-1",
             "child raw schema mismatch")
    contract = raw.get("contention_fixed_contract")
    _require(isinstance(contract, dict), "fixed block contract is missing")
    _require(contract.get("schema") in _BLOCK_SCHEMAS,
             "fixed block schema mismatch")
    _require(raw.get("validation", {}).get("all_streams_valid") is True,
             "child stream validation failed")
    evidence = raw.get("endpoint_evidence")
    client._validate_endpoint_evidence_bundle(evidence)
    stages = {
        name: _endpoint_index(evidence[name])
        for name in ("before", "midpoint", "after")
    }
    request_index = contract["request_index"]
    foreground = []
    background = []
    for row in raw["requests"]:
        _require(row.get("valid") is True, "invalid request in fixed block")
        e2e_ms = (
            row["stream_end_offset_ns"] - row["dispatch_offset_ns"]
        ) / 1_000_000.0
        tenant = request_index[row["request_id"]]["tenant"]
        (foreground if tenant == "foreground" else background).append(e2e_ms)
    _require(bool(foreground) and bool(background),
             "fixed block lacks foreground or background requests")
    midpoint_load = {}
    for endpoint_id, probe in stages["midpoint"].items():
        metrics = probe["endpoint"]["metrics"]
        midpoint_load[endpoint_id] = {
            "role": probe["endpoint"]["role"],
            "pair_index": probe["endpoint"]["pair_index"],
            "running_requests": metrics["running_requests"]["value"],
            "waiting_requests": metrics["waiting_requests"]["value"],
            "kv_cache_usage_fraction": metrics[
                "kv_cache_usage_fraction"]["value"],
        }
    cassini = _cassini_block(stages["midpoint"], stages["after"])
    cumulative = _vllm_cumulative_block(stages["before"], stages["after"])
    fabric_event_observed = any(
        any(value not in (None, 0, 0.0)
            for value in endpoint["fraction_max"].values())
        or any(value not in (None, 0)
               for value in endpoint["fault_count"].values())
        for endpoint in cassini.values()
    )
    return {
        "artifact": str(path.resolve()),
        "phase": contract["phase"],
        "foreground_arm": contract["foreground_arm"],
        "replicate": contract["replicate"],
        "request_counts": contract["request_counts"],
        "foreground_e2e_ms": {
            "median": statistics.median(foreground),
            "p99": _percentile(foreground, 0.99),
        },
        "background_e2e_ms": {
            "median": statistics.median(background),
            "p99": _percentile(background, 0.99),
        },
        "midpoint_load": midpoint_load,
        "vllm_cumulative": cumulative,
        "cassini": cassini,
        "cassini_fabric_event_observed": fabric_event_observed,
    }


def analyze(input_path: Path) -> dict[str, object]:
    parent = json.loads(input_path.read_text(encoding="utf-8"))
    _require(parent.get("schema") in _CLIENT_SCHEMAS,
             "fixed client parent schema mismatch")
    _require(parent.get("controller_tuning_allowed") is True,
             "input workload did not pass crossover gate")
    artifacts = parent.get("artifacts")
    _require(isinstance(artifacts, dict) and len(artifacts) == 8,
             "exact eight-block artifact map is required")
    blocks = [_block(Path(path)) for path in artifacts.values()]
    return {
        "schema": SCHEMA,
        "source": str(input_path.resolve()),
        "crossover_gate": parent["crossover_gate"],
        "blocks": blocks,
        "invariants": {
            "endpoint_count": 4,
            "snapshots_per_endpoint_per_block": 3,
            "cross_endpoint_clock_subtraction": False,
            "synchronous_decision_path_sampling": False,
            "synthetic_network_background": False,
        },
        "interpretation_boundary": {
            "characterization_only": True,
            "performance_claim_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
            "zero_pause_or_ecn_proves_uncongested_fabric": False,
        },
    }


def main() -> int:
    args = _parse()
    _require(args.input.is_file(), "input artifact is missing")
    _require(not args.output.exists(), "refusing to overwrite output")
    result = analyze(args.input)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": SCHEMA, "output": str(args.output.resolve())},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
