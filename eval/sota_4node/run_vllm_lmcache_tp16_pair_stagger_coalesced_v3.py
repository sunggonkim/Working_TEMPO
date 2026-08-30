#!/usr/bin/env python3
"""Clock-domain-safe entrypoint for the four-node TP16 campaign.

Rank zero's SSE timestamps remain the source of foreground TTFT/TPOT/E2E.
Background transfer intervals use the receiving rank's ``perf_counter_ns``
immediately after each Gloo control broadcast.  Raw monotonic timestamps are
never subtracted across hosts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v2 as v2


v1 = v2.impl
CONTRACT_ID = "real-tp16-pair-stagger-coalesced-v3"
CONTRACT_SCHEMA = "tempo-real-tp16-pair-stagger-coalesced-contract-3"
RESULT_SCHEMA = "tempo-vllm-tp16-lmcache-pair-stagger-coalesced-screen-3"
TIMING_SEMANTICS = {
    "foreground_metrics_clock_domain": "rank_zero_perf_counter_ns",
    "sidecar_metrics_clock_domain": "rank_local_perf_counter_ns",
    "control_trigger_timestamp": "local_receipt_immediately_after_gloo_broadcast",
    "background_finish_origin": "rank_local_request_start_control_receipt",
    "post_foreground_drain_origin": "rank_local_final_token_control_receipt",
    "control_delivery_lag_field": "local_control_receipt_to_enqueue_dispatch_lag",
    "cross_host_raw_monotonic_subtraction": False,
}

_v1_run_block = v1._run_block
_v2_install = v2._install
_v2_aggregate = v2.aggregate_rank_records


class _RankLocalControlClock:
    """Proxy a process group and localize only start/token event timestamps."""

    def __init__(
        self,
        dist: Any,
        *,
        now_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._dist = dist
        self._now_ns = now_ns

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)

    def broadcast_object_list(self, values: list[Any], *, src: int) -> None:
        self._dist.broadcast_object_list(values, src=src)
        received_ns = int(self._now_ns())
        if len(values) != 1 or not isinstance(values[0], dict):
            return
        event = values[0]
        kind = event.get("kind")
        if kind == "started":
            event["value"] = received_ns
        elif kind == "token" and isinstance(event.get("value"), dict):
            event["value"]["arrival_ns"] = received_ns


def _decorate_clock_safe_block(
    block: dict[str, Any], *, request_timeout_s: float
) -> dict[str, Any]:
    """Validate local timelines and restore exact rank-zero SSE metrics."""

    maximum_dispatch_lag_ms = request_timeout_s * 1_000.0
    for record in block.get("transfer_records", []):
        timeline = [
            int(record[name])
            for name in ("trigger_ns", "enqueue_ns", "started_ns", "finished_ns")
        ]
        if timeline != sorted(timeline):
            raise RuntimeError(
                "rank-local transfer timeline is not monotonic; mixed clock domains"
            )
        dispatch_lag_ms = float(record["control_delivery_lag_ms"])
        if not 0.0 <= dispatch_lag_ms <= maximum_dispatch_lag_ms:
            raise RuntimeError(
                "control dispatch lag exceeds the request timeout; probable mixed clock domains"
            )
        record["clock_domain"] = "rank_local_perf_counter_ns"
        record["trigger_semantics"] = "local_control_broadcast_receipt"
        record["local_control_dispatch_lag_ms"] = dispatch_lag_ms

    client = block.get("client")
    if isinstance(client, dict):
        request_started_ns = int(client["request_started_ns"])
        arrivals = [int(value) for value in client["token_arrival_ns"]]
        client["ttft_ms"] = (arrivals[0] - request_started_ns) / 1_000_000.0
        client["request_e2e_ms"] = (
            int(client["finished_ns"]) - request_started_ns
        ) / 1_000_000.0
        client["clock_domain"] = "rank_zero_perf_counter_ns"

    block["timing_clock_domain"] = "rank_local_perf_counter_ns"
    block["control_trigger_semantics"] = (
        "local_receipt_immediately_after_gloo_broadcast"
    )
    return block


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if len(args) < 2:
        raise TypeError("_run_block requires torch and dist positional arguments")
    proxied_args = list(args)
    proxied_args[1] = _RankLocalControlClock(args[1])
    result = _v1_run_block(*proxied_args, **kwargs)
    return _decorate_clock_safe_block(
        result,
        request_timeout_s=float(kwargs["request_timeout_s"]),
    )


def _expected_contract() -> dict[str, Any]:
    payload = v2._expected_contract()
    payload["schema_version"] = CONTRACT_SCHEMA
    payload["contract_id"] = CONTRACT_ID
    payload["timing"] = dict(TIMING_SEMANTICS)
    return payload


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TP16 coalesced v3 contract: {exc}") from exc
    if payload != _expected_contract():
        raise ValueError("TP16 coalesced v3 contract changed")
    v1.validate_schedule()
    return payload, CONTRACT_ID


def _decorate_result(result: dict[str, Any]) -> dict[str, Any]:
    result["schema_version"] = RESULT_SCHEMA
    result["contract_id"] = CONTRACT_ID
    result["timing_semantics"] = dict(TIMING_SEMANTICS)
    result["config"]["timing_semantics"] = dict(TIMING_SEMANTICS)
    result["foreground"]["clock_domain"] = "rank_zero_perf_counter_ns"
    result["background"]["clock_domain"] = "rank_local_perf_counter_ns"
    for block in result["blocks"]:
        block["timing_clock_domain"] = "rank_local_perf_counter_ns"
        block["control_trigger_semantics"] = (
            "local_receipt_immediately_after_gloo_broadcast"
        )
    return result


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _decorate_result(_v2_aggregate(records))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("eval/sota_4node/real_tp16_pair_stagger_coalesced_v3.json"),
    )
    parser.add_argument("--api-host", required=True)
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument("--model", default="models/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--nixl-port-base", type=int, default=35100)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--campaign-index", type=int, choices=range(3), required=True)
    parser.add_argument("--allocation-id", default=os.environ.get("SLURM_JOB_ID"))
    args = parser.parse_args()
    if not args.allocation_id:
        parser.error("allocation-id is required outside Slurm")
    if not 1024 <= args.api_port <= 65535:
        parser.error("api-port must be a valid TCP port")
    if not 1024 <= args.nixl_port_base <= 65535 - v1.PAIR_COUNT:
        parser.error("nixl-port-base must leave eight valid TCP ports")
    if args.request_timeout_s <= 0:
        parser.error("request-timeout-s must be positive")
    v2._allocation_id = str(args.allocation_id)
    return args


def _install(campaign_index: int) -> None:
    _v2_install(campaign_index)
    v1.CONTRACT_ID = CONTRACT_ID
    v1.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    v1.load_contract = load_contract
    v1.aggregate_rank_records = aggregate_rank_records
    v1._run_block = _run_block
    v1.base.EXPECTED_PLAN_SIGNATURE = CONTRACT_ID
    v1.base.load_frozen_plan = load_contract
    v1.base.aggregate_rank_records = aggregate_rank_records
    v1.base._run_block = _run_block


def main() -> None:
    v1._parse_args = _parse_args
    v1._install = _install
    v1.CONTRACT_ID = CONTRACT_ID
    v1.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    v1.load_contract = load_contract
    v1.aggregate_rank_records = aggregate_rank_records
    v1.main()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
