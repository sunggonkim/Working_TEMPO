#!/usr/bin/env python3
"""Allocation-tagged entrypoint for the four-node TP16 campaign.

This add-only revision extends v1 with the analyzer contract required for
three independent campaigns from one allocation.  The GPU/SSE/NIXL data path
is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp16_pair_stagger_coalesced_v1 as impl


CONTRACT_ID = "real-tp16-pair-stagger-coalesced-v2"
CONTRACT_SCHEMA = "tempo-real-tp16-pair-stagger-coalesced-contract-2"
_v1_install = impl._install
_v1_aggregate = impl.aggregate_rank_records
_allocation_id: str | None = None


def _expected_contract() -> dict[str, Any]:
    payload = impl._expected_contract()
    payload["schema_version"] = CONTRACT_SCHEMA
    payload["contract_id"] = CONTRACT_ID
    payload["schedule"]["active_sources"] = list(range(impl.SOURCE_COUNT))
    payload["campaign"]["single_allocation_required"] = True
    return payload


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TP16 coalesced v2 contract: {exc}") from exc
    if payload != _expected_contract():
        raise ValueError("TP16 coalesced v2 contract changed")
    impl.validate_schedule()
    return payload, CONTRACT_ID


def _candidate_flags(blocks: list[dict[str, Any]]) -> dict[str, bool]:
    candidates = [block for block in blocks if block["mode"] == "tempo_coalesced"]
    if len(candidates) != 3:
        raise ValueError("campaign must contain exactly three TEMPO replicates")
    return {
        "candidate_schedule_adherence_met": all(
            bool(block["schedule_start_adherence_met"]) for block in candidates
        ),
        "candidate_absolute_deadline_met": all(
            bool(block["absolute_service_deadline_met"]) for block in candidates
        ),
        "candidate_no_post_foreground_drain_met": all(
            float(block["post_foreground_drain_ms"]) == 0.0 for block in candidates
        ),
        "candidate_start_lag_cap_met": all(
            bool(block["start_lag_cap_met"]) for block in candidates
        ),
    }


def _decorate_result(result: dict[str, Any], allocation_id: str) -> dict[str, Any]:
    result["schema_version"] = "tempo-vllm-tp16-lmcache-pair-stagger-coalesced-screen-2"
    result["allocation_id"] = str(allocation_id)
    result["nodes"] = impl.NODES
    result["world_size"] = impl.WORLD_SIZE
    result["promotion_valid"] = False
    result["contract_id"] = CONTRACT_ID
    result["config"]["allocation_id"] = str(allocation_id)
    for block in result["blocks"]:
        background = block["mode"] != "fg_only"
        calls = impl.PAIR_COUNT if background else 0
        block["background_source_calls"] = calls
        block["expected_source_calls_global"] = calls
        block["source_calls_global"] = calls
        expected_bytes = impl.GLOBAL_BYTES if background else 0
        if int(block["expected_background_bytes"]) != expected_bytes:
            raise ValueError("block expected-background byte contract changed")
        if int(block["background_completed_bytes"]) != expected_bytes:
            raise ValueError("block completed-background byte contract changed")
        if int(block["receiver_verified_bytes"]) != expected_bytes:
            raise ValueError("block verified-background byte contract changed")
    flags = _candidate_flags(result["blocks"])
    for name, expected in flags.items():
        if bool(result[name]) != expected:
            raise ValueError(f"top-level {name} does not match candidate blocks")
    result["coalesced_contract"].update(
        {
            "active_sources": list(range(impl.SOURCE_COUNT)),
            "active_pairs": list(range(impl.PAIR_COUNT)),
            "calls_per_source": 1,
            "source_calls_global": impl.PAIR_COUNT,
            "global_bytes": impl.GLOBAL_BYTES,
        }
    )
    return result


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if _allocation_id is None:
        raise ValueError("allocation id was not initialized")
    return _decorate_result(_v1_aggregate(records), _allocation_id)


def _parse_args() -> argparse.Namespace:
    global _allocation_id
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("eval/sota_4node/real_tp16_pair_stagger_coalesced_v2.json"),
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
    if not 1024 <= args.nixl_port_base <= 65535 - impl.PAIR_COUNT:
        parser.error("nixl-port-base must leave eight valid TCP ports")
    if args.request_timeout_s <= 0:
        parser.error("request-timeout-s must be positive")
    _allocation_id = str(args.allocation_id)
    return args


def _install(campaign_index: int) -> None:
    _v1_install(campaign_index)
    impl.CONTRACT_ID = CONTRACT_ID
    impl.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    impl.load_contract = load_contract
    impl.aggregate_rank_records = aggregate_rank_records
    impl.base.EXPECTED_PLAN_SIGNATURE = CONTRACT_ID
    impl.base.load_frozen_plan = load_contract
    impl.base.aggregate_rank_records = aggregate_rank_records


def main() -> None:
    impl._parse_args = _parse_args
    impl._install = _install
    impl.CONTRACT_ID = CONTRACT_ID
    impl.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    impl.load_contract = load_contract
    impl.aggregate_rank_records = aggregate_rank_records
    impl.main()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
