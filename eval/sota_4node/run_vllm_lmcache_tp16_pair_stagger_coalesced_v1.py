#!/usr/bin/env python3
"""Four-node TP16 coalesced LMCache/NIXL contention campaign.

The foreground is one real vLLM TP16 engine.  The component sidecar has 16
node-major ranks: ranks 0..7 are sources on nodes 0..1 and ranks 8..15 are
their receivers on nodes 2..3.  Each source moves two 8 MiB request images in
one official LMCache ``NixlChannel.batched_write`` call.  The TEMPO candidate
admits pair i at scheduled output token i+1; calls are not globally
single-flight and may overlap physically.

This remains a research component screen: the registered GPU buffers have KV
traffic geometry but are not vLLM-owned live KV-cache pages.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np

from eval.sota_4node import run_vllm_lmcache_tp8_sidecar as base


WORLD_SIZE = 16
NODES = 4
LOCAL_RANKS_PER_NODE = 4
SOURCE_COUNT = 8
PAIR_COUNT = 8
RECEIVER_OFFSET = 8
REQUESTS = 2
CHUNKS_PER_REQUEST = 16
CHUNK_BYTES = 512 * (1 << 10)
BYTES_PER_REQUEST = CHUNKS_PER_REQUEST * CHUNK_BYTES
BYTES_PER_SOURCE = REQUESTS * BYTES_PER_REQUEST
GLOBAL_BYTES = PAIR_COUNT * BYTES_PER_SOURCE
TOKENS = 64
SCHEDULED_TOKENS = tuple(range(1, PAIR_COUNT + 1))
MODES = ("fg_only", "lmcache_greedy", "tempo_coalesced")
CANONICAL_LATIN_ROWS = tuple(
    tuple(MODES[(column + row) % len(MODES)] for column in range(len(MODES)))
    for row in range(len(MODES))
)
POLICY = "tp16_pair_staggered_coalesced_admission_v1"
CONTRACT_ID = "real-tp16-pair-stagger-coalesced-v1"
CONTRACT_SCHEMA = "tempo-real-tp16-pair-stagger-coalesced-contract-1"
DEADLINE_NS = 1_250_000_000

_original_run_block = base._run_block
_original_aggregate = base.aggregate_rank_records
_runtime_shift = False
_runtime_token_zero_calls = 0


def campaign_latin_rows(campaign_index: int) -> tuple[tuple[str, ...], ...]:
    if isinstance(campaign_index, bool) or campaign_index not in range(3):
        raise ValueError("campaign_index must be 0, 1, or 2")
    return tuple(
        CANONICAL_LATIN_ROWS[(campaign_index + offset) % 3]
        for offset in range(3)
    )


def campaign_block_specs(campaign_index: int) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (prompt_index, position, mode)
        for prompt_index, row in enumerate(campaign_latin_rows(campaign_index))
        for position, mode in enumerate(row)
    )


def coalesced_indices(mode: str, scheduled_token: int, *, pair_index: int) -> tuple[int, ...]:
    """Return rank-local objects for a declared schedule position."""

    if mode not in (*MODES, "tempo_group2"):
        raise ValueError(f"unknown mode: {mode}")
    if isinstance(scheduled_token, bool) or not isinstance(scheduled_token, int):
        raise ValueError("scheduled_token must be an int")
    if not 0 <= scheduled_token < TOKENS:
        raise ValueError(f"scheduled_token must be in 0..{TOKENS - 1}")
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise ValueError("pair_index must be an int")
    if not 0 <= pair_index < PAIR_COUNT:
        raise ValueError("pair_index must be in 0..7")
    if mode == "fg_only":
        return ()
    all_objects = tuple(range(REQUESTS * CHUNKS_PER_REQUEST))
    if mode == "lmcache_greedy":
        return all_objects if scheduled_token == 0 else ()
    if scheduled_token not in SCHEDULED_TOKENS:
        return ()
    return all_objects if pair_index == scheduled_token - 1 else ()


def validate_schedule() -> None:
    if GLOBAL_BYTES != 134_217_728 or BYTES_PER_SOURCE != 16_777_216:
        raise RuntimeError("TP16 transfer geometry changed")
    if campaign_latin_rows(0)[0][0] != "fg_only":
        raise RuntimeError("campaign zero first mode changed")
    if campaign_latin_rows(1)[0][0] != "lmcache_greedy":
        raise RuntimeError("campaign one first mode changed")
    if campaign_latin_rows(2)[0][0] != "tempo_coalesced":
        raise RuntimeError("campaign two first mode changed")
    for campaign_index in range(3):
        sequence = [mode for _, _, mode in campaign_block_specs(campaign_index)]
        if len(sequence) != 9 or any(sequence.count(mode) != 3 for mode in MODES):
            raise RuntimeError("each campaign must contain three Latin rows")
    for pair in range(PAIR_COUNT):
        active = [
            token
            for token in range(TOKENS)
            if coalesced_indices("tempo_coalesced", token, pair_index=pair)
        ]
        if active != [pair + 1]:
            raise RuntimeError(f"pair {pair} admission changed")
        if coalesced_indices("tempo_coalesced", active[0], pair_index=pair) != tuple(range(32)):
            raise RuntimeError(f"pair {pair} object coverage changed")


def _expected_contract() -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA,
        "contract_id": CONTRACT_ID,
        "provenance": {
            "source": "four_node_real_tp16_research_screen",
            "adaptive_pilot": True,
            "promotion_valid": False,
            "policy_label": POLICY,
            "global_single_flight": False,
            "physical_transfer_overlap_possible": True,
        },
        "topology": {
            "nodes": NODES,
            "world_size": WORLD_SIZE,
            "ranks_per_node": LOCAL_RANKS_PER_NODE,
            "source_ranks": list(range(SOURCE_COUNT)),
            "receiver_ranks": list(range(RECEIVER_OFFSET, WORLD_SIZE)),
            "pairing": [[rank, rank + RECEIVER_OFFSET] for rank in range(PAIR_COUNT)],
        },
        "schedule": {
            "scheduled_tokens": list(SCHEDULED_TOKENS),
            "active_pairs": list(range(PAIR_COUNT)),
            "calls_per_source": 1,
            "source_calls_global": PAIR_COUNT,
            "requests_per_source": REQUESTS,
            "chunks_per_request": CHUNKS_PER_REQUEST,
            "chunk_bytes": CHUNK_BYTES,
            "bytes_per_request": BYTES_PER_REQUEST,
            "bytes_per_source_call": BYTES_PER_SOURCE,
            "global_bytes": GLOBAL_BYTES,
            "deadline_ns": DEADLINE_NS,
        },
        "campaign": {
            "independent_campaigns": 3,
            "latin_rows_per_campaign": 3,
            "replicates_per_mode_per_campaign": 3,
            "first_mode_by_campaign": list(MODES),
        },
    }


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TP16 coalesced contract: {exc}") from exc
    if payload != _expected_contract():
        raise ValueError("TP16 coalesced contract changed")
    validate_schedule()
    return payload, CONTRACT_ID


def _runtime_schedule(mode: str, token_index: int, *, pair_index: int) -> tuple[int, ...]:
    """Translate arrival of output i into scheduled boundary i+1."""

    global _runtime_token_zero_calls
    if mode != "tempo_group2" or not _runtime_shift:
        return coalesced_indices(mode, token_index, pair_index=pair_index)
    # The shared data path invokes enqueue(0) at request start and again after
    # output token zero arrives.  Only the second call is boundary 0 -> 1.
    if token_index == 0 and _runtime_token_zero_calls == 0:
        _runtime_token_zero_calls += 1
        return ()
    scheduled_token = token_index + 1
    if scheduled_token >= TOKENS:
        return ()
    return coalesced_indices(
        "tempo_coalesced", scheduled_token, pair_index=pair_index
    )


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    global _runtime_shift, _runtime_token_zero_calls
    requested_mode = kwargs.get("mode")
    if requested_mode != "tempo_coalesced":
        result = _original_run_block(*args, **kwargs)
        result["coalesced_calls_per_source"] = 0 if requested_mode == "fg_only" else 1
        return result
    if _runtime_shift:
        raise RuntimeError("nested sidecar block execution is not supported")
    translated = dict(kwargs)
    translated["mode"] = "tempo_group2"
    _runtime_shift = True
    _runtime_token_zero_calls = 0
    try:
        result = _original_run_block(*args, **translated)
    finally:
        _runtime_shift = False
    for record in result.get("transfer_records", []):
        record["triggered_after_token_index"] = int(record["scheduled_token"])
        record["scheduled_token"] = int(record["scheduled_token"]) + 1
        record["trigger_semantics"] = "observed_t_minus_1_to_t_decode_boundary"
    result["mode"] = "tempo_coalesced"
    result["candidate_policy"] = POLICY
    result["coalesced_calls_per_source"] = 1
    result["global_single_flight"] = False
    result["physical_transfer_overlap_possible"] = True
    return result


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = _original_aggregate(records)
    for block in result["blocks"]:
        background = block["mode"] != "fg_only"
        block["expected_source_calls_global"] = PAIR_COUNT if background else 0
        block["source_calls_global"] = PAIR_COUNT if background else 0
        if background:
            if int(block["expected_background_bytes"]) != GLOBAL_BYTES:
                raise ValueError("aggregated expected bytes changed")
            if int(block["background_completed_bytes"]) != GLOBAL_BYTES:
                raise ValueError("aggregated completed bytes changed")
            if int(block["receiver_verified_bytes"]) != GLOBAL_BYTES:
                raise ValueError("aggregated verified bytes changed")

    candidate_blocks = [
        block for block in result["blocks"] if block["mode"] == "tempo_coalesced"
    ]
    if len(candidate_blocks) != 3:
        raise ValueError("campaign must contain exactly three TEMPO replicates")
    adherence = all(bool(block["schedule_start_adherence_met"]) for block in candidate_blocks)
    deadline = all(bool(block["absolute_service_deadline_met"]) for block in candidate_blocks)
    no_drain = all(float(block["post_foreground_drain_ms"]) == 0.0 for block in candidate_blocks)
    lag_cap = all(bool(block["start_lag_cap_met"]) for block in candidate_blocks)
    overall = bool(result["overall_correctness_met"])
    if not overall:
        outcome = "invalid_output_or_transfer_correctness"
    elif not adherence:
        outcome = "kill_external_token_trigger_adherence_miss"
    elif not deadline:
        outcome = "kill_absolute_service_deadline_miss"
    elif not no_drain:
        outcome = "kill_post_foreground_drain"
    elif not lag_cap:
        outcome = "valid_service_but_lag_cap_not_met"
    else:
        outcome = "valid_component_screen_requires_performance_comparison"

    campaign_index = int(result["config"]["campaign_index"])
    result.update(
        {
            "schema_version": "tempo-vllm-tp16-lmcache-pair-stagger-coalesced-screen-1",
            "evidence_state": "four_node_real_tp16_component_screen",
            "claim_scope": "research_component_screen_not_promotion_not_end_to_end_kv_connector",
            "nodes": NODES,
            "world_size": WORLD_SIZE,
            "campaign_index": campaign_index,
            "candidate_policy": POLICY,
            "contract_id": CONTRACT_ID,
            "adaptive_pilot": True,
            "promotion_valid": False,
            "global_single_flight": False,
            "physical_transfer_overlap_possible": True,
            "pairing": [[rank, rank + RECEIVER_OFFSET] for rank in range(PAIR_COUNT)],
            "coalesced_contract": {
                "scheduled_tokens": list(SCHEDULED_TOKENS),
                "active_pairs": list(range(PAIR_COUNT)),
                "calls_per_source": 1,
                "source_calls_global": PAIR_COUNT,
                "bytes_per_source": BYTES_PER_SOURCE,
                "global_bytes": GLOBAL_BYTES,
                "absolute_deadline_ns": DEADLINE_NS,
            },
            "candidate_schedule_adherence_met": adherence,
            "candidate_absolute_deadline_met": deadline,
            "candidate_no_post_foreground_drain_met": no_drain,
            "candidate_start_lag_cap_met": lag_cap,
            "screen_outcome": outcome,
        }
    )
    result["foreground"]["expected_tensor_parallel_size"] = 16
    result["frozen_group2"] = {
        "policy_label": POLICY,
        "contract_id": CONTRACT_ID,
        "scheduled_tokens": list(SCHEDULED_TOKENS),
        "promotion_valid": False,
    }
    return result


def _validate_topology(dist: Any, rank: int, local_rank: int) -> list[str]:
    layouts: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(layouts, (socket.gethostname(), local_rank))
    hosts = [str(item[0]) for item in layouts]
    local_ranks = [int(item[1]) for item in layouts]
    valid = len(set(hosts)) == NODES
    for node_index in range(NODES):
        start = node_index * LOCAL_RANKS_PER_NODE
        stop = start + LOCAL_RANKS_PER_NODE
        valid = valid and len(set(hosts[start:stop])) == 1
        valid = valid and local_ranks[start:stop] == list(range(LOCAL_RANKS_PER_NODE))
    valid = valid and rank == (rank // LOCAL_RANKS_PER_NODE) * LOCAL_RANKS_PER_NODE + local_rank
    if not valid:
        raise RuntimeError("requires four node-major groups of ranks 0..3")
    return hosts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("eval/sota_4node/real_tp16_pair_stagger_coalesced_v1.json"),
    )
    parser.add_argument("--api-host", required=True)
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument("--model", default="models/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--nixl-port-base", type=int, default=35100)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--campaign-index", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    if not 1024 <= args.api_port <= 65535:
        parser.error("api-port must be a valid TCP port")
    if not 1024 <= args.nixl_port_base <= 65535 - PAIR_COUNT:
        parser.error("nixl-port-base must leave eight valid TCP ports")
    if args.request_timeout_s <= 0:
        parser.error("request-timeout-s must be positive")
    return args


def _install(campaign_index: int) -> None:
    rows = campaign_latin_rows(campaign_index)
    base.WORLD_SIZE = WORLD_SIZE
    base.NODES = NODES
    # The audited TP8 data path used this symbol both as source-count/peer
    # stride and as local ranks per node.  Runtime topology and GPU visibility
    # are handled explicitly here; within the reused transfer path it is the
    # desired eight-source peer stride.
    base.RANKS_PER_NODE = SOURCE_COUNT
    base.PAIR_COUNT = PAIR_COUNT
    base.REQUESTS = REQUESTS
    base.TOKENS = TOKENS
    base.CHUNKS_PER_REQUEST = CHUNKS_PER_REQUEST
    base.CHUNK_BYTES = CHUNK_BYTES
    base.KV_BYTES_PER_RANK = BYTES_PER_REQUEST
    base.MODES = MODES
    base.LATIN_ROWS = rows
    base.BLOCK_SPECS = campaign_block_specs(campaign_index)
    base.ABSOLUTE_DEADLINE_NS = DEADLINE_NS
    base.EXPECTED_PLAN_SIGNATURE = CONTRACT_ID
    base.validate_frozen_schedule = validate_schedule
    base.load_frozen_plan = load_contract
    base.schedule_object_indices = _runtime_schedule
    base._run_block = _run_block
    base.aggregate_rank_records = aggregate_rank_records
    base._validate_topology = _validate_topology


def main() -> None:
    args = _parse_args()
    _install(args.campaign_index)
    repo_root = Path(__file__).resolve().parents[2]
    args.output_dir = base._resolve_below_repo(args.output_dir, repo_root, label="output-dir")
    args.plan = base._resolve_below_repo(args.plan, repo_root, label="plan")
    args.model = str((repo_root / args.model).resolve()) if not Path(args.model).is_absolute() else str(Path(args.model).resolve())
    validate_schedule()
    _, contract_id = load_contract(args.plan)
    if contract_id != CONTRACT_ID:
        raise SystemExit("TP16 coalesced contract mismatch")

    base._set_rank_environment()
    try:
        import torch
        import torch.distributed as dist
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch with CUDA and Gloo is required") from exc
    if not torch.cuda.is_available() or not dist.is_gloo_available():
        raise SystemExit("CUDA and Gloo are required")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise SystemExit("WORLD_SIZE must be exactly 16")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    visible_devices = torch.cuda.device_count()
    if visible_devices not in (1, LOCAL_RANKS_PER_NODE):
        raise SystemExit("sidecar rank must see one GPU or all four local GPUs")
    device_index = 0 if visible_devices == 1 else local_rank
    torch.cuda.set_device(device_index)
    dist.init_process_group("gloo")
    try:
        hosts = _validate_topology(dist, rank, local_rank)
        NixlChannel, TensorMemoryObj, MemoryObjMetadata, MemoryFormat = base.official._load_official_lmcache(repo_root)
        base.microburst.install_microburst_geometry()
        backing, buffer, objects, index_by_address = base.epoch._make_chunk_memory(
            torch,
            TensorMemoryObj,
            MemoryObjMetadata,
            MemoryFormat,
            requests=REQUESTS,
            chunk_bytes=CHUNK_BYTES,
        )
        pair_index = rank % SOURCE_COUNT
        is_source = rank < SOURCE_COUNT
        peer_rank = rank + RECEIVER_OFFSET if is_source else rank - RECEIVER_OFFSET
        channel = NixlChannel(
            async_mode=False,
            role="sender" if is_source else "receiver",
            buffer_ptr=buffer.data_ptr(),
            buffer_size=buffer.numel(),
            align_bytes=CHUNK_BYTES,
            tp_rank=local_rank,
            peer_init_url=None if is_source else f"*:{args.nixl_port_base + pair_index}",
            backends=["UCX"],
            device=f"cuda:{device_index}",
        )
        base.epoch._install_descriptor_index_shim(channel, index_by_address)
        dist.barrier()
        if is_source:
            channel.lazy_init_peer_connection(
                local_id=f"rank-{rank}",
                peer_id=f"rank-{peer_rank}",
                peer_init_url=f"{hosts[peer_rank]}:{args.nixl_port_base + pair_index}",
            )
        dist.barrier()
        if not channel.remote_xfer_handler_exists(f"rank-{peer_rank}"):
            raise RuntimeError("LMCache/NIXL peer handshake did not install a handler")
        warmup = base._warm_channel(
            torch,
            dist,
            channel=channel,
            objects=objects,
            rank=rank,
            pair_index=pair_index,
        )

        warmup_status: list[Any] = [None]
        if rank == 0:
            try:
                warm = base.request_completion(
                    api_host=args.api_host,
                    api_port=args.api_port,
                    model=args.model,
                    prompt=base.WARMUP_PROMPT,
                    max_tokens=8,
                    timeout_s=args.request_timeout_s,
                )
                warmup_status[0] = {"ok": True, "generated_tokens": len(warm["token_ids"])}
            except BaseException as exc:
                warmup_status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(warmup_status, src=0)
        if not warmup_status[0].get("ok"):
            raise RuntimeError(f"vLLM warmup failed: {warmup_status[0]}")
        dist.barrier()

        blocks = [
            _run_block(
                torch,
                dist,
                channel=channel,
                objects=objects,
                rank=rank,
                device_index=device_index,
                pair_index=pair_index,
                block_index=block_index,
                prompt_index=prompt_index,
                latin_position=latin_position,
                mode=mode,
                api_host=args.api_host,
                api_port=args.api_port,
                model=args.model,
                request_timeout_s=args.request_timeout_s,
            )
            for block_index, (prompt_index, latin_position, mode) in enumerate(base.BLOCK_SPECS)
        ]
        config = {
            "model": args.model,
            "api_host": args.api_host,
            "api_port": args.api_port,
            "nodes": NODES,
            "world_size": WORLD_SIZE,
            "ranks_per_node": LOCAL_RANKS_PER_NODE,
            "source_ranks": list(range(SOURCE_COUNT)),
            "receiver_ranks": list(range(RECEIVER_OFFSET, WORLD_SIZE)),
            "requests": REQUESTS,
            "tokens_per_request": TOKENS,
            "chunks_per_rank_request": CHUNKS_PER_REQUEST,
            "chunk_bytes": CHUNK_BYTES,
            "kv_bytes_per_rank_request": BYTES_PER_REQUEST,
            "global_background_bytes": GLOBAL_BYTES,
            "nixl_port_base": args.nixl_port_base,
            "campaign_index": args.campaign_index,
            "latin_rows": [list(row) for row in base.LATIN_ROWS],
            "block_sequence": [mode for _, _, mode in base.BLOCK_SPECS],
            "replicates_per_mode": 3,
            "plan_path": str(args.plan.relative_to(repo_root)),
            "plan_signature": contract_id,
            "lmcache_commit": base.official.LMCACHE_COMMIT,
            "nixl_version": importlib.metadata.version("nixl"),
            "nixl_backend": "UCX",
            "control_process_group": "Gloo",
            "foreground_expected_tensor_parallel_size": 16,
            "prefix_cache_exposure_balanced_by_latin_prompt_rows": True,
            "unmeasured_nixl_warmup": warmup,
            "unmeasured_vllm_warmup": warmup_status[0],
        }
        rank_record = {
            "schema_version": "tempo-vllm-tp16-lmcache-sidecar-rank-1",
            "rank": rank,
            "local_rank": local_rank,
            "device_index": device_index,
            "hostname": hosts[rank],
            "config": config,
            "blocks": blocks,
        }
        gathered = [None] * WORLD_SIZE if rank == 0 else None
        dist.gather_object(rank_record, gathered, dst=0)
        final_status: list[Any] = [None]
        if rank == 0:
            try:
                assert gathered is not None
                args.output_dir.mkdir(parents=True, exist_ok=True)
                for item in gathered:
                    (args.output_dir / f"rank_{int(item['rank'])}.json").write_text(
                        json.dumps(item, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                result = aggregate_rank_records(gathered)
                result_path = args.output_dir / "result.json"
                result_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                final_status[0] = {
                    "ok": bool(result["overall_correctness_met"]),
                    "output": str(result_path),
                    "screen_outcome": result["screen_outcome"],
                }
            except BaseException as exc:
                final_status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(final_status, src=0)
        dist.barrier()
        if not isinstance(final_status[0], dict) or not final_status[0].get("ok"):
            raise RuntimeError(f"vLLM/LMCache TP16 sidecar screen failed: {final_status[0]}")
        if rank == 0:
            print(json.dumps(final_status[0], sort_keys=True))
        del backing
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
