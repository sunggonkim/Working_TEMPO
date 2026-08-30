#!/usr/bin/env python3
"""Corrected add-only entrypoint for the vLLM TP8 + LMCache sidecar screen.

The first add-only draft cannot be edited in the current workspace snapshot,
so this module applies four narrow corrections before delegating to it:

* reject a changed signed envelope before deeper artifact decoding;
* issue frozen pulse ``t`` from the observed ``t-1 -> t`` decode boundary;
* evaluate that issue against output-token ``t`` arrival;
* resolve the default served-model name to the absolute local model path.

All experiment geometry, NixlChannel execution, metrics, aggregation, and
honesty boundary remain those of the fully audited draft.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp8_sidecar as _v1


# Public frozen-contract aliases used by static checks and analysis.
WORLD_SIZE = _v1.WORLD_SIZE
NODES = _v1.NODES
RANKS_PER_NODE = _v1.RANKS_PER_NODE
PAIR_COUNT = _v1.PAIR_COUNT
REQUESTS = _v1.REQUESTS
TOKENS = _v1.TOKENS
CHUNKS_PER_REQUEST = _v1.CHUNKS_PER_REQUEST
CHUNK_BYTES = _v1.CHUNK_BYTES
KV_BYTES_PER_RANK = _v1.KV_BYTES_PER_RANK
ABSOLUTE_DEADLINE_NS = _v1.ABSOLUTE_DEADLINE_NS
START_LAG_CAP_NS = _v1.START_LAG_CAP_NS
EXPECTED_PLAN_SIGNATURE = _v1.EXPECTED_PLAN_SIGNATURE
EXPECTED_ARTIFACT_SIGNATURE = _v1.EXPECTED_ARTIFACT_SIGNATURE
PULSE_TOKENS = _v1.PULSE_TOKENS
MODES = _v1.MODES
LATIN_ROWS = _v1.LATIN_ROWS
BLOCK_SPECS = _v1.BLOCK_SPECS
PROMPTS = _v1.PROMPTS
schedule_object_indices = _v1.schedule_object_indices
validate_frozen_schedule = _v1.validate_frozen_schedule
iter_sse_chunks = _v1.iter_sse_chunks
request_completion = _v1.request_completion
aggregate_rank_records = _v1.aggregate_rank_records
percentile = _v1.percentile


_original_load_frozen_plan = _v1.load_frozen_plan
_original_parse_args = _v1._parse_args
_original_schedule_object_indices = _v1.schedule_object_indices
_original_run_block = _v1._run_block
_shift_group2_schedule = False


def load_frozen_plan(path: Path) -> tuple[dict[str, Any], str]:
    """Fail on a changed signed envelope before decoding its nested plan."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid frozen group2 artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid frozen group2 artifact: artifact must contain an object")
    if payload.get("artifact_signature_sha256") != EXPECTED_ARTIFACT_SIGNATURE:
        raise ValueError("frozen group2 artifact signature changed")
    return _original_load_frozen_plan(path)


def _parse_args() -> Any:
    args = _original_parse_args()
    if args.model == "models/TinyLlama-1.1B-Chat-v1.0":
        repo_root = Path(_v1.__file__).resolve().parents[2]
        args.model = str((repo_root / args.model).resolve())
    return args


def _runtime_schedule_object_indices(
    mode: str,
    token_index: int,
    *,
    pair_index: int,
) -> tuple[int, ...]:
    if _shift_group2_schedule and mode == "tempo_group2":
        scheduled_token = token_index + 1
        if scheduled_token >= TOKENS:
            return ()
        return _original_schedule_object_indices(
            mode, scheduled_token, pair_index=pair_index
        )
    return _original_schedule_object_indices(mode, token_index, pair_index=pair_index)


def _run_block(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Trigger group2 token t after token t-1, before token t arrives."""

    global _shift_group2_schedule
    if _shift_group2_schedule:
        raise RuntimeError("nested sidecar block execution is not supported")
    _shift_group2_schedule = True
    try:
        result = _original_run_block(*args, **kwargs)
    finally:
        _shift_group2_schedule = False
    if result.get("mode") == "tempo_group2":
        for record in result.get("transfer_records", []):
            # The v1 calculation already used arrival[old_token + 1] as the
            # window end. Relabeling old_token to scheduled token t therefore
            # records the corrected [arrival(t-1), arrival(t)] issue window.
            record["triggered_after_token"] = int(record["scheduled_token"])
            record["scheduled_token"] = int(record["scheduled_token"]) + 1
            record["trigger_semantics"] = "observed_t_minus_1_to_t_decode_boundary"
    result["sidecar_revision"] = 2
    result["group2_trigger_semantics"] = "observed_t_minus_1_to_t_decode_boundary"
    return result


def _install_corrections() -> None:
    _v1.load_frozen_plan = load_frozen_plan
    _v1._parse_args = _parse_args
    _v1.schedule_object_indices = _runtime_schedule_object_indices
    _v1._run_block = _run_block


def main() -> None:
    _install_corrections()
    _v1.main()


if __name__ == "__main__":
    main()
