#!/usr/bin/env python3
"""Apply the preregistered C8 held-out workload transform and run frozen v45.

The discovery client is intentionally imported, not edited.  This module adds
only exogenous workload variation: a contract-bound request seed, deterministic
sub-spacing arrival jitter, and a disjoint P_ONLY prompt-marker namespace.
None of these values are exposed to the TEMPO controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable

from eval.sota_4node import run_tempo_go_c8_dual_regime_client as frozen


SCHEMA = frozen.SCHEMA
CONTRACT_SCHEMA = frozen.CONTRACT_SCHEMA
CONTRACT_ENV = frozen.CONTRACT_ENV
ARM_ENV = frozen.ARM_ENV
INDEPENDENT_SCHEMA = "tempo-go-c8-independent-validation-v1"
EXECUTION_SCHEMA = "tempo-go-c8-independent-execution-v1"

# Re-export the seams used by the frozen four-node lifecycle.
_decoder_contract = frozen._decoder_contract
configure_node_environment = frozen.configure_node_environment

_ORIGINAL_UNIFORM_OFFSETS = frozen.c7._uniform_offsets
_ORIGINAL_C7_MATERIALIZE = frozen.c7._materialize_schedule
_ORIGINAL_REMOTE_MATERIALIZE = frozen._materialize_remote_schedule
_ORIGINAL_P_ONLY_PROMPT = frozen._p_only_prompt

_HELDOUT: dict[str, object] | None = None
_SCHEDULE_CONTEXT: str | None = None
_SCHEDULE_STREAM_INDEX = 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _argument(name: str) -> str:
    try:
        index = sys.argv.index(name)
        return sys.argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"missing required held-out argument: {name}") from exc


def _load_independent_contract() -> tuple[Path, dict[str, object]]:
    raw_path = os.environ.get(CONTRACT_ENV, "")
    _require(bool(raw_path), "held-out C8 contract environment is missing")
    path = Path(raw_path).resolve()
    _require(path.is_file(), "held-out C8 contract is missing")
    contract = json.loads(path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == CONTRACT_SCHEMA,
             "held-out C8 base schema differs")
    heldout = contract.get("independent_validation")
    _require(
        isinstance(heldout, dict)
        and heldout.get("schema") == INDEPENDENT_SCHEMA
        and heldout.get("preregistered_before_fresh_allocation") is True
        and heldout.get("controller_receives_workload_seed") is False
        and heldout.get("controller_receives_future_arrivals") is False,
        "held-out C8 preregistration is invalid",
    )
    return path, contract


def _heldout() -> dict[str, object]:
    _require(isinstance(_HELDOUT, dict), "held-out transform is not installed")
    assert isinstance(_HELDOUT, dict)
    return _HELDOUT


def _unit_interval(*parts: object) -> float:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _jittered_uniform_offsets(
    duration_ms: float, rate_per_s: float,
) -> list[float]:
    """Keep count/rate fixed while changing arrivals within each spacing."""

    global _SCHEDULE_STREAM_INDEX
    base = _ORIGINAL_UNIFORM_OFFSETS(duration_ms, rate_per_s)
    if not base:
        _SCHEDULE_STREAM_INDEX += 1
        return base
    heldout = _heldout()
    jitter = heldout["arrival_jitter"]
    _require(isinstance(jitter, dict), "held-out arrival jitter is missing")
    _require(jitter.get("algorithm") == "sha256_centered_subspacing_v1",
             "held-out arrival jitter algorithm differs")
    fraction = float(jitter["maximum_spacing_fraction"])
    _require(0.0 < fraction <= 0.25,
             "held-out jitter can reorder adjacent arrivals")
    seed = int(heldout["request_seed"])
    spacing_ms = 1000.0 / rate_per_s
    context = _SCHEDULE_CONTEXT or "unscoped"
    stream = _SCHEDULE_STREAM_INDEX
    _SCHEDULE_STREAM_INDEX += 1
    values = []
    for ordinal, offset in enumerate(base):
        centered = 2.0 * _unit_interval(
            seed, context, stream, ordinal, "arrival") - 1.0
        value = offset + centered * fraction * spacing_ms
        _require(0.0 < value < duration_ms,
                 "held-out arrival escaped its phase")
        values.append(value)
    _require(all(left < right for left, right in zip(values, values[1:])),
             "held-out jitter changed arrival order")
    return values


def _with_schedule_context(
    name: str, operation: Callable[[], object],
) -> object:
    global _SCHEDULE_CONTEXT, _SCHEDULE_STREAM_INDEX
    prior_context = _SCHEDULE_CONTEXT
    prior_stream = _SCHEDULE_STREAM_INDEX
    _SCHEDULE_CONTEXT = name
    _SCHEDULE_STREAM_INDEX = 0
    try:
        return operation()
    finally:
        _SCHEDULE_CONTEXT = prior_context
        _SCHEDULE_STREAM_INDEX = prior_stream


def _materialize_c7(*, spec, section):
    return _with_schedule_context(
        str(spec["name"]),
        lambda: _ORIGINAL_C7_MATERIALIZE(spec=spec, section=section),
    )


def _materialize_remote(*, spec, section):
    return _with_schedule_context(
        str(spec["name"]),
        lambda: _ORIGINAL_REMOTE_MATERIALIZE(spec=spec, section=section),
    )


def _p_only_prompt(
    tokenizer, template: tuple[int, ...], *, sequence: int,
    owner: int, pool_index: int, pool_size: int,
) -> str:
    heldout = _heldout()
    prompt = heldout["p_only_prompt_namespace"]
    _require(isinstance(prompt, dict), "held-out P_ONLY namespace is missing")
    _require(prompt.get("algorithm") == "c8_marker_offset_v1",
             "held-out P_ONLY namespace algorithm differs")
    marker = (
        int(prompt["base_marker"])
        + int(prompt["marker_offset"])
        + sequence * 32
        + owner * pool_size
        + pool_index
    )
    _require(marker < (1 << 18), "held-out P_ONLY marker space exhausted")
    return frozen.c7.fixed._unique_prompt(tokenizer, template, marker)


def _install_transform(heldout: dict[str, object]) -> None:
    global _HELDOUT
    _HELDOUT = heldout
    frozen.c7._uniform_offsets = _jittered_uniform_offsets
    frozen.c7._materialize_schedule = _materialize_c7
    frozen._materialize_remote_schedule = _materialize_remote
    frozen._p_only_prompt = _p_only_prompt


def _execution_receipt(
    *, output: Path, contract_path: Path, contract: dict[str, object],
) -> None:
    bundle = json.loads(output.read_text(encoding="utf-8"))
    heldout = contract["independent_validation"]
    expected_seed = int(heldout["request_seed"])
    expected_blocks = [
        str(row["name"]) for row in contract["joint_control"]["blocks"]
    ]
    _require(bundle.get("schema") == SCHEMA,
             "held-out measured bundle schema differs")
    _require(bundle.get("arm") == os.environ.get(ARM_ENV),
             "held-out measured arm differs")
    _require(bundle.get("qualification_contract_sha256") == _sha256(contract_path),
             "held-out measured contract digest differs")
    artifacts = bundle.get("artifacts")
    serialized_order = bundle.get("block_order")
    _require(
        isinstance(artifacts, dict)
        and set(artifacts) == set(expected_blocks)
        and isinstance(serialized_order, list)
        and [
            str(row.get("name")) for row in serialized_order
            if isinstance(row, dict)
        ] == expected_blocks,
        "held-out measured block population/order differs",
    )
    workload_receipts: dict[str, object] = {}
    remote_prompt_hashes: set[str] = set()
    for name in expected_blocks:
        raw_value = artifacts[name]
        raw_path = Path(str(raw_value)).resolve()
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        workload = raw.get("workload")
        _require(
            isinstance(workload, dict)
            and workload.get("seed") == expected_seed
            and isinstance(workload.get("sha256"), str),
            f"held-out request seed differs in block {name}",
        )
        workload_receipts[str(name)] = {
            "raw": str(raw_path),
            "raw_sha256": _sha256(raw_path),
            "workload_sha256": workload["sha256"],
            "request_seed": workload["seed"],
        }
        if str(name) == heldout["remote_favorable_block"]:
            block_contract = raw.get("c8_dual_regime_contract")
            _require(isinstance(block_contract, dict),
                     "held-out remote block contract is missing")
            for metadata in block_contract["request_index"].values():
                if metadata.get("role") == "victim":
                    remote_prompt_hashes.add(str(metadata["prompt_sha256"]))
    owner_count = 1 if bundle["arm"] in frozen.c7.FIXED_ARMS else 2
    expected_pool_prompts = owner_count * int(
        contract["joint_control"]["remote_activation"]["p_only_pool_per_owner"])
    _require(len(remote_prompt_hashes) == expected_pool_prompts,
             "held-out P_ONLY prompt pool cardinality differs")
    bundle["independent_validation_execution"] = {
        "schema": EXECUTION_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "request_seed": expected_seed,
        "arrival_jitter": heldout["arrival_jitter"],
        "p_only_prompt_namespace": heldout["p_only_prompt_namespace"],
        "controller_receives_workload_seed": False,
        "controller_receives_future_arrivals": False,
        "block_order": expected_blocks,
        "block_workloads": workload_receipts,
        "remote_p_only_prompt_sha256s": sorted(remote_prompt_hashes),
    }
    output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    contract_path, contract = _load_independent_contract()
    heldout = contract["independent_validation"]
    expected_seed = int(heldout["request_seed"])
    actual_seed = int(_argument("--seed"))
    _require(actual_seed == expected_seed,
             "runtime request seed differs from held-out contract")
    _install_transform(heldout)
    output = Path(_argument("--output")).resolve()
    run_id = _argument("--run-id")
    result = frozen.main()
    _require(result == 0, "frozen C8 client returned a nonzero status")
    if not run_id.endswith("-warmup"):
        _require(output.is_file(), "held-out measured bundle is missing")
        _execution_receipt(
            output=output, contract_path=contract_path, contract=contract)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
