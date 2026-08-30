#!/usr/bin/env python3
"""Summarize EOF-scoped frontend semantic pressure in C4 block artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


SCHEMA = "tempo-pd-c4-semantic-load-analysis-v1"
LOAD_SCHEMA = "tempo-frontend-semantic-load-v1"
LOAD_SOURCE = "frontend_pair_ledger_request_start_to_http_eof"
PHASES = (
    "c0_cool",
    "c1_decoder_hot",
    "c2_remote_hot",
    "c2_kv_remote_hot",
    "c3_both_hot",
    "recovery",
)
_CONTRACT_LAYOUTS = {
    "c4_phase_screen_contract": ("arm", "block_sequence_index"),
    "c4_fixed_phase_contract": ("foreground_arm", "sequence"),
    "c4_adaptive_screen_contract": ("arm", "sequence"),
    "independent_validation_contract": ("arm", "sequence"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nearest_rank(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    copied = [float(value) for value in values]
    return {
        "count": len(copied),
        "minimum": min(copied) if copied else None,
        "median": _nearest_rank(copied, 0.5),
        "p90": _nearest_rank(copied, 0.9),
        "p99": _nearest_rank(copied, 0.99),
        "maximum": max(copied) if copied else None,
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    max_num_seqs = {int(row["max_num_seqs"]) for row in rows}
    _require(len(max_num_seqs) == 1, "max_num_seqs differs within group")
    capacity = next(iter(max_num_seqs))
    active = [int(row["active_requests_before"]) for row in rows]
    decode = [int(row["decode_tokens_before"]) for row in rows]
    occupancy = [float(row["occupancy_ratio_before"]) for row in rows]
    return {
        "requests": len(rows),
        "max_num_seqs": capacity,
        "active_requests_before": _distribution(active),
        "decode_tokens_before": _distribution(decode),
        "occupancy_ratio_before": _distribution(occupancy),
        "capacity_event_fraction": {
            "at_least_half": sum(value * 2 >= capacity for value in active)
            / len(active),
            "at_least_three_quarters": sum(
                value * 4 >= capacity * 3 for value in active
            ) / len(active),
            "at_or_above_max_num_seqs": sum(
                value >= capacity for value in active
            ) / len(active),
        },
        "pair_counts": dict(sorted(Counter(
            int(row["pair_index"]) for row in rows
        ).items())),
        "tenant_counts": dict(sorted(Counter(
            str(row["tenant"]) for row in rows
        ).items())),
    }


def _block_rows(raw: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    present = [name for name in _CONTRACT_LAYOUTS if name in raw]
    _require(len(present) == 1, "C4 semantic block contract is not unique")
    contract_name = present[0]
    contract = raw.get(contract_name)
    decisions = raw.get("router_decisions")
    _require(isinstance(contract, dict), "C4 block contract is missing")
    _require(isinstance(decisions, list), "router_decisions must be a list")
    request_index = contract.get("request_index")
    _require(isinstance(request_index, dict), "request_index must be an object")
    decision_index = {}
    for decision in decisions:
        _require(isinstance(decision, dict), "router decision must be an object")
        request_id = decision.get("request_id")
        _require(isinstance(request_id, str), "router decision ID is invalid")
        _require(request_id not in decision_index, "duplicate router decision ID")
        decision_index[request_id] = decision
    _require(set(request_index) <= set(decision_index),
             "indexed request lacks a router decision")

    arm_field, sequence_field = _CONTRACT_LAYOUTS[contract_name]
    arm = contract.get(arm_field)
    sequence = contract.get(sequence_field)
    replicate = contract.get("replicate")
    _require(isinstance(arm, str) and arm,
             "semantic block arm differs")
    _require(type(sequence) is int and sequence >= 0,
             "semantic block sequence differs")
    _require(type(replicate) is int and replicate >= 0,
             "semantic block replicate differs")
    normalized_contract = dict(contract)
    normalized_contract["arm"] = arm
    normalized_contract["block_sequence_index"] = sequence
    normalized_contract["semantic_contract_name"] = contract_name

    rows = []
    for request_id, metadata in request_index.items():
        _require(isinstance(metadata, dict), "request metadata must be an object")
        phase = metadata.get("phase")
        tenant = metadata.get("tenant")
        _require(phase in PHASES, "request phase differs")
        _require(isinstance(tenant, str) and tenant, "request tenant differs")
        decision = decision_index[request_id]
        _require(
            decision.get("frontend_semantic_load_schema") == LOAD_SCHEMA,
            "semantic-load schema is missing",
        )
        _require(
            decision.get("frontend_semantic_load_source") == LOAD_SOURCE,
            "semantic-load source differs",
        )
        pair = decision.get("frontend_semantic_pair_index")
        active = decision.get("frontend_semantic_active_requests_before")
        decode = decision.get("frontend_semantic_decode_tokens_before")
        capacity = decision.get("frontend_semantic_max_num_seqs")
        occupancy = decision.get("frontend_semantic_occupancy_ratio_before")
        _require(type(pair) is int and pair in (0, 1), "semantic pair differs")
        _require(pair == decision.get("frontend_pair_index"),
                 "semantic/frontend pair assignment differs")
        _require(type(active) is int and active >= 0,
                 "active request count differs")
        _require(type(decode) is int and decode >= 0,
                 "decode-token pressure differs")
        _require(type(capacity) is int and capacity in (8, 16),
                 "max_num_seqs differs")
        _require(
            not isinstance(occupancy, bool)
            and isinstance(occupancy, (int, float))
            and math.isfinite(float(occupancy))
            and math.isclose(float(occupancy), active / capacity),
            "occupancy ratio differs",
        )
        rows.append({
            "request_id": request_id,
            "phase": phase,
            "tenant": tenant,
            "foreground": tenant == "foreground",
            "pair_index": pair,
            "active_requests_before": active,
            "decode_tokens_before": decode,
            "max_num_seqs": capacity,
            "occupancy_ratio_before": float(occupancy),
        })
    _require(len(rows) == len(request_index), "semantic row count differs")
    return normalized_contract, rows


def analyze(paths: list[Path]) -> dict[str, object]:
    _require(bool(paths), "at least one C4 block is required")
    blocks = []
    pooled_foreground: dict[tuple[str, int], list[dict[str, object]]] = (
        defaultdict(list))
    pooled_all: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    pooled_phase_foreground: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_block_keys = set()
    for source in paths:
        source = source.resolve()
        _require(source.is_file(), "C4 block is missing")
        raw = json.loads(source.read_text(encoding="utf-8"))
        _require(isinstance(raw, dict), "C4 block must be an object")
        contract, rows = _block_rows(raw)
        arm = contract.get("arm")
        replicate = contract.get("replicate")
        sequence = contract.get("block_sequence_index")
        _require(isinstance(arm, str) and arm, "block arm differs")
        _require(type(replicate) is int and replicate >= 0,
                 "block replicate differs")
        _require(type(sequence) is int and sequence >= 0,
                 "block sequence differs")
        block_key = (replicate, arm)
        _require(block_key not in seen_block_keys, "duplicate arm/replicate block")
        seen_block_keys.add(block_key)
        phase_pair_all: dict[tuple[str, int], list[dict[str, object]]] = (
            defaultdict(list))
        phase_pair_foreground: dict[
            tuple[str, int], list[dict[str, object]]
        ] = defaultdict(list)
        phase_foreground: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            key = (str(row["phase"]), int(row["pair_index"]))
            phase_pair_all[key].append(row)
            pooled_all[key].append(row)
            if row["foreground"]:
                phase_pair_foreground[key].append(row)
                phase_foreground[str(row["phase"])].append(row)
                pooled_foreground[key].append(row)
                pooled_phase_foreground[str(row["phase"])].append(row)
        blocks.append({
            "arm": arm,
            "replicate": replicate,
            "block_sequence_index": sequence,
            "source": {"path": str(source), "sha256": _sha256(source)},
            "requests": len(rows),
            "phase_pair_all_tenants": {
                f"{phase}:pair{pair}": _summary(group)
                for (phase, pair), group in sorted(phase_pair_all.items())
            },
            "phase_pair_foreground": {
                f"{phase}:pair{pair}": _summary(group)
                for (phase, pair), group in sorted(
                    phase_pair_foreground.items())
            },
            "phase_foreground": {
                phase: _summary(group)
                for phase, group in sorted(phase_foreground.items())
            },
        })
    return {
        "schema": SCHEMA,
        "evidence_scope": "pair_local_frontend_ledger_request_start_to_http_eof",
        "policy_input_used": False,
        "blocks": sorted(
            blocks, key=lambda value: int(value["block_sequence_index"])),
        "pooled_phase_pair_all_tenants": {
            f"{phase}:pair{pair}": _summary(group)
            for (phase, pair), group in sorted(pooled_all.items())
        },
        "pooled_phase_pair_foreground": {
            f"{phase}:pair{pair}": _summary(group)
            for (phase, pair), group in sorted(pooled_foreground.items())
        },
        "pooled_phase_foreground": {
            phase: _summary(group)
            for phase, group in sorted(pooled_phase_foreground.items())
        },
        "interpretation_limits": {
            "calibration_only": True,
            "performance_claim_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
            "threshold_selected": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite semantic analysis")
    value = analyze(args.block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "sha256": _sha256(args.output.resolve()),
        "blocks": len(value["blocks"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
