#!/usr/bin/env python3
"""Fail-closed validation for raw two-node fabric observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "tempo-rd-g2-fabric-observation-1"
EXPECTED_KEYS = {
    "schema_version", "evidence_state", "promotion_eligible", "world_size", "nodes",
    "policy", "placement", "source_files", "gate_manifest", "collective_observations",
    "runtime_manifest", "incomplete_collective_groups", "fabric_splits", "route_witness", "counter_contract",
    "limitations",
}


def _sha(value: object, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


def validate_observation(raw: object) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != EXPECTED_KEYS:
        raise ValueError("raw fabric observation keys are not exact")
    if raw["schema_version"] != SCHEMA or raw["evidence_state"] != "raw_observation":
        raise ValueError("unsupported raw observation schema/state")
    if raw["promotion_eligible"] is not False or raw["world_size"] != 8 or raw["nodes"] != 2:
        raise ValueError("raw observation cannot be promoted and must be 2-node/8-rank")
    if raw["policy"] not in {"fg_only", "open_combined", "d2h_only", "persist_only", "combined", "v4_open", "tempo_v4"}:
        raise ValueError("unknown raw observation policy")
    placement = raw["placement"]
    if type(placement) is not list or len(placement) != 8:
        raise ValueError("placement must contain eight ranks")
    ranks = []
    hosts: dict[str, list[int]] = {}
    for item in placement:
        if type(item) is not dict or set(item) != {"rank", "local_rank", "host", "source", "source_sha256"}:
            raise ValueError("placement record keys are not exact")
        if type(item["rank"]) is not int or type(item["local_rank"]) is not int or type(item["host"]) is not str:
            raise ValueError("placement types are invalid")
        _sha(item["source_sha256"], "placement source")
        ranks.append(item["rank"])
        hosts.setdefault(item["host"], []).append(item["rank"])
    if sorted(ranks) != list(range(8)) or len(hosts) != 2 or sorted(len(v) for v in hosts.values()) != [4, 4]:
        raise ValueError("placement must be two hosts with ranks 0..7")
    sources = raw["source_files"]
    if type(sources) is not list or not sources:
        raise ValueError("source_files must be non-empty")
    for item in sources:
        if type(item) is not dict or set(item) != {"path", "sha256"} or not item["path"]:
            raise ValueError("source file record is invalid")
        _sha(item["sha256"], "source file")
    gate = raw["gate_manifest"]
    if type(gate) is not dict or set(gate) != {"path", "sha256", "source_hashes", "source_hash_binding"}:
        raise ValueError("gate_manifest keys are not exact")
    if gate["path"]:
        _sha(gate["sha256"], "gate manifest")
    if type(gate["source_hash_binding"]) is not bool or type(gate["source_hashes"]) is not dict:
        raise ValueError("gate manifest binding types are invalid")
    for value in gate["source_hashes"].values():
        if type(value) is not str:
            raise ValueError("gate source hash is not a string")
        _sha(value, "gate source")
    runtime = raw["runtime_manifest"]
    if type(runtime) is not dict or set(runtime) != {"path", "sha256", "source_bundle_sha256", "source_bundle_binding"}:
        raise ValueError("runtime_manifest keys are not exact")
    if runtime["path"]:
        _sha(runtime["sha256"], "runtime manifest")
        _sha(runtime["source_bundle_sha256"], "runtime source bundle")
    if type(runtime["source_bundle_binding"]) is not bool:
        raise ValueError("runtime source binding type is invalid")
    observations = raw["collective_observations"]
    if type(observations) is not list or not observations:
        raise ValueError("collective observations are empty")
    if raw["incomplete_collective_groups"] != []:
        raise ValueError("incomplete collective groups cannot be used as raw evidence")
    for item in observations:
        if type(item) is not dict or item.get("rank_count") != 8:
            raise ValueError("collective observation is not complete eight-rank data")
        for key in ("global_ready_span_ns", "global_completion_span_ns"):
            if type(item.get(key)) is not int or item[key] < 0:
                raise ValueError("collective span is invalid")
    contract = raw["counter_contract"]
    if type(contract) is not dict or set(contract) != {
        "rank_or_slice_bound_monotonic_counter", "intervention", "host_wide_hsn_counters", "causal_claim_allowed"
    }:
        raise ValueError("counter contract keys are not exact")
    if any(type(contract[key]) is not bool for key in contract):
        raise ValueError("counter contract values must be bool")
    if contract["causal_claim_allowed"] is not False:
        raise ValueError("raw observer cannot claim causality")
    for name, split in raw["fabric_splits"].items():
        if name not in {"gdr_gpu_originated", "host_originated", "pfs_endpoint"} or type(split) is not dict:
            raise ValueError("fabric split is invalid")
        if split.get("supported") is not False:
            raise ValueError("unsupported raw split was marked supported")
    route = raw["route_witness"]
    if type(route) is not dict or route.get("bound_to_rank_bytes") is not False or route.get("bound_to_collective_slice") is not False:
        raise ValueError("route witness was incorrectly promoted")
    return {
        "schema_version": SCHEMA,
        "policy": raw["policy"],
        "groups": len(observations),
        "promotion_eligible": False,
        "causal_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("observation", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_observation(json.loads(args.observation.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
