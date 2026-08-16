#!/usr/bin/env python3
"""Extract a conservative two-node fabric observation from an existing run.

This is an evidence extractor, never a Slurm launcher.  It deliberately keeps
node-partition spans and NCCL route strings separate from causal counters:
collective CSVs provide rank/step timing, while NCCL logs provide only an
initialization/route witness.  A result is never promotion-eligible without a
rank- or slice-bound monotonic counter (or an explicit intervention).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "tempo-rd-g2-fabric-observation-1"
EXPECTED_WORLD = 8
EXPECTED_NODES = 2
ROUTE_PATTERNS = {
    "gdr_gpu_originated": re.compile(r"GDRDMA|GPU Direct RDMA Enabled", re.I),
    "host_originated": re.compile(r"NET/.*host|HOST|non.?GDR|Host", re.I),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if type(obj) is not dict:
        raise ValueError(f"{path} must contain an object")
    return obj


def _placement(root: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("placement_rank*.env")):
        values: dict[str, str] = {}
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
        rank = int(values.get("rank", path.stem.rsplit("rank", 1)[-1]))
        if rank in out:
            raise ValueError(f"duplicate placement rank {rank}")
        out[rank] = {
            "rank": rank,
            "local_rank": int(values["local_rank"]),
            "host": values["host"],
            "source": path.name,
            "source_sha256": _sha(path),
        }
    if sorted(out) != list(range(EXPECTED_WORLD)):
        raise ValueError("placement must contain exactly ranks 0..7")
    hosts = defaultdict(list)
    for record in out.values():
        hosts[record["host"]].append(record["rank"])
    if len(hosts) != EXPECTED_NODES or sorted(map(len, hosts.values())) != [4, 4]:
        raise ValueError("G2 placement must contain two hosts with four ranks each")
    for host, ranks in hosts.items():
        if sorted(out[rank]["local_rank"] for rank in ranks) != list(range(4)):
            raise ValueError(f"local ranks must be 0..3 on host {host}")
    return out


def _float(value: str, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field}")
    return number


def _collective_rows(policy_root: Path, placement: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in range(EXPECTED_WORLD):
        path = policy_root / f"collectives_rank{rank}.csv"
        if not path.is_file():
            raise ValueError(f"missing {path}")
        with path.open(newline="") as stream:
            for raw in csv.DictReader(stream):
                if int(raw["rank"]) != rank:
                    raise ValueError(f"rank mismatch in {path}")
                step = int(raw["step"])
                phase = int(raw["phase_index"])
                # Diagnostic all_reduce rows do not describe a training
                # collective phase and are retained only in source counts.
                if step < 0 or phase < 0 or raw["collective"] == "all_reduce":
                    continue
                rows.append({
                    "rank": rank,
                    "host": placement[rank]["host"],
                    "node_index": 0 if placement[rank]["host"] == sorted({p["host"] for p in placement.values()})[0] else 1,
                    "step": step,
                    "phase_index": phase,
                    "collective": raw["collective"],
                    "phase_signature": raw["phase_signature"],
                    "ready_corrected_ns": int(raw["ready_corrected_ns"]),
                    "completion_unix_ns": int(raw["completion_callback_unix_ns"]),
                    "gpu_ms": _float(raw["gpu_ms"], "gpu_ms"),
                    "tensor_bytes": int(raw["tensor_bytes"]),
                    "output_tensor_bytes": int(raw["output_tensor_bytes"]),
                })
    return rows


def _group_slices(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["step"], row["phase_index"], row["phase_signature"])
        grouped[key].append(row)
    observations: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for key, group in sorted(grouped.items()):
        if len(group) != EXPECTED_WORLD or {r["rank"] for r in group} != set(range(EXPECTED_WORLD)):
            incomplete.append(":".join(map(str, key)))
            continue
        all_ready = [r["ready_corrected_ns"] for r in group]
        all_done = [r["completion_unix_ns"] for r in group]
        node_spans = {}
        for node in (0, 1):
            subset = [r for r in group if r["node_index"] == node]
            node_spans[str(node)] = {
                "ready_span_ns": max(r["ready_corrected_ns"] for r in subset) - min(r["ready_corrected_ns"] for r in subset),
                "completion_span_ns": max(r["completion_unix_ns"] for r in subset) - min(r["completion_unix_ns"] for r in subset),
                "rank_count": len(subset),
            }
        observations.append({
            "step": key[0],
            "phase_index": key[1],
            "phase_signature": key[2],
            "collective": group[0]["collective"],
            "tensor_bytes": group[0]["tensor_bytes"],
            "output_tensor_bytes": group[0]["output_tensor_bytes"],
            "rank_count": EXPECTED_WORLD,
            "node_partition_method": "placement_host_partition_proxy",
            "global_ready_span_ns": max(all_ready) - min(all_ready),
            "global_completion_span_ns": max(all_done) - min(all_done),
            "node_spans": node_spans,
        })
    return observations, incomplete


def _route_witness(policy_root: Path) -> dict[str, Any]:
    files = sorted(policy_root.glob("train_nccl_*.log"))
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for path in files:
        for line in path.read_text(errors="replace").splitlines():
            for label, pattern in ROUTE_PATTERNS.items():
                if pattern.search(line):
                    counts[label] += 1
                    if len(examples[label]) < 3:
                        examples[label].append(line.strip()[:300])
    return {
        "source_files": [{"path": p.name, "sha256": _sha(p)} for p in files],
        "counts": dict(sorted(counts.items())),
        "examples": {key: value for key, value in sorted(examples.items())},
        "bound_to_rank_bytes": False,
        "bound_to_collective_slice": False,
        "interpretation": "NCCL initialization/route witness only; not a causal counter",
    }


def build_observation(root: Path, policy: str) -> dict[str, Any]:
    root = root.resolve()
    placement = _placement(root)
    policy_root = root / policy
    rows = _collective_rows(policy_root, placement)
    slices, incomplete = _group_slices(rows)
    route = _route_witness(policy_root)
    gate_manifest = root / "gate_manifest.json"
    gate_sources: dict[str, Any] = {}
    gate_manifest_sha256 = ""
    if gate_manifest.is_file():
        gate_manifest_sha256 = _sha(gate_manifest)
        raw_manifest = _read_json(gate_manifest)
        sources = raw_manifest.get("sources", {})
        if isinstance(sources, dict):
            gate_sources = {
                str(name): value["sha256"]
                for name, value in sorted(sources.items())
                if isinstance(value, dict) and isinstance(value.get("sha256"), str)
            }
    runtime_manifest = root / "raw_manifest.json"
    runtime_sha256 = _sha(runtime_manifest) if runtime_manifest.is_file() else ""
    runtime_bundle_sha256 = ""
    if runtime_manifest.is_file():
        runtime_raw = _read_json(runtime_manifest)
        candidate = runtime_raw.get("source_bundle_sha256")
        if isinstance(candidate, str):
            runtime_bundle_sha256 = candidate
    source_files = []
    for path in sorted(root.glob("placement_rank*.env")):
        source_files.append({"path": path.name, "sha256": _sha(path)})
    for path in sorted(policy_root.glob("collectives_rank*.csv")):
        source_files.append({"path": f"{policy}/{path.name}", "sha256": _sha(path)})
    for path in sorted(policy_root.glob("tempo_v4_telemetry_rank*.jsonl")):
        source_files.append({"path": f"{policy}/{path.name}", "sha256": _sha(path)})
    # Newer raw runs may emit opt-in node-slice HSN intervals from the
    # collective observer.  Keep their hashes in the evidence bundle even
    # though this raw schema deliberately does not promote host-wide counters
    # to a causal rank/slice contract.
    for path in sorted(policy_root.glob("fabric_phase_counters_rank*.json")):
        source_files.append({"path": f"{policy}/{path.name}", "sha256": _sha(path)})
    if runtime_manifest.is_file():
        source_files.append({"path": runtime_manifest.name, "sha256": runtime_sha256})
    return {
        "schema_version": SCHEMA,
        "evidence_state": "raw_observation",
        "promotion_eligible": False,
        "world_size": EXPECTED_WORLD,
        "nodes": EXPECTED_NODES,
        "policy": policy,
        "placement": sorted(placement.values(), key=lambda item: item["rank"]),
        "source_files": source_files,
        "gate_manifest": {
            "path": gate_manifest.name if gate_manifest.is_file() else "",
            "sha256": gate_manifest_sha256,
            "source_hashes": gate_sources,
            "source_hash_binding": bool(gate_sources),
        },
        "runtime_manifest": {
            "path": runtime_manifest.name if runtime_manifest.is_file() else "",
            "sha256": runtime_sha256,
            "source_bundle_sha256": runtime_bundle_sha256,
            "source_bundle_binding": bool(runtime_bundle_sha256),
        },
        "collective_observations": slices,
        "incomplete_collective_groups": incomplete,
        "fabric_splits": {
            "gdr_gpu_originated": {"supported": False, "reason": "route witness is not rank/slice byte-bound"},
            "host_originated": {"supported": False, "reason": "no rank/slice host-byte counter"},
            "pfs_endpoint": {"supported": False, "reason": "logical stage only; no Lustre/CXI counter"},
        },
        "route_witness": route,
        "counter_contract": {
            "rank_or_slice_bound_monotonic_counter": False,
            "intervention": False,
            "host_wide_hsn_counters": False,
            "causal_claim_allowed": False,
        },
        "limitations": [
            "collective rows provide timing and bytes, not network-route bytes",
            "node partition spans are a placement proxy, not isolated intra/inter traffic",
            "NCCL route strings describe initialization, not per-collective service",
            "optional fabric_phase_counters files are node-slice HSN evidence only and are not a causal promotion",
            "PFS is represented as a logical persistence stage until Lustre/CXI counters exist",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", default="tempo_v4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_observation(args.root, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "groups": len(result["collective_observations"]), "promotion_eligible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
