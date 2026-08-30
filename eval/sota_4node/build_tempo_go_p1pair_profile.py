#!/usr/bin/env python3
"""Derive a one-inference-pair profile for the 4-node co-job campaign.

The profile still names the complete native 4-node/16-GPU allocation.  It
represents one TP4 P/D pair as TEMPO-managed inference capacity while the
other two nodes are an independent, exogenous NCCL/LMCache co-job whose
fabric pressure is observed by the same global decision loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tempo.pd_global_profile import (
    global_profile_fingerprint,
    load_global_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile-id",
        default="tempo-go-qwen25-perlmutter-p1pair-cojob-discovery-v1",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_profile = load_global_profile(source)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("source global profile is not an object")
    raw["profile_id"] = args.profile_id
    topology = raw["topology"]
    if not isinstance(topology, dict):
        raise ValueError("source topology is not an object")
    topology.update({"pair_count": 1, "prewarmed_pair_count": 1})
    raw["capacities"] = list(raw["capacities"][:1])
    controller = raw["controller"]
    if not isinstance(controller, dict):
        raise ValueError("source controller is not an object")
    controller.update({
        "minimum_active_pairs": 1,
        "maximum_active_pairs": 1,
    })
    raw["fingerprint_sha256"] = global_profile_fingerprint(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    derived = load_global_profile(output)
    if derived.topology.pair_count != 1:
        raise RuntimeError("derived profile did not retain one-pair topology")
    print(json.dumps({
        "output": str(output),
        "source": str(source),
        "source_fingerprint_sha256": source_profile.fingerprint_sha256,
        "fingerprint_sha256": derived.fingerprint_sha256,
        "pair_count": derived.topology.pair_count,
        "allocation_topology": {
            "node_count": derived.topology.node_count,
            "gpu_count": derived.topology.gpu_count,
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
