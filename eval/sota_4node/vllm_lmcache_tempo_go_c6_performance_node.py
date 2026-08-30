#!/usr/bin/env python3
"""Native four-node node shim for the frozen C6 performance campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.sota_4node import run_tempo_go_c6_performance_client as client
from eval.sota_4node import vllm_lmcache_tempo_go_c6_qualification_node as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf


SCHEMA = "tempo-go-c6-performance-node-result-v1"
STAGE_NAME = "tempo_go_c6_performance"
CLIENT_MODULE = "eval.sota_4node.run_tempo_go_c6_performance_client"
CONTRACT_ENV = "TEMPO_GO_C6_PERFORMANCE_CONTRACT"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification(repo_root: Path) -> tuple[Path, dict[str, object]]:
    raw = os.environ.get(
        CONTRACT_ENV,
        "eval/sota_4node/tempo_go_c6_performance_contract_v1.json",
    )
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), "C6 performance contract is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    perf._require(value.get("schema") == client.CONTRACT_SCHEMA,
                  "C6 performance contract schema differs")
    boundary = value.get("claim_boundary", {})
    perf._require(
        boundary.get("controller_performance_claim_allowed") is True
        and boundary.get("performance_claim_allowed") is True,
        "C6 performance contract does not authorize this discovery run",
    )
    inventory = value.get("source_inventory")
    if inventory is not None:
        perf._require(
            isinstance(inventory, dict) and inventory,
            "C6 source inventory is invalid",
        )
        for relative, expected in inventory.items():
            perf._require(
                isinstance(relative, str)
                and relative
                and isinstance(expected, str)
                and len(expected) == 64,
                "C6 source inventory entry is invalid",
            )
            source = (repo_root / relative).resolve()
            perf._require(
                repo_root in source.parents and source.is_file(),
                f"C6 source inventory path is invalid: {relative}",
            )
            perf._require(
                _sha256(source) == expected,
                f"C6 source drift detected: {relative}",
            )
    return path, value


def main() -> int:
    base.SCHEMA = SCHEMA
    base.STAGE_NAME = STAGE_NAME
    base.CLIENT_MODULE = CLIENT_MODULE
    base.CONTRACT_ENV = CONTRACT_ENV
    base.client = client
    base._qualification = _qualification
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
