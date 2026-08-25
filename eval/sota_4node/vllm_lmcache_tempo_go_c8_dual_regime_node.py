#!/usr/bin/env python3
"""Four-node native node shim for one source-bound C8 server epoch."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.sota_4node import run_tempo_go_c8_dual_regime_client as client
from eval.sota_4node import vllm_lmcache_tempo_go_c6_qualification_node as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf


SCHEMA = "tempo-go-c8-dual-regime-node-result-v1"
STAGE_NAME = "tempo_go_c8_dual_regime"
CLIENT_MODULE = "eval.sota_4node.run_tempo_go_c8_dual_regime_client"
CONTRACT_ENV = client.CONTRACT_ENV


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification(repo_root: Path) -> tuple[Path, dict[str, object]]:
    raw = os.environ.get(CONTRACT_ENV, "")
    perf._require(bool(raw), "C8 contract environment is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), "C8 contract is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    perf._require(value.get("schema") == client.CONTRACT_SCHEMA,
                  "C8 contract schema differs")
    boundary = value.get("claim_boundary", {})
    perf._require(
        boundary.get("controller_performance_claim_allowed") is True
        and boundary.get("performance_claim_allowed") is True
        and boundary.get("independent_validation_claim_allowed") is False,
        "C8 discovery claim boundary differs",
    )
    inventory = value.get("source_inventory", {})
    perf._require(isinstance(inventory, dict) and inventory,
                  "C8 source inventory is missing")
    for relative, expected in inventory.items():
        source = (repo_root / relative).resolve()
        perf._require(
            isinstance(relative, str)
            and repo_root in source.parents
            and source.is_file()
            and isinstance(expected, str)
            and len(expected) == 64,
            f"C8 source inventory entry is invalid: {relative}",
        )
        perf._require(_sha256(source) == expected,
                      f"C8 source drift detected: {relative}")
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
