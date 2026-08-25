#!/usr/bin/env python3
"""Four-node node shim for the preregistered C8 independent validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.sota_4node import run_tempo_go_c8_independent_validation_client as client
from eval.sota_4node import vllm_lmcache_tempo_go_c6_qualification_node as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf


SCHEMA = "tempo-go-c8-independent-validation-node-result-v1"
STAGE_NAME = "tempo_go_c8_independent_validation"
CLIENT_MODULE = "eval.sota_4node.run_tempo_go_c8_independent_validation_client"
CONTRACT_ENV = client.CONTRACT_ENV

_FROZEN_CLIENT_COMMAND = base._client_command


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification(repo_root: Path) -> tuple[Path, dict[str, object]]:
    raw = os.environ.get(CONTRACT_ENV, "")
    perf._require(bool(raw), "independent C8 contract environment is missing")
    path = Path(raw).resolve()
    perf._require(path.is_file(), "independent C8 contract is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    perf._require(value.get("schema") == client.CONTRACT_SCHEMA,
                  "independent C8 base schema differs")
    heldout = value.get("independent_validation")
    perf._require(
        isinstance(heldout, dict)
        and heldout.get("schema") == client.INDEPENDENT_SCHEMA
        and heldout.get("preregistered_before_fresh_allocation") is True
        and heldout.get("fresh_allocation_required") is True
        and heldout.get("one_shot_no_retry") is True,
        "independent C8 preregistration differs",
    )
    boundary = value.get("claim_boundary", {})
    perf._require(
        boundary.get("controller_performance_claim_allowed") is True
        and boundary.get("performance_claim_allowed") is False
        and boundary.get("independent_validation_claim_allowed") is False,
        "independent C8 preregistration pre-authorizes a claim",
    )
    job_id = os.environ.get("SLURM_JOB_ID", "")
    forbidden = {str(item) for item in heldout["forbidden_discovery_job_ids"]}
    perf._require(job_id.isdigit() and job_id not in forbidden,
                  "independent C8 requires a fresh Slurm allocation")
    inventory = value.get("source_inventory", {})
    perf._require(isinstance(inventory, dict) and inventory,
                  "independent C8 source inventory is missing")
    for relative, expected in inventory.items():
        source = (repo_root / str(relative)).resolve()
        perf._require(
            isinstance(relative, str)
            and repo_root in source.parents
            and source.is_file()
            and isinstance(expected, str)
            and len(expected) == 64,
            f"independent C8 source inventory entry is invalid: {relative}",
        )
        perf._require(_sha256(source) == expected,
                      f"independent C8 source drift detected: {relative}")
    return path, value


def _client_command(*args, **kwargs) -> list[str]:
    command = _FROZEN_CLIENT_COMMAND(*args, **kwargs)
    perf._require("--seed" not in command,
                  "frozen C8 lifecycle unexpectedly supplies a seed")
    contract_path = Path(os.environ[CONTRACT_ENV]).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    seed = int(contract["independent_validation"]["request_seed"])
    command.extend(("--seed", str(seed)))
    return command


def main() -> int:
    base.SCHEMA = SCHEMA
    base.STAGE_NAME = STAGE_NAME
    base.CLIENT_MODULE = CLIENT_MODULE
    base.CONTRACT_ENV = CONTRACT_ENV
    base.client = client
    base._qualification = _qualification
    base._client_command = _client_command
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
