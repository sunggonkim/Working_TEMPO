#!/usr/bin/env python3
"""Four-node node shim for a post-hoc paper-baseline extension of C9."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.sota_4node import run_tempo_go_c10_paper_sota_client as client
from eval.sota_4node import vllm_lmcache_elastic_pd_node as canonical
from eval.sota_4node import vllm_lmcache_tempo_go_c6_qualification_node as base
from eval.sota_4node import vllm_lmcache_tempo_go_c8_independent_validation_node as parent
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf
from tempo.pd_paper_baselines import KAIROS_X512, POLICY_ENV


SCHEMA = "tempo-go-c10-paper-sota-node-result-v1"
STAGE_NAME = "tempo_go_c10_paper_sota"
CLIENT_MODULE = "eval.sota_4node.run_tempo_go_c10_paper_sota_client"
CONTRACT_ENV = client.CONTRACT_ENV
EXTENSION_ENV = "TEMPO_GO_C10_PAPER_SOTA_MANIFEST"

_BASE_CLIENT_COMMAND = base._client_command
_CANONICAL_FRONTEND_COMMAND = canonical._frontend_command
_CANONICAL_VLLM_COMMAND = canonical._vllm_command


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification(repo_root: Path):
    contract_path, contract = parent._qualification(repo_root)
    raw_manifest = os.environ.get(EXTENSION_ENV, "")
    perf._require(bool(raw_manifest), "C10 paper SOTA manifest is missing")
    manifest_path = Path(raw_manifest).resolve()
    perf._require(
        manifest_path.is_file() and repo_root in manifest_path.parents,
        "C10 paper SOTA manifest path is invalid",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    perf._require(
        manifest.get("schema") == "tempo-go-c10-paper-sota-extension-v1"
        and manifest.get("claim_boundary", {}).get("post_hoc_extension") is True
        and manifest.get("claim_boundary", {}).get(
            "independent_validation_claim_allowed") is False,
        "C10 paper SOTA claim boundary differs",
    )
    parent_spec = manifest.get("parent_independent_validation", {})
    perf._require(
        parent_spec.get("path") == str(contract_path.relative_to(repo_root))
        and parent_spec.get("sha256") == _sha256(contract_path),
        "C10 parent held-out contract differs",
    )
    policy = os.environ.get(POLICY_ENV, "")
    perf._require(
        policy in manifest.get("policies", {}),
        "C10 runtime policy is not frozen in the extension manifest",
    )
    inventory = manifest.get("source_inventory", {})
    perf._require(isinstance(inventory, dict) and inventory,
                  "C10 source inventory is missing")
    for relative, expected in inventory.items():
        source = (repo_root / str(relative)).resolve()
        perf._require(
            source.is_file() and repo_root in source.parents
            and _sha256(source) == expected,
            f"C10 source drift detected: {relative}",
        )
    return contract_path, contract


def _client_command(*args, **kwargs) -> list[str]:
    command = _BASE_CLIENT_COMMAND(*args, **kwargs)
    perf._require("--seed" not in command,
                  "C10 inherited client unexpectedly supplies a seed")
    contract = json.loads(Path(os.environ[CONTRACT_ENV]).read_text(
        encoding="utf-8"))
    command.extend((
        "--seed", str(contract["independent_validation"]["request_seed"])))
    return command


def _paper_frontend_command(*args, **kwargs) -> list[str]:
    command = _CANONICAL_FRONTEND_COMMAND(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_elastic_frontend"
    perf._require(command.count(old) == 1,
                  "C10 inherited frontend module seam differs")
    command[command.index(old)] = (
        "eval.sota_4node.tempo_pd_paper_baseline_frontend")
    return command


def _paper_vllm_command(*args, **kwargs) -> list[str]:
    command = _CANONICAL_VLLM_COMMAND(*args, **kwargs)
    is_prefill = kwargs.get("is_prefill")
    perf._require(type(is_prefill) is bool,
                  "C10 vLLM role identity is missing")
    if os.environ.get(POLICY_ENV) == KAIROS_X512 and not is_prefill:
        marker = "--max-num-batched-tokens"
        perf._require(command.count(marker) == 1,
                      "C10 Kairos chunk seam differs")
        command[command.index(marker) + 1] = "512"
    return command


def main() -> int:
    base.SCHEMA = SCHEMA
    base.STAGE_NAME = STAGE_NAME
    base.CLIENT_MODULE = CLIENT_MODULE
    base.CONTRACT_ENV = CONTRACT_ENV
    base.client = client
    base._qualification = _qualification
    base._client_command = _client_command
    canonical._frontend_command = _paper_frontend_command
    canonical._vllm_command = _paper_vllm_command
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
