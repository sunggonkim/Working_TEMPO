#!/usr/bin/env python3
"""Bind a C9 causal-burst template to a current source-bound C8 base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _c9_source_inventory(repo_root: Path) -> dict[str, str]:
    relative_paths = (
        "eval/sota_4node/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh",
        "eval/sota_4node/run_lmcache_nixl_contention_2node_in_allocation.sh",
        "eval/sota_4node/c9_gate_node_entry.sh",
        "eval/sota_4node/vllm_lmcache_tempo_go_c9_gate_node.py",
        "eval/sota_4node/analyze_tempo_go_c9_causal_burst_discovery.py",
    )
    inventory: dict[str, str] = {}
    for relative in relative_paths:
        source = (repo_root / relative).resolve()
        _require(source.is_file(), f"C9 source file is missing: {relative}")
        inventory[relative] = _sha256(source)
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    template = args.template.resolve()
    base_path = args.base_contract.resolve()
    output = args.output.resolve()
    _require(template.is_file(), "C9 template is missing")
    _require(base_path.is_file(), "C9 base contract is missing")
    _require(not output.exists(), "refusing to overwrite C9 contract")
    _require(repo_root in template.parents and repo_root in base_path.parents,
             "C9 inputs must be inside the repository")
    _require(repo_root in output.parents,
             "C9 output must be inside the repository")

    contract: dict[str, Any] = json.loads(template.read_text(encoding="utf-8"))
    base: dict[str, Any] = json.loads(base_path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == "tempo-go-c9-causal-burst-discovery-v1",
             "C9 template schema differs")
    _require(base.get("schema") == "tempo-go-c8-dual-regime-contract-v1",
             "C9 base must be a C8 contract")
    _require(base.get("source_inventory"), "C9 base source inventory is missing")
    _require(isinstance(contract.get("system_under_test"), dict),
             "C9 system-under-test section is missing")
    _require(isinstance(contract.get("execution"), dict),
             "C9 execution section is missing")
    _require(isinstance(contract.get("burst"), dict),
             "C9 burst section is missing")
    _require(contract["execution"].get("one_campaign_no_retry") is True,
             "C9 must remain one-campaign/no-retry")
    _require(contract["claim_boundary"].get("discovery_only") is True,
             "C9 builder only creates discovery contracts")

    relative_base = base_path.relative_to(repo_root).as_posix()
    system = contract["system_under_test"]
    system["base_contract"] = relative_base
    system["base_contract_sha256"] = _sha256(base_path)
    # The C9 campaign needs the C9 gate shim: it reuses the C8 dual-regime
    # lifecycle but adds the measurement-start gate.  Never let the runner's
    # historical C8-independent fallback select a different preregistration.
    system["node_entry"] = "eval/sota_4node/c9_gate_node_entry.sh"
    system["source_policy"] = "current_source_bound_c8_base_v1"
    source_inventory = _c9_source_inventory(repo_root)
    contract["provenance"] = {
        "schema": "tempo-go-c9-current-source-binding-v1",
        "template": template.relative_to(repo_root).as_posix(),
        "template_sha256": _sha256(template),
        "base_contract": relative_base,
        "base_contract_sha256": _sha256(base_path),
        "source_inventory": source_inventory,
        "c9_source_inventory_count": len(source_inventory),
        "base_source_inventory_count": len(base["source_inventory"]),
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(output)
    print("contract_sha256", _sha256(output))
    print("base_contract", relative_base)
    print("c9_source_inventory_count", len(source_inventory))
    print("base_source_inventory_count", len(base["source_inventory"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
