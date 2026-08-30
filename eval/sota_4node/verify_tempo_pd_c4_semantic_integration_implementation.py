#!/usr/bin/env python3
"""Fail closed unless the post-C4 semantic integration path is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import verify_tempo_pd_c4_adaptive_implementation as adaptive
from eval.sota_4node import verify_tempo_pd_c4_implementation as fixed


SCHEMA = "tempo-pd-c4-semantic-integration-implementation-contract-v1"
REQUIRED_FILES = frozenset({
    "eval/sota_4node/analyze_tempo_pd_c4_semantic_epoch_screen.py",
    "eval/sota_4node/analyze_tempo_pd_c4_semantic_integration_screen.py",
    "eval/sota_4node/build_tempo_pd_c4_semantic_integration_run_contract.py",
    "eval/sota_4node/build_tempo_pd_semantic_epoch_endpoint_profile.py",
    "eval/sota_4node/run_tempo_pd_c4_semantic_integration_campaign_in_allocation.sh",
    "eval/sota_4node/run_tempo_pd_c4_semantic_integration_screen_in_allocation.sh",
    "eval/sota_4node/verify_tempo_pd_c4_semantic_integration_implementation.py",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_contract(
    *, repo_root: Path, contract_path: Path, expected_sha256: str,
    adaptive_contract: Path,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    contract_path = contract_path.resolve()
    adaptive_contract = adaptive_contract.resolve()
    _require(repo_root.is_dir(), "repository root is missing")
    _require(contract_path.is_file(),
             "semantic integration implementation contract is missing")
    _require(
        _sha256(contract_path)
        == fixed._canonical_sha(
            expected_sha256,
            name="semantic integration implementation SHA-256",
        ),
        "semantic integration implementation contract digest differs",
    )
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("schema") == SCHEMA
        and value.get("fingerprint_sha256") == contract_fingerprint(value),
        "semantic integration implementation schema or fingerprint differs",
    )
    _require(
        value.get("purpose")
        == "post-C4 calibrated semantic-epoch integration screen"
        and value.get("performance_claim_allowed") is False,
        "semantic integration implementation purpose or claim differs",
    )
    parent = value.get("adaptive_implementation_contract")
    _require(
        isinstance(parent, dict)
        and set(parent) == {"path", "sha256", "fingerprint_sha256"},
        "semantic integration implementation lacks its adaptive parent",
    )
    bound_parent = fixed._relative_file(
        repo_root, parent["path"], name="adaptive implementation contract")
    _require(
        bound_parent == adaptive_contract
        and _sha256(bound_parent) == parent["sha256"],
        "semantic integration adaptive-parent digest differs",
    )
    adaptive_value = json.loads(bound_parent.read_text(encoding="utf-8"))
    fixed_binding = adaptive_value.get("fixed_c4_implementation_contract")
    _require(isinstance(fixed_binding, dict),
             "adaptive parent lacks its fixed-C4 binding")
    fixed_path = fixed._relative_file(
        repo_root, fixed_binding.get("path"), name="fixed C4 contract")
    verified_parent = adaptive.verify_contract(
        repo_root=repo_root,
        contract_path=bound_parent,
        expected_sha256=str(parent["sha256"]),
        fixed_c4_contract=fixed_path,
    )
    _require(
        verified_parent.get("fingerprint_sha256")
        == parent["fingerprint_sha256"],
        "semantic integration adaptive-parent fingerprint differs",
    )

    files = value.get("files")
    _require(isinstance(files, list) and bool(files),
             "semantic integration implementation file bindings are missing")
    observed: set[str] = set()
    for index, entry in enumerate(files):
        _require(
            isinstance(entry, dict) and set(entry) == {"path", "sha256"},
            f"semantic integration implementation file[{index}] binding differs",
        )
        raw_path = entry["path"]
        _require(type(raw_path) is str and raw_path not in observed,
                 "semantic integration paths are invalid or duplicated")
        path = fixed._relative_file(
            repo_root, raw_path,
            name=f"semantic integration implementation file[{index}]",
        )
        _require(
            _sha256(path) == fixed._canonical_sha(
                entry["sha256"],
                name=(
                    f"semantic integration implementation file[{index}] "
                    "SHA-256"
                ),
            ),
            f"semantic integration implementation file drifted: {raw_path}",
        )
        observed.add(raw_path)
    _require(REQUIRED_FILES.issubset(observed),
             "semantic integration contract omits a required file")
    _require(
        value.get("git_heads") == {
            "repository": fixed._git_head(repo_root),
            "third_party_lmcache": fixed._git_head(
                repo_root / "third_party/lmcache"),
        },
        "semantic integration Git provenance differs",
    )
    _require(value.get("environment_versions") == fixed._environment_versions(),
             "semantic integration environment versions differ")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--adaptive-contract", type=Path, required=True)
    args = parser.parse_args()
    value = verify_contract(
        repo_root=args.repo_root,
        contract_path=args.contract,
        expected_sha256=args.expected_sha256,
        adaptive_contract=args.adaptive_contract,
    )
    print(json.dumps({
        "schema": SCHEMA,
        "fingerprint_sha256": value["fingerprint_sha256"],
        "files": len(value["files"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
