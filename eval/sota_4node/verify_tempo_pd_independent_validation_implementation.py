#!/usr/bin/env python3
"""Fail closed unless the independent-validation implementation is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import verify_tempo_pd_c4_adaptive_implementation as adaptive
from eval.sota_4node import verify_tempo_pd_c4_implementation as fixed


SCHEMA = "tempo-pd-independent-validation-implementation-contract-v1"
REQUIRED_FILES = frozenset({
    "eval/sota_4node/analyze_tempo_pd_independent_validation.py",
    "eval/sota_4node/analyze_tempo_pd_c4_semantic_load.py",
    "eval/sota_4node/build_tempo_pd_independent_validation_manifest.py",
    "eval/sota_4node/build_tempo_pd_independent_validation_run_contract.py",
    "eval/sota_4node/independent_validation_pd_node_entry.sh",
    "eval/sota_4node/prepare_tempo_pd_independent_validation.sh",
    "eval/sota_4node/promote_tempo_pd_profiles_for_independent_validation.py",
    "eval/sota_4node/run_tempo_pd_independent_validation_client.py",
    "eval/sota_4node/run_tempo_pd_independent_validation_in_allocation.sh",
    "eval/sota_4node/tempo_pd_independent_validation_preregistration_v1.json",
    "eval/sota_4node/verify_tempo_pd_independent_validation_implementation.py",
    "eval/sota_4node/vllm_lmcache_pd_independent_validation_node.py",
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
             "independent implementation contract is missing")
    _require(
        _sha256(contract_path)
        == fixed._canonical_sha(
            expected_sha256, name="independent implementation SHA-256"),
        "independent implementation contract digest differs",
    )
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("schema") == SCHEMA
        and value.get("fingerprint_sha256") == contract_fingerprint(value)
        and value.get("purpose")
        == "frozen held-out independent validation"
        and value.get("post_validation_tuning_allowed") is False
        and value.get("performance_claim_allowed_before_analysis") is False,
        "independent implementation schema or claim contract differs",
    )
    adaptive_binding = value.get("adaptive_implementation_contract")
    _require(
        isinstance(adaptive_binding, dict)
        and set(adaptive_binding) == {
            "path", "sha256", "fingerprint_sha256"},
        "independent implementation lacks adaptive parent binding",
    )
    bound_adaptive = fixed._relative_file(
        repo_root, adaptive_binding["path"],
        name="adaptive implementation contract")
    _require(
        bound_adaptive == adaptive_contract
        and _sha256(bound_adaptive) == adaptive_binding["sha256"],
        "independent implementation adaptive-parent digest differs",
    )
    adaptive_value = json.loads(
        bound_adaptive.read_text(encoding="utf-8"))
    _require(
        adaptive_value.get("schema") == adaptive.SCHEMA
        and adaptive_value.get("fingerprint_sha256")
        == adaptive.contract_fingerprint(adaptive_value)
        == adaptive_binding["fingerprint_sha256"],
        "independent implementation adaptive-parent fingerprint differs",
    )
    fixed_binding = adaptive_value.get("fixed_c4_implementation_contract")
    _require(isinstance(fixed_binding, dict),
             "adaptive implementation fixed parent binding is missing")
    fixed_path = fixed._relative_file(
        repo_root, fixed_binding["path"], name="fixed C4 contract")
    adaptive.verify_contract(
        repo_root=repo_root,
        contract_path=bound_adaptive,
        expected_sha256=str(adaptive_binding["sha256"]),
        fixed_c4_contract=fixed_path,
    )

    files = value.get("files")
    _require(isinstance(files, list) and bool(files),
             "independent implementation file bindings are missing")
    observed = set()
    for index, entry in enumerate(files):
        _require(
            isinstance(entry, dict) and set(entry) == {"path", "sha256"},
            f"independent implementation file[{index}] binding differs",
        )
        raw_path = entry["path"]
        _require(type(raw_path) is str and raw_path not in observed,
                 "independent implementation paths are invalid or duplicated")
        path = fixed._relative_file(
            repo_root, raw_path,
            name=f"independent implementation file[{index}]",
        )
        _require(
            _sha256(path) == fixed._canonical_sha(
                entry["sha256"],
                name=f"independent implementation file[{index}] SHA-256",
            ),
            f"independent implementation file drifted: {raw_path}",
        )
        observed.add(raw_path)
    _require(REQUIRED_FILES.issubset(observed),
             "independent implementation contract omits a required file")
    _require(
        value.get("git_heads") == {
            "repository": fixed._git_head(repo_root),
            "third_party_lmcache": fixed._git_head(
                repo_root / "third_party/lmcache"),
        },
        "independent implementation Git provenance differs",
    )
    _require(value.get("environment_versions") == fixed._environment_versions(),
             "independent implementation environment versions differ")
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
