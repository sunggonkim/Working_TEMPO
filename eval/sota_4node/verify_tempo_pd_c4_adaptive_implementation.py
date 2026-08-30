#!/usr/bin/env python3
"""Fail closed unless the post-C4 adaptive-screen implementation is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import verify_tempo_pd_c4_implementation as fixed


SCHEMA = "tempo-pd-c4-adaptive-implementation-contract-v1"
REQUIRED_FILES = frozenset({
    "eval/sota_4node/analyze_tempo_pd_c4_adaptive_screen.py",
    "eval/sota_4node/build_tempo_pd_c4_adaptive_run_contract.py",
    "eval/sota_4node/build_tempo_pd_c4_adaptive_screen_manifest.py",
    "eval/sota_4node/build_tempo_pd_c4_calibrated_profiles.py",
    "eval/sota_4node/c4_adaptive_screen_pd_node_entry.sh",
    "eval/sota_4node/replay_tempo_pd_c4_calibrated_controller.py",
    "eval/sota_4node/run_tempo_pd_c4_adaptive_screen_client.py",
    "eval/sota_4node/run_tempo_pd_c4_adaptive_screen_in_allocation.sh",
    "eval/sota_4node/run_tempo_pd_c4_persistent_campaign_in_allocation.sh",
    "eval/sota_4node/verify_tempo_pd_c4_adaptive_implementation.py",
    "eval/sota_4node/vllm_lmcache_pd_c4_adaptive_screen_node.py",
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
    fixed_c4_contract: Path,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    contract_path = contract_path.resolve()
    fixed_c4_contract = fixed_c4_contract.resolve()
    _require(repo_root.is_dir(), "repository root is missing")
    _require(contract_path.is_file(),
             "adaptive implementation contract is missing")
    _require(
        _sha256(contract_path)
        == fixed._canonical_sha(
            expected_sha256, name="adaptive implementation SHA-256"),
        "adaptive implementation contract digest differs",
    )
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("schema") == SCHEMA
        and value.get("fingerprint_sha256") == contract_fingerprint(value),
        "adaptive implementation schema or fingerprint differs",
    )
    _require(
        value.get("purpose") == "post-C4 calibration-only adaptive screen"
        and value.get("performance_claim_allowed") is False,
        "adaptive implementation purpose or claim contract differs",
    )
    fixed_binding = value.get("fixed_c4_implementation_contract")
    _require(
        isinstance(fixed_binding, dict)
        and set(fixed_binding) == {"path", "sha256", "fingerprint_sha256"},
        "adaptive implementation lacks its fixed-C4 binding",
    )
    bound_fixed = fixed._relative_file(
        repo_root, fixed_binding["path"], name="fixed C4 contract")
    _require(
        bound_fixed == fixed_c4_contract
        and _sha256(bound_fixed) == fixed_binding["sha256"],
        "adaptive implementation fixed-C4 digest differs",
    )
    fixed_value = json.loads(bound_fixed.read_text(encoding="utf-8"))
    _require(
        fixed_value.get("schema") == fixed.SCHEMA
        and fixed_value.get("fingerprint_sha256")
        == fixed.contract_fingerprint(fixed_value)
        == fixed_binding["fingerprint_sha256"],
        "adaptive implementation fixed-C4 fingerprint differs",
    )
    fixed_manifest_entry = fixed_value.get("phase_manifest")
    _require(isinstance(fixed_manifest_entry, dict),
             "fixed-C4 phase-manifest binding is missing")
    fixed_manifest = fixed._relative_file(
        repo_root,
        fixed_manifest_entry.get("path"),
        name="fixed-C4 phase manifest",
    )
    fixed.verify_contract(
        repo_root=repo_root,
        contract_path=bound_fixed,
        expected_sha256=str(fixed_binding["sha256"]),
        phase_manifest=fixed_manifest,
    )

    files = value.get("files")
    _require(isinstance(files, list) and bool(files),
             "adaptive implementation file bindings are missing")
    observed = set()
    for index, entry in enumerate(files):
        _require(
            isinstance(entry, dict) and set(entry) == {"path", "sha256"},
            f"adaptive implementation file[{index}] binding differs",
        )
        raw_path = entry["path"]
        _require(type(raw_path) is str and raw_path not in observed,
                 "adaptive implementation paths are invalid or duplicated")
        path = fixed._relative_file(
            repo_root, raw_path,
            name=f"adaptive implementation file[{index}]",
        )
        _require(
            _sha256(path) == fixed._canonical_sha(
                entry["sha256"],
                name=f"adaptive implementation file[{index}] SHA-256",
            ),
            f"adaptive implementation file drifted: {raw_path}",
        )
        observed.add(raw_path)
    _require(REQUIRED_FILES.issubset(observed),
             "adaptive implementation contract omits a required file")
    _require(
        value.get("git_heads") == {
            "repository": fixed._git_head(repo_root),
            "third_party_lmcache": fixed._git_head(
                repo_root / "third_party/lmcache"),
        },
        "adaptive implementation Git provenance differs",
    )
    _require(value.get("environment_versions") == fixed._environment_versions(),
             "adaptive implementation environment versions differ")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--fixed-c4-contract", type=Path, required=True)
    args = parser.parse_args()
    value = verify_contract(
        repo_root=args.repo_root,
        contract_path=args.contract,
        expected_sha256=args.expected_sha256,
        fixed_c4_contract=args.fixed_c4_contract,
    )
    print(json.dumps({
        "schema": SCHEMA,
        "fingerprint_sha256": value["fingerprint_sha256"],
        "files": len(value["files"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
