#!/usr/bin/env python3
"""Bind the C7 goodput-aware mesh lease profile to a fresh source contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    base = args.base.resolve()
    profile = args.profile.resolve()
    output = args.output.resolve()
    _require(base.is_file(), "base C7 contract is missing")
    _require(profile.is_file(), "goodput profile is missing")
    _require(not output.exists(), "refusing to overwrite C7 contract")
    _require(repo_root in profile.parents, "profile must be in repository")

    raw = json.loads(base.read_text(encoding="utf-8"))
    _require(raw.get("schema") == "tempo-go-c7-joint-control-contract-v1",
             "base contract schema differs")
    section = raw["joint_control"]
    profile_rel = profile.relative_to(repo_root).as_posix()
    section["global_profile"] = {
        "path": profile_rel,
        "sha256": _sha256(profile),
        "fingerprint_sha256": json.loads(
            profile.read_text(encoding="utf-8"))["fingerprint_sha256"],
    }

    inventory = dict(raw["source_inventory"])
    for relative in tuple(inventory) + (
        "tempo/pd_global_orchestrator.py",
        "tempo/pd_global_profile.py",
    ):
        source = repo_root / relative
        _require(source.is_file(), f"source inventory file is missing: {relative}")
        inventory[relative] = _sha256(source)
    raw["source_inventory"] = dict(sorted(inventory.items()))
    raw["candidate"] = {
        "id": "c7-goodput-aware-mesh-lease-v1",
        "base_contract": str(base.relative_to(repo_root).as_posix()),
        "purpose": (
            "offered-goodput admission with measured mesh completion credit "
            "and bounded native endpoint queue debt"),
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print("profile_sha256", section["global_profile"]["sha256"])
    print("profile_fingerprint", section["global_profile"]["fingerprint_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
