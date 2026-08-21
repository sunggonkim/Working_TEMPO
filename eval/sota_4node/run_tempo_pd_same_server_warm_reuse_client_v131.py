#!/usr/bin/env python3
"""Order-balanced warm-reuse client with keys stable within each route arm."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as base


_ORIGINAL_DERIVE = base._derive
_STABLE_OFFSET = {"local": 100, "tempo": 200, "remote": 300}
_CONTRACT_TO_KEY = {
    "fixed_local": "local",
    "tempo": "tempo",
    "lmcache_remote": "remote",
}


def _arm_from_prefix(prefix: str) -> str:
    for arm in _STABLE_OFFSET:
        if prefix.startswith(f"ssb-{arm}-"):
            return arm
    raise ValueError(f"unknown balanced prefix: {prefix}")


def _derive(rows: list[dict], *, prefix: str, offset: int) -> list[dict]:
    del offset
    return _ORIGINAL_DERIVE(
        rows, prefix=prefix, offset=_STABLE_OFFSET[_arm_from_prefix(prefix)]
    )


def _patch_contract(value: dict) -> None:
    contract = value.get("same_server_balanced_contract")
    if not isinstance(contract, dict):
        return
    arm = contract.get("arm")
    if arm not in _CONTRACT_TO_KEY:
        raise ValueError("warm-reuse arm mismatch")
    arm_key = _CONTRACT_TO_KEY[arm]
    contract.update({
        "nonce_offset": _STABLE_OFFSET[arm_key],
        "cache_keys_disjoint_across_all_blocks": False,
        "cache_keys_reused_within_arm": True,
        "cache_keys_disjoint_across_arms": True,
        "cache_reuse_contract": "same-prompt-warm-and-measured-within-arm-v131",
    })


def _patch_artifacts(output: Path) -> None:
    public = json.loads(output.read_text(encoding="utf-8"))
    orchestration = public.get("same_server_balanced_orchestration")
    if not isinstance(orchestration, dict):
        raise ValueError("balanced orchestration missing")
    artifacts = orchestration.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("balanced artifacts missing")
    for artifact_path in artifacts.values():
        path = Path(artifact_path)
        value = json.loads(path.read_text(encoding="utf-8"))
        _patch_contract(value)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    _patch_contract(public)
    orchestration.update({
        "cache_keys_disjoint_across_all_blocks": False,
        "cache_keys_reused_within_arm": True,
        "cache_keys_disjoint_across_arms": True,
        "cache_reuse_contract": "same-prompt-warm-and-measured-within-arm-v131",
    })
    output.write_text(json.dumps(public, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")


def _argument(name: str) -> Path:
    return Path(sys.argv[sys.argv.index(name) + 1]).resolve()


def main() -> int:
    original = base._derive
    base._derive = _derive
    try:
        status = base.main()
    finally:
        base._derive = original
    _patch_artifacts(_argument("--output"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
