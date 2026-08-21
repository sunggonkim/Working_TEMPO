#!/usr/bin/env python3
"""Warm-reuse client with stable per-item cache identities across phases."""

from __future__ import annotations

from eval.sota_4node import run_tempo_pd_same_server_warm_reuse_client_v131 as warm


_ORIGINAL_DERIVE = warm._derive
_ORIGINAL_PATCH_CONTRACT = warm._patch_contract


def _derive(rows: list[dict], *, prefix: str, offset: int) -> list[dict]:
    derived = _ORIGINAL_DERIVE(rows, prefix=prefix, offset=offset)
    for index, row in enumerate(derived):
        row["request_id"] = f"{prefix}cache-item-{index:02d}"
    return derived


def _patch_contract(value: dict) -> None:
    _ORIGINAL_PATCH_CONTRACT(value)
    contract = value.get("same_server_balanced_contract")
    if not isinstance(contract, dict):
        return
    old_ids = contract.get("base_request_ids")
    old_hashes = contract.get("base_prompt_sha256")
    if not isinstance(old_ids, list) or not isinstance(old_hashes, dict):
        raise ValueError("cache-catalog canonical identity metadata missing")
    if len(old_ids) != len(set(old_ids)) or any(key not in old_hashes for key in old_ids):
        raise ValueError("cache-catalog source identities invalid")
    canonical_ids = [f"cache-item-{index:02d}" for index in range(len(old_ids))]
    contract["base_request_ids"] = canonical_ids
    contract["base_prompt_sha256"] = {
        canonical: old_hashes[old]
        for canonical, old in zip(canonical_ids, old_ids, strict=True)
    }
    contract["cache_catalog_identity"] = "stable-item-index-v136"


def main() -> int:
    original_derive = warm._derive
    original_patch = warm._patch_contract
    warm._derive = _derive
    warm._patch_contract = _patch_contract
    try:
        return warm.main()
    finally:
        warm._derive = original_derive
        warm._patch_contract = original_patch


if __name__ == "__main__":
    raise SystemExit(main())
