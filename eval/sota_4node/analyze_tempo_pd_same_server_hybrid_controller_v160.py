#!/usr/bin/env python3
"""Production analysis accepting the explicit remote-first warmup order."""

from __future__ import annotations
import json
from eval.sota_4node import analyze_tempo_pd_same_server_warm_reuse_v132 as warm
from eval.sota_4node import analyze_tempo_pd_same_server_hybrid_controller_v151 as final


def _validate(stage_root):
    order = ("lmcache_remote", "fixed_local", "tempo")
    warm_maps = {}
    for sequence_index, arm in enumerate(order):
        path = (stage_root / "same_server_balanced_warm" /
                f"{sequence_index:02d}_{arm}_r0.raw.json")
        value = json.loads(path.read_text(encoding="utf-8"))
        contract = warm._validate_contract(value, arm, "warm")
        warm_maps[arm] = warm._prompt_hashes(value, str(contract["request_prefix"]))
    if set(warm._ACTUAL_CONTRACTS) != set(range(6)):
        raise ValueError("exact six measured contracts required")
    warm_hashes = {arm: sorted(values.values()) for arm, values in warm_maps.items()}
    for arm in set(warm._ORDER):
        measured = warm._MEASURED_PROMPTS[arm]
        if len(measured) != 2 or any(sorted(values.values()) != warm_hashes[arm]
                                     for values in measured):
            raise ValueError("prompt keys were not reused within arm")
    lists = list(warm_hashes.values())
    if any(lists[left] == lists[right] for left in range(len(lists))
           for right in range(left + 1, len(lists))):
        raise ValueError("prompt keys were not isolated across arms")
    return True


def main() -> int:
    original = warm._validate_warm_and_measured
    warm._validate_warm_and_measured = _validate
    try:
        return final.main()
    finally:
        warm._validate_warm_and_measured = original


if __name__ == "__main__": raise SystemExit(main())
