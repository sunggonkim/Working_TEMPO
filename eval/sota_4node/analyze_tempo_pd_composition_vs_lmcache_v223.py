#!/usr/bin/env python3
"""Anchored comparison of the composition-aware TEMPO epoch and LMCache.

The two candidates were measured in separate server lifecycles in the same
allocation.  A fixed-local arm in each lifecycle is therefore used as a drift
anchor.  The result is stronger than an unanchored cross-run comparison, but
it is deliberately not labelled a simultaneous/head-to-head measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


SCHEMA = "tempo-pd-composition-vs-lmcache-analysis-223"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metric(arm: dict[str, Any], name: str) -> float:
    perf = arm["performance"]
    if name == "throughput":
        return float(perf["request_throughput_per_s"])
    if name == "e2e_p99":
        return float(perf["e2e_ms"]["p99"])
    if name == "tpot_p99":
        return float(perf["tpot_ms"]["p99"])
    raise AssertionError(name)


def _fingerprint(arm: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    rows = arm["request_metrics"]
    _require(isinstance(rows, list) and len(rows) == 48, "every arm must contain 48 requests")
    ids = [str(row["request_id"]) for row in rows]
    fps = [str(row["workload_fingerprint"]) for row in rows]
    _require(len(ids) == len(set(ids)), "request ids must be unique")
    return (
        hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        hashlib.sha256("\n".join(fps).encode()).hexdigest(),
        rows,
    )


def _percent_gain(higher: float, lower: float) -> float:
    _require(lower > 0.0, "metric denominator must be positive")
    return (higher / lower - 1.0) * 100.0


def _percent_reduction(old: float, new: float) -> float:
    _require(old > 0.0, "metric denominator must be positive")
    return (1.0 - new / old) * 100.0


def _all_true(values: Iterable[bool]) -> bool:
    return all(bool(value) for value in values)


def analyze(
    old_path: Path,
    new_path: Path,
    *,
    allocation_id: str,
) -> dict[str, Any]:
    old = _load(old_path)
    new = _load(new_path)
    _require(old.get("schema") == "tempo-pd-production-hybrid-controller-analysis-151", "old schema changed")
    _require(new.get("schema") == "tempo-pd-hybrid-saturation-analysis-192", "new schema changed")
    _require(allocation_id and allocation_id in str(old_path) and allocation_id in str(new_path), "paths must bind the same allocation id")

    lmcache = old["lmcache_remote"]
    old_local = old["fixed_local"]
    tempo = new["tempo"]
    new_local = new["fixed_local_primary"]
    arms = {"old_lmcache": lmcache, "old_local": old_local, "new_tempo": tempo, "new_local": new_local}

    model_hashes = {str(arm["model_config_sha256"]) for arm in arms.values()}
    _require(len(model_hashes) == 1, "model config hashes differ")
    fingerprints = {name: _fingerprint(arm) for name, arm in arms.items()}
    request_id_hashes = {value[0] for value in fingerprints.values()}
    workload_hashes = {value[1] for value in fingerprints.values()}
    _require(len(request_id_hashes) == 1, "request id sequences differ")
    _require(len(workload_hashes) == 1, "workload fingerprint sequences differ")

    lm_rows = {str(row["request_id"]): row for row in fingerprints["old_lmcache"][2]}
    tempo_rows = {str(row["request_id"]): row for row in fingerprints["new_tempo"][2]}
    paired_deltas = [float(tempo_rows[key]["e2e_ms"]) - float(lm_rows[key]["e2e_ms"]) for key in lm_rows]

    old_values = {name: _metric(lmcache, name) for name in ("throughput", "e2e_p99", "tpot_p99")}
    old_anchor = {name: _metric(old_local, name) for name in old_values}
    new_values = {name: _metric(tempo, name) for name in old_values}
    new_anchor = {name: _metric(new_local, name) for name in old_values}

    direct = {
        "throughput_gain_percent": _percent_gain(new_values["throughput"], old_values["throughput"]),
        "e2e_p99_reduction_percent": _percent_reduction(old_values["e2e_p99"], new_values["e2e_p99"]),
        "tpot_p99_reduction_percent": _percent_reduction(old_values["tpot_p99"], new_values["tpot_p99"]),
        "paired_e2e_win_count": sum(delta < 0.0 for delta in paired_deltas),
        "paired_e2e_delta_median_ms": median(paired_deltas),
    }
    anchor_drift = {
        "throughput_percent": _percent_gain(new_anchor["throughput"], old_anchor["throughput"]),
        "e2e_p99_percent": _percent_gain(new_anchor["e2e_p99"], old_anchor["e2e_p99"]),
        "tpot_p99_percent": _percent_gain(new_anchor["tpot_p99"], old_anchor["tpot_p99"]),
    }
    anchored = {
        "throughput_gain_percent": _percent_gain(
            new_values["throughput"] / new_anchor["throughput"],
            old_values["throughput"] / old_anchor["throughput"],
        ),
        "e2e_p99_reduction_percent": _percent_reduction(
            old_values["e2e_p99"] / old_anchor["e2e_p99"],
            new_values["e2e_p99"] / new_anchor["e2e_p99"],
        ),
        "tpot_p99_reduction_percent": _percent_reduction(
            old_values["tpot_p99"] / old_anchor["tpot_p99"],
            new_values["tpot_p99"] / new_anchor["tpot_p99"],
        ),
    }
    gates = {
        "identical_model_and_workload": True,
        "local_anchor_drift_bounded": (
            abs(anchor_drift["throughput_percent"]) <= 3.0
            and abs(anchor_drift["e2e_p99_percent"]) <= 5.0
            and abs(anchor_drift["tpot_p99_percent"]) <= 10.0
        ),
        "direct_throughput_beats_lmcache": direct["throughput_gain_percent"] > 0.0,
        "direct_e2e_p99_beats_lmcache": direct["e2e_p99_reduction_percent"] > 0.0,
        "direct_tpot_p99_beats_lmcache": direct["tpot_p99_reduction_percent"] > 0.0,
        "paired_majority_beats_lmcache": direct["paired_e2e_win_count"] >= 25 and direct["paired_e2e_delta_median_ms"] < 0.0,
        "anchored_throughput_beats_lmcache": anchored["throughput_gain_percent"] > 0.0,
        "anchored_e2e_p99_beats_lmcache": anchored["e2e_p99_reduction_percent"] > 0.0,
        "anchored_tpot_p99_beats_lmcache": anchored["tpot_p99_reduction_percent"] > 0.0,
    }
    passes = _all_true(gates.values())
    return {
        "schema": SCHEMA,
        "allocation_id": allocation_id,
        "verdict": "anchored_composition_advantage_over_lmcache" if passes else "anchored_comparison_needs_revision",
        "passes": passes,
        "claim_boundary": (
            "Same allocation, identical model and request/workload fingerprints, but separate server lifecycles. "
            "Fixed-local arms anchor lifecycle drift. This is not a simultaneous LMCache head-to-head and is not a Mooncake comparison."
        ),
        "provenance": {
            "old_path": str(old_path),
            "new_path": str(new_path),
            "model_config_sha256": next(iter(model_hashes)),
            "request_id_sequence_sha256": next(iter(request_id_hashes)),
            "workload_fingerprint_sequence_sha256": next(iter(workload_hashes)),
            "request_count_per_arm": 48,
        },
        "metrics": {
            "old_lmcache": old_values,
            "old_fixed_local": old_anchor,
            "new_tempo": new_values,
            "new_fixed_local": new_anchor,
        },
        "direct": direct,
        "local_anchor_drift": anchor_drift,
        "anchored": anchored,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.old, args.new, allocation_id=args.allocation_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": report["verdict"], "passes": report["passes"], "direct": report["direct"], "anchored": report["anchored"]}, sort_keys=True))


if __name__ == "__main__":
    main()
