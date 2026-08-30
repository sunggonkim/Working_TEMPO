#!/usr/bin/env python3
"""Fail-closed gate for the frozen coupled-C3 ABBA characterization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


SCHEMA = "tempo-pd-c3-coupled-abba-gate-v1"
RESULT_SCHEMA = "tempo-pd-kv-only-attribution-node-v1"
CHARACTERIZATION_SCHEMA = "tempo-pd-kv-only-characterization-v3"
MANIFEST_SCHEMA = "tempo-pd-c3-coupled-abba-manifest-v2"
RATES = (0.0, 4.0, 8.0, 12.0)
REPETITIONS = 2
MINIMUM_GAIN = 0.05


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str) -> dict[str, object]:
    _require(path.is_file(), f"{name} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} is not an object")
    return value


def evaluate(
    *, result_path: Path, characterization_path: Path, manifest_path: Path,
) -> dict[str, object]:
    result_path = result_path.resolve()
    characterization_path = characterization_path.resolve()
    manifest_path = manifest_path.resolve()
    result = _load(result_path, "C3 ABBA result")
    characterization = _load(
        characterization_path, "C3 ABBA characterization")
    manifest = _load(manifest_path, "C3 ABBA manifest")

    _require(result.get("schema") == RESULT_SCHEMA,
             "C3 ABBA result schema mismatch")
    _require(characterization.get("schema") == CHARACTERIZATION_SCHEMA,
             "C3 ABBA characterization schema mismatch")
    _require(manifest.get("schema") == MANIFEST_SCHEMA,
             "C3 ABBA manifest schema mismatch")
    _require(manifest.get("performance_claim_allowed") is False,
             "C3 ABBA manifest cannot permit a performance claim")
    _require(result.get("performance_claim_allowed") is False,
             "C3 ABBA result cannot permit a performance claim")
    _require(result.get("physical_switch_bottleneck_claim_allowed") is False,
             "C3 ABBA result cannot permit a switch-bottleneck claim")

    manifest_sha256 = _sha256(manifest_path)
    _require(result.get("coupled_manifest_sha256") == manifest_sha256,
             "C3 ABBA result/manifest digest mismatch")
    _require(Path(str(result.get("coupled_manifest"))).resolve() == manifest_path,
             "C3 ABBA result/manifest path mismatch")
    raw_path = Path(str(result.get("raw"))).resolve()
    _require(raw_path.is_file(), "C3 ABBA aggregate raw artifact is missing")
    _require(Path(str(characterization.get("source"))).resolve() == raw_path,
             "C3 ABBA characterization source mismatch")

    _require(manifest.get("replicates") == REPETITIONS,
             "C3 ABBA manifest repetition count differs")
    _require(manifest.get("arm_order_policy") == "paired_abba",
             "C3 ABBA manifest order policy differs")
    _require(manifest.get("within_rate_block_order") == [
        "local", "remote", "remote", "local"],
        "C3 ABBA manifest block order differs")
    _require(tuple(float(value) for value in manifest.get(
        "p_only_rates_per_s", [])) == RATES,
        "C3 ABBA manifest rate ladder differs")
    _require(float(manifest.get("decoder_hot_rate_per_s")) == 22.4,
             "C3 ABBA decoder-hot rate differs")

    _require(result.get("block_count") == 2 * len(RATES) * REPETITIONS,
             "C3 ABBA result block count differs")
    _require(result.get("stopped_after_first_invalid_block") is None,
             "C3 ABBA stopped after an invalid block")
    _require(result.get("repetitions_per_rate") == REPETITIONS,
             "C3 ABBA result repetition count differs")
    _require(result.get("arm_order_policy") == "paired_abba",
             "C3 ABBA result order policy differs")
    _require(result.get("paired_semantic_schedules_exact") is True,
             "C3 ABBA semantic schedules differ")

    _require(characterization.get("repetitions_per_rate") == REPETITIONS,
             "C3 ABBA characterization repetition count differs")
    _require(characterization.get("arm_order_policy") == "paired_abba",
             "C3 ABBA characterization order policy differs")
    _require(characterization.get("all_measured_requests_valid") is True,
             "C3 ABBA has an invalid measured request")
    proof = characterization.get("p_only_source_compute_attribution")
    _require(isinstance(proof, dict)
             and proof.get("long_producer_prefill_removed") is True
             and proof.get("expected_residual_recompute_tokens_per_request") == 1
             and proof.get("zero_producer_compute_claim_allowed") is False,
             "C3 ABBA P_ONLY source-compute proof failed")

    rows = characterization.get("paired_replicate_summary")
    _require(isinstance(rows, list)
             and len(rows) == len(RATES) * REPETITIONS,
             "C3 ABBA paired replicate matrix is incomplete")
    indexed: dict[tuple[float, int], dict[str, object]] = {}
    for row in rows:
        _require(isinstance(row, dict), "C3 ABBA replicate row is malformed")
        key = (float(row["background_rate_per_s"]),
               int(row["replicate_index"]))
        _require(key not in indexed, "C3 ABBA replicate row is duplicated")
        indexed[key] = row
    expected = {(rate, replicate) for rate in RATES
                for replicate in range(REPETITIONS)}
    _require(set(indexed) == expected, "C3 ABBA replicate keys differ")
    for rate in RATES:
        _require(indexed[(rate, 0)].get("measured_arm_order") == "local_remote",
                 f"C3 ABBA rate {rate:g} replicate 0 order differs")
        _require(indexed[(rate, 1)].get("measured_arm_order") == "remote_local",
                 f"C3 ABBA rate {rate:g} replicate 1 order differs")

    remote_control = [indexed[(0.0, replicate)]
                      for replicate in range(REPETITIONS)]
    local_overload = [indexed[(12.0, replicate)]
                      for replicate in range(REPETITIONS)]
    remote_gains = [float(row["remote_gain_over_local"])
                    for row in remote_control]
    local_gains = [float(row["local_gain_over_remote"])
                   for row in local_overload]
    remote_direction = [row.get("winner") == "remote"
                        for row in remote_control]
    local_direction = [row.get("winner") == "local"
                       for row in local_overload]
    remote_gain = statistics.median(remote_gains)
    local_gain = statistics.median(local_gains)
    passed = (
        all(remote_direction)
        and all(local_direction)
        and remote_gain >= MINIMUM_GAIN
        and local_gain >= MINIMUM_GAIN
    )

    return {
        "schema": SCHEMA,
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "characterization": str(characterization_path),
        "characterization_sha256": _sha256(characterization_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "minimum_gain": MINIMUM_GAIN,
        "remote_control_rate_per_s": 0.0,
        "remote_control_replicate_gains": remote_gains,
        "remote_control_replicate_direction_correct": remote_direction,
        "remote_control_median_gain": remote_gain,
        "local_overload_rate_per_s": 12.0,
        "local_overload_replicate_gains": local_gains,
        "local_overload_replicate_direction_correct": local_direction,
        "local_overload_median_gain": local_gain,
        "intermediate_rates_are_descriptive_not_gated": [4.0, 8.0],
        "c3_coupled_characterization_valid": passed,
        "authorizes_c4_phase_trace": passed,
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--characterization", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    _require(not args.output.exists(), "refusing to overwrite C3 ABBA gate")
    value = evaluate(
        result_path=args.result,
        characterization_path=args.characterization,
        manifest_path=args.manifest,
    )
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "pass": value["c3_coupled_characterization_valid"],
    }, sort_keys=True))
    return 0 if value["c3_coupled_characterization_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
