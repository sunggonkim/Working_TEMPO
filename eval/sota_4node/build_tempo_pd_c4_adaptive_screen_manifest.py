#!/usr/bin/env python3
"""Build the post-C4 adaptive-screen workload contract without fitting data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as analyzer
from eval.sota_4node import build_tempo_pd_c4_calibrated_profiles as profiles
from eval.sota_4node import build_tempo_pd_c4_phase_manifest as c4_manifest
from tempo.pd_contention_workload import VALIDATION_FOREGROUND_GEOMETRIES


SCHEMA = profiles.LIVE_MANIFEST_SCHEMA
ARM_ORDER_BY_REPLICATE = (
    ("local", "remote", "predictor", "tempo"),
    ("tempo", "predictor", "remote", "local"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be lowercase SHA-256",
    )
    return value


def manifest_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    analysis_path: Path, *, expected_analysis_sha256: str,
) -> dict[str, object]:
    analysis_path = analysis_path.resolve()
    expected_analysis_sha256 = _canonical_sha(
        expected_analysis_sha256, name="C4 analysis SHA-256")
    _require(analysis_path.is_file(), "C4 analysis is missing")
    _require(_sha256(analysis_path) == expected_analysis_sha256,
             "C4 analysis digest differs")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    _require(
        isinstance(analysis, dict)
        and analysis.get("schema") == analyzer.SCHEMA
        and analysis.get("fingerprint_sha256")
        == analyzer._analysis_fingerprint(analysis)
        and analysis.get("authorizes_profile_fit") is True
        and analysis.get("authorizes_controller_parameter_search") is False
        and analysis.get("performance_claim_allowed") is False,
        "C4 analysis does not authorize the adaptive screen",
    )
    source_binding = analysis.get("phase_manifest")
    _require(isinstance(source_binding, dict)
             and set(source_binding) == {"path", "sha256", "fingerprint_sha256"},
             "C4 source phase-manifest binding differs")
    source_path = Path(str(source_binding["path"])).resolve()
    _require(source_path.is_file()
             and _sha256(source_path) == source_binding["sha256"],
             "C4 source phase manifest digest differs")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _require(
        source.get("schema") == c4_manifest.SCHEMA
        and source.get("fingerprint_sha256")
        == c4_manifest.manifest_fingerprint(source)
        and source.get("performance_claim_allowed") is False,
        "C4 source phase manifest is invalid",
    )
    value: dict[str, object] = {
        "schema": SCHEMA,
        "purpose": "calibration-only adaptive endpoint-feedback screen",
        "calibration_analysis": {
            "path": str(analysis_path),
            "sha256": expected_analysis_sha256,
            "fingerprint_sha256": analysis["fingerprint_sha256"],
        },
        "source_fixed_phase_manifest": {
            "path": str(source_path),
            "sha256": source_binding["sha256"],
            "fingerprint_sha256": source_binding["fingerprint_sha256"],
        },
        "source_workload": source["source_workload"],
        "phase_order": source["phase_order"],
        "phase_duration_ms": source["phase_duration_ms"],
        "foreground_rate_per_s": source["foreground_rate_per_s"],
        "background_rates_per_s": source["background_rates_per_s"],
        "foreground_geometries": [
            {
                "prompt_tokens": row.prompt_tokens,
                "output_tokens": row.output_tokens,
                "cache_state": row.cache_state.value,
            }
            for row in VALIDATION_FOREGROUND_GEOMETRIES
        ],
        "cache_state_protocol": source["cache_state_protocol"],
        "endpoint_evidence_contract": source["endpoint_evidence_contract"],
        "profile_fit_formula": profiles.FORMULA_ID,
        "arm_order_by_replicate": [
            list(order) for order in ARM_ORDER_BY_REPLICATE
        ],
        "cooldown_s": 2.0,
        "measurement": {
            "e2e_slo_ms": profiles.DEFAULT_E2E_SLO_MS,
            "ttft_slo_ms": profiles.TTFT_SLO_MS,
            "tpot_slo_ms": profiles.TPOT_SLO_MS,
            "policy_arms": [
                "fixed_local", "fixed_remote", "simple_predictor",
                "tempo_endpoint_feedback",
            ],
            "strongest_fixed_selected_only_from_independent_validation": True,
            "paired_semantic_requests_required": True,
            "counterfactual_local_and_remote_required": True,
            "repetitions": 2,
            "max_workers": 128,
        },
        "slurm": {
            "nodes": 4,
            "gpus": 16,
            "interactive_time_limit": "04:00:00",
            "persistent_allocation_reuse_required": True,
            "login_node_experiment_execution_allowed": False,
        },
        "transport": "LMCacheConnectorV1:UCX",
        "unchanged_pd_data_plane": True,
        "controller_tuning_allowed": False,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
    }
    value["fingerprint_sha256"] = manifest_fingerprint(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--expected-analysis-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite adaptive manifest")
    value = build_manifest(
        args.analysis,
        expected_analysis_sha256=args.expected_analysis_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "fingerprint_sha256": value["fingerprint_sha256"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
