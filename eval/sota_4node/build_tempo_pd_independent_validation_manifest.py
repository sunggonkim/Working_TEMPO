#!/usr/bin/env python3
"""Build the held-out TEMPO validation manifest from frozen preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_adaptive_screen as adaptive_analysis
from eval.sota_4node import analyze_tempo_pd_c4_semantic_integration_screen as semantic_analysis
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as adaptive_manifest
from eval.sota_4node import build_tempo_pd_c4_phase_manifest as c4_manifest
from tempo.pd_contention_workload import VALIDATION_FOREGROUND_GEOMETRIES


SCHEMA = "tempo-pd-independent-validation-manifest-v2"
PREREGISTRATION_SCHEMA = "tempo-pd-independent-validation-preregistration-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREREGISTRATION = (
    Path(__file__).resolve().parent
    / "tempo_pd_independent_validation_preregistration_v1.json"
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


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    _require(path.is_file(), f"{name} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def manifest_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding(path: Path, *, fingerprint: str | None = None) -> dict[str, str]:
    value = {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
    if fingerprint is not None:
        value["fingerprint_sha256"] = fingerprint
    return value


def _bound_path(entry: object, *, name: str) -> Path:
    _require(isinstance(entry, Mapping), f"{name} binding is missing")
    raw = entry.get("path")
    expected = _canonical_sha(entry.get("sha256"), name=f"{name} SHA-256")
    _require(type(raw) is str and raw, f"{name} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    _require(path.is_file() and _sha256(path) == expected,
             f"{name} digest differs")
    return path


def _load_preregistration(
    path: Path, *, expected_sha256: str,
) -> dict[str, object]:
    path = path.resolve()
    expected_sha256 = _canonical_sha(
        expected_sha256, name="independent preregistration SHA-256")
    _require(path.is_file() and _sha256(path) == expected_sha256,
             "independent preregistration digest differs")
    value = _load_object(path, name="independent preregistration")
    _require(
        value.get("schema") == PREREGISTRATION_SCHEMA
        and value.get("post_validation_tuning_allowed") is False
        and value.get("performance_claim_allowed_before_all_gates_pass") is False
        and value.get("physical_switch_bottleneck_claim_allowed") is False,
        "independent preregistration claim contract differs",
    )
    workload = value.get("workload")
    measurement = value.get("measurement")
    gates = value.get("success_gates")
    execution = value.get("execution")
    _require(
        isinstance(workload, dict)
        and workload.get("phase_order")
        == [phase.value for phase in c4_manifest.PHASES]
        and workload.get("traffic_shape") == "burst"
        and workload.get("burst_epoch_ms") == 1000
        and workload.get("burst_active_fraction") == 0.25
        and workload.get("phase_duration_ms") == 12000
        and workload.get("foreground_rate_per_s") == 2.0
        and workload.get("replicate_ids") == [2, 3, 4, 5]
        and workload.get("paired_foreground_samples_per_group") == 16
        and workload.get("workload_group_count") == 36,
        "independent held-out workload contract differs",
    )
    expected_geometries = [
        {
            "prompt_tokens": row.prompt_tokens,
            "output_tokens": row.output_tokens,
            "cache_state": row.cache_state.value,
        }
        for row in VALIDATION_FOREGROUND_GEOMETRIES
    ]
    _require(workload.get("foreground_geometries") == expected_geometries,
             "independent geometry/cache inventory differs")
    expected_orders = {
        "2": ["local", "predictor", "tempo", "remote"],
        "3": ["remote", "tempo", "predictor", "local"],
        "4": ["predictor", "local", "remote", "tempo"],
        "5": ["tempo", "remote", "local", "predictor"],
    }
    _require(workload.get("arm_order_by_replicate") == expected_orders,
             "independent arm order differs")
    _require(
        isinstance(measurement, dict)
        and measurement.get("pooled_e2e_statistic") == "median"
        and measurement.get("strongest_fixed_selection")
        == "single_lower_pooled_median_e2e_arm_local_tie_break"
        and measurement.get("request_goodput_definition")
        == (
            "paired_valid_foreground_requests_per_second_from_first_dispatch_"
            "to_last_stream_end"
        )
        and measurement.get("minimum_route_counterfactual_samples_per_route")
        == 12,
        "independent measurement definition differs",
    )
    _require(
        isinstance(gates, dict)
        and gates.get("minimum_pooled_median_e2e_gain_vs_strongest_fixed")
        == 0.10
        and gates.get("minimum_pooled_median_e2e_gain_vs_predictor") == 0.05
        and gates.get("minimum_request_goodput_gain_vs_strongest_fixed") == 0.05
        and gates.get("minimum_overall_paired_win_fraction_vs_strongest_fixed")
        == 0.75
        and gates.get("minimum_group_paired_win_fraction_vs_group_strongest_fixed")
        == 0.60
        and gates.get("maximum_group_e2e_p99_regression_vs_group_strongest_fixed")
        == 0.05
        and gates.get("maximum_group_tpot_p99_regression_vs_group_strongest_fixed")
        == 0.05
        and gates.get("maximum_worst_paired_e2e_regression_ms_vs_strongest_fixed")
        == 100.0
        and gates.get("minimum_median_local_selection_gain_vs_remote_counterfactual")
        == 0.05
        and gates.get("minimum_median_remote_selection_gain_vs_local_counterfactual")
        == 0.05,
        "independent final success thresholds differ",
    )
    _require(
        isinstance(execution, dict)
        and execution.get("nodes") == 4
        and execution.get("gpus") == 16
        and execution.get("interactive_time_limit") == "04:00:00"
        and execution.get("one_persistent_allocation_for_entire_validation") is True
        and execution.get("must_use_different_slurm_job_from_calibration") is True
        and execution.get("login_node_gpu_or_inference_execution_allowed") is False,
        "independent Slurm scope differs",
    )
    return value


def _load_authorized_analysis(
    path: Path, *, expected_sha256: str,
) -> tuple[
    dict[str, object], Path, dict[str, object], dict[str, object]
]:
    path = path.resolve()
    expected_sha256 = _canonical_sha(
        expected_sha256, name="adaptive analysis SHA-256")
    _require(path.is_file() and _sha256(path) == expected_sha256,
             "adaptive analysis digest differs")
    value = _load_object(path, name="candidate screen analysis")
    schema = value.get("schema")
    if schema == adaptive_analysis.SCHEMA:
        analyzer = adaptive_analysis
        candidate = {
            "kind": "candidate_a_instant_score_v1",
            "endpoint_routing_policy": "instant_score_v1",
            "passive_external_credit": False,
            "implementation_entry": "adaptive_implementation_contract",
        }
    elif schema == semantic_analysis.SCHEMA:
        analyzer = semantic_analysis
        candidate = {
            "kind": "candidate_b_semantic_epoch_v1",
            "endpoint_routing_policy": "semantic_epoch_v1",
            "passive_external_credit": True,
            "implementation_entry": (
                "semantic_integration_implementation_contract"),
        }
    else:
        raise ValueError("candidate analysis schema is unsupported")
    _require(
        value.get("fingerprint_sha256") == analyzer.analysis_fingerprint(value)
        and value.get("authorizes_independent_validation") is True
        and value.get("strongest_fixed_selection_authoritative") is False
        and value.get("calibration_only") is True
        and value.get("performance_claim_allowed") is False,
        "candidate analysis does not authorize independent validation",
    )
    source = value.get("source_node_result")
    _require(isinstance(source, Mapping),
             "candidate analysis source result binding is missing")
    source_path = _bound_path(source, name="candidate node result")
    _require(
        analyzer.analyze(
            source_path, expected_result_sha256=str(source["sha256"])) == value,
        "candidate analysis does not reproduce from raw evidence",
    )
    run_contract_path = _bound_path(
        value.get("run_contract"), name="candidate run contract")
    run_contract = analyzer._validate_run_contract(run_contract_path)
    _require(
        value["run_contract"].get("fingerprint_sha256")
        == run_contract.get("fingerprint_sha256"),
        "candidate analysis/run-contract fingerprint differs",
    )
    _require(
        run_contract.get("endpoint_routing_policy", "instant_score_v1")
        == candidate["endpoint_routing_policy"]
        and run_contract.get("passive_external_credit", False)
        is candidate["passive_external_credit"],
        "candidate analysis/run-contract routing policy differs",
    )
    return value, run_contract_path, run_contract, candidate


def build_manifest(
    *, adaptive_analysis_path: Path, adaptive_analysis_sha256: str,
    preregistration_path: Path, preregistration_sha256: str,
) -> dict[str, object]:
    preregistration_path = preregistration_path.resolve()
    preregistration = _load_preregistration(
        preregistration_path, expected_sha256=preregistration_sha256)
    analysis, adaptive_contract_path, adaptive_contract, candidate = (
        _load_authorized_analysis(
            adaptive_analysis_path,
            expected_sha256=adaptive_analysis_sha256,
        )
    )
    adaptive_manifest_path = _bound_path(
        adaptive_contract.get("phase_manifest"), name="adaptive manifest")
    source_manifest = _load_object(
        adaptive_manifest_path, name="adaptive manifest")
    _require(
        source_manifest.get("schema") == adaptive_manifest.SCHEMA
        and source_manifest.get("fingerprint_sha256")
        == adaptive_manifest.manifest_fingerprint(source_manifest)
        and source_manifest.get("controller_tuning_allowed") is False
        and source_manifest.get("performance_claim_allowed") is False,
        "adaptive source manifest is invalid",
    )
    source_workload_path = _bound_path(
        adaptive_contract.get("source_workload"), name="source workload")
    workload = preregistration["workload"]
    value: dict[str, object] = {
        "schema": SCHEMA,
        "purpose": "held-out frozen independent validation",
        "candidate": candidate,
        "preregistration": _binding(preregistration_path),
        "candidate_screen_analysis": _binding(
            adaptive_analysis_path,
            fingerprint=str(analysis["fingerprint_sha256"]),
        ),
        "adaptive_screen_analysis": _binding(
            adaptive_analysis_path,
            fingerprint=str(analysis["fingerprint_sha256"]),
        ),
        "candidate_run_contract": _binding(
            adaptive_contract_path,
            fingerprint=str(adaptive_contract["fingerprint_sha256"]),
        ),
        "adaptive_run_contract": _binding(
            adaptive_contract_path,
            fingerprint=str(adaptive_contract["fingerprint_sha256"]),
        ),
        "source_candidate_manifest": _binding(
            adaptive_manifest_path,
            fingerprint=str(source_manifest["fingerprint_sha256"]),
        ),
        "source_adaptive_manifest": _binding(
            adaptive_manifest_path,
            fingerprint=str(source_manifest["fingerprint_sha256"]),
        ),
        "source_workload": _binding(source_workload_path),
        "phase_order": list(workload["phase_order"]),
        "traffic_shape": workload["traffic_shape"],
        "burst_epoch_ms": workload["burst_epoch_ms"],
        "burst_active_fraction": workload["burst_active_fraction"],
        "phase_duration_ms": workload["phase_duration_ms"],
        "foreground_rate_per_s": workload["foreground_rate_per_s"],
        "background_rates_per_s": source_manifest["background_rates_per_s"],
        "foreground_geometries": workload["foreground_geometries"],
        "cache_state_protocol": source_manifest["cache_state_protocol"],
        "endpoint_evidence_contract": source_manifest[
            "endpoint_evidence_contract"],
        "replicate_ids": workload["replicate_ids"],
        "arm_order_by_replicate": [
            {
                "replicate": replicate,
                "arms": workload["arm_order_by_replicate"][str(replicate)],
            }
            for replicate in workload["replicate_ids"]
        ],
        "cooldown_s": workload["cooldown_s"],
        "workload_groups": workload["workload_groups"],
        "workload_group_count": workload["workload_group_count"],
        "paired_foreground_samples_per_group": workload[
            "paired_foreground_samples_per_group"],
        "measurement": preregistration["measurement"],
        "success_gates": preregistration["success_gates"],
        "correctness": preregistration["correctness"],
        "profile_promotion": preregistration["profile_promotion"],
        "slurm": preregistration["execution"],
        "transport": "LMCacheConnectorV1:UCX",
        "unchanged_pd_data_plane": True,
        "controller_parameter_search_allowed": False,
        "post_validation_tuning_allowed": False,
        "calibration_only": False,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "claim_boundary": preregistration["claim_boundary"],
    }
    value["fingerprint_sha256"] = manifest_fingerprint(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-analysis", "--adaptive-analysis",
        dest="adaptive_analysis", type=Path, required=True)
    parser.add_argument(
        "--candidate-analysis-sha256", "--adaptive-analysis-sha256",
        dest="adaptive_analysis_sha256", required=True)
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(),
             "refusing to overwrite independent manifest")
    value = build_manifest(
        adaptive_analysis_path=args.adaptive_analysis,
        adaptive_analysis_sha256=args.adaptive_analysis_sha256,
        preregistration_path=args.preregistration,
        preregistration_sha256=args.preregistration_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "fingerprint_sha256": value["fingerprint_sha256"],
        "sha256": _sha256(args.output),
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
