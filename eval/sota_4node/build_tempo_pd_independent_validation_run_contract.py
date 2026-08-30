#!/usr/bin/env python3
"""Bind the frozen TEMPO candidate to one held-out validation execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import build_tempo_pd_c4_adaptive_run_contract as adaptive_contract_builder
from eval.sota_4node import build_tempo_pd_c4_semantic_integration_run_contract as semantic_contract_builder
from eval.sota_4node import build_tempo_pd_independent_validation_manifest as manifest_builder
from eval.sota_4node import promote_tempo_pd_profiles_for_independent_validation as promotion
from eval.sota_4node import verify_tempo_pd_independent_validation_implementation as implementation
from eval.sota_4node import verify_tempo_pd_c4_semantic_integration_implementation as semantic_implementation
from tempo.pd_elastic_profile import load_elastic_profile, require_replicated_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile


SCHEMA = "tempo-pd-independent-validation-run-contract-v2"
INDEPENDENT_FIXED_RUNTIME_ENVIRONMENT = dict(
    adaptive_contract_builder.ADAPTIVE_FIXED_RUNTIME_ENVIRONMENT)
INDEPENDENT_FIXED_RUNTIME_ENVIRONMENT.pop("TEMPO_PD_C4_ADAPTIVE_APPROVED")
INDEPENDENT_FIXED_RUNTIME_ENVIRONMENT.update({
    "TEMPO_PD_INDEPENDENT_VALIDATION_APPROVED": "YES",
    "TEMPO_PD_C4_PHASE_DURATION_MS": "12000",
    "TEMPO_ELASTIC_PD_PROFILE_SCOPE": "replicated",
})


def independent_runtime_environment(
    candidate: Mapping[str, object],
) -> dict[str, str]:
    if candidate.get("kind") == "candidate_a_instant_score_v1":
        return dict(INDEPENDENT_FIXED_RUNTIME_ENVIRONMENT)
    _require(
        candidate.get("kind") == "candidate_b_semantic_epoch_v1"
        and candidate.get("endpoint_routing_policy") == "semantic_epoch_v1"
        and candidate.get("passive_external_credit") is True,
        "independent candidate runtime policy is unsupported",
    )
    value = dict(semantic_contract_builder.SEMANTIC_FIXED_RUNTIME_ENVIRONMENT)
    value.pop("TEMPO_PD_C4_SEMANTIC_INTEGRATION_APPROVED")
    value.update({
        "TEMPO_PD_INDEPENDENT_VALIDATION_APPROVED": "YES",
        "TEMPO_PD_C4_PHASE_DURATION_MS": "12000",
        "TEMPO_ELASTIC_PD_PROFILE_SCOPE": "replicated",
    })
    return value


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


def _load_bound(
    path: Path, expected_sha256: str, *, name: str,
) -> tuple[Path, dict[str, object]]:
    path = path.resolve()
    expected_sha256 = manifest_builder._canonical_sha(
        expected_sha256, name=f"{name} SHA-256")
    _require(path.is_file() and _sha256(path) == expected_sha256,
             f"{name} digest differs")
    return path, manifest_builder._load_object(path, name=name)


def _binding(
    path: Path, *, fingerprint_sha256: str | None = None,
) -> dict[str, str]:
    value = {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
    if fingerprint_sha256 is not None:
        value["fingerprint_sha256"] = fingerprint_sha256
    return value


def _entry_path(contract: Mapping[str, object], name: str) -> Path:
    entry = contract.get(name)
    _require(isinstance(entry, Mapping), f"adaptive contract lacks {name}")
    path = Path(str(entry.get("path", ""))).resolve()
    _require(path.is_file() and _sha256(path) == entry.get("sha256"),
             f"adaptive contract {name} digest differs")
    return path


def build_run_contract(
    *, manifest_path: Path, manifest_sha256: str,
    adaptive_analysis_path: Path, adaptive_analysis_sha256: str,
    preregistration_path: Path, preregistration_sha256: str,
    elastic_path: Path, elastic_sha256: str,
    endpoint_path: Path, endpoint_sha256: str,
    promotion_receipt_path: Path, promotion_receipt_sha256: str,
    implementation_path: Path, implementation_sha256: str,
    repo_root: Path,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    manifest_path, manifest = _load_bound(
        manifest_path, manifest_sha256, name="independent manifest")
    rebuilt_manifest = manifest_builder.build_manifest(
        adaptive_analysis_path=adaptive_analysis_path,
        adaptive_analysis_sha256=adaptive_analysis_sha256,
        preregistration_path=preregistration_path,
        preregistration_sha256=preregistration_sha256,
    )
    _require(
        manifest == rebuilt_manifest
        and manifest.get("fingerprint_sha256")
        == manifest_builder.manifest_fingerprint(manifest)
        and manifest.get("post_validation_tuning_allowed") is False
        and manifest.get("performance_claim_allowed") is False,
        "independent manifest does not reproduce",
    )
    adaptive_analysis_path, adaptive_analysis = _load_bound(
        adaptive_analysis_path, adaptive_analysis_sha256,
        name="candidate analysis")
    candidate = manifest.get("candidate")
    _require(isinstance(candidate, Mapping),
             "independent manifest candidate is missing")
    adaptive_run_contract_path = manifest_builder._bound_path(
        manifest.get("candidate_run_contract"), name="candidate run contract")
    candidate_analyzer = (
        manifest_builder.semantic_analysis
        if candidate.get("kind") == "candidate_b_semantic_epoch_v1"
        else manifest_builder.adaptive_analysis)
    adaptive_run_contract = candidate_analyzer._validate_run_contract(
        adaptive_run_contract_path)
    source_elastic_path = _entry_path(
        adaptive_run_contract, "elastic_profile")
    source_endpoint_path = _entry_path(
        adaptive_run_contract, "endpoint_service_profile")
    source_receipt_path = _entry_path(
        adaptive_run_contract, "profile_receipt")

    elastic_path, elastic_raw = _load_bound(
        elastic_path, elastic_sha256, name="promoted Elastic profile")
    endpoint_path, endpoint_raw = _load_bound(
        endpoint_path, endpoint_sha256, name="promoted endpoint profile")
    promotion_receipt_path, promotion_receipt = _load_bound(
        promotion_receipt_path, promotion_receipt_sha256,
        name="profile promotion receipt")
    rebuilt_profiles = promotion.promote_profiles(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        adaptive_analysis_path=adaptive_analysis_path,
        adaptive_analysis_sha256=adaptive_analysis_sha256,
        preregistration_path=preregistration_path,
        preregistration_sha256=preregistration_sha256,
        source_elastic_path=source_elastic_path,
        source_elastic_sha256=_sha256(source_elastic_path),
        source_endpoint_path=source_endpoint_path,
        source_endpoint_sha256=_sha256(source_endpoint_path),
        source_receipt_path=source_receipt_path,
        source_receipt_sha256=_sha256(source_receipt_path),
    )
    _require(rebuilt_profiles == (elastic_raw, endpoint_raw, promotion_receipt),
             "promoted profiles or receipt do not reproduce")
    elastic = load_elastic_profile(elastic_path)
    require_replicated_profile(elastic)
    endpoint = load_endpoint_service_profile(endpoint_path)
    _require(
        endpoint.deployment_scope == "frozen_validation"
        and endpoint.elastic_profile_fingerprint_sha256
        == elastic.fingerprint_sha256
        and endpoint.workload_manifest_sha256 == manifest_sha256
        and promotion_receipt.get("schema") == promotion.SCHEMA
        and promotion_receipt.get("fingerprint_sha256")
        == promotion._fingerprint(promotion_receipt)
        and promotion_receipt.get("controller_parameters_unchanged") is True
        and promotion_receipt.get("post_validation_tuning_allowed") is False,
        "promoted profile lineage or scope differs",
    )
    if candidate.get("kind") == "candidate_b_semantic_epoch_v1":
        _require(
            endpoint.routing_policy is not None
            and endpoint.routing_policy.policy
            == candidate["endpoint_routing_policy"]
            and promotion_receipt.get("candidate") == candidate,
            "semantic candidate policy changed during profile promotion",
        )
    else:
        _require(endpoint.routing_policy is None,
                 "instant-score candidate acquired a semantic routing policy")

    candidate_implementation_path = _entry_path(
        adaptive_run_contract, str(candidate["implementation_entry"]))
    adaptive_implementation_path = candidate_implementation_path
    candidate_implementation = json.loads(
        candidate_implementation_path.read_text(encoding="utf-8"))
    if candidate.get("kind") == "candidate_b_semantic_epoch_v1":
        parent = candidate_implementation.get(
            "adaptive_implementation_contract")
        _require(isinstance(parent, Mapping),
                 "semantic candidate lacks its adaptive implementation parent")
        adaptive_implementation_path = (
            repo_root / str(parent.get("path", ""))).resolve()
        if Path(str(parent.get("path", ""))).is_absolute():
            adaptive_implementation_path = Path(str(parent["path"])).resolve()
        _require(
            adaptive_implementation_path.is_file()
            and _sha256(adaptive_implementation_path) == parent.get("sha256"),
            "semantic candidate adaptive-parent digest differs",
        )
        semantic_implementation.verify_contract(
            repo_root=repo_root,
            contract_path=candidate_implementation_path,
            expected_sha256=_sha256(candidate_implementation_path),
            adaptive_contract=adaptive_implementation_path,
        )
    implementation_path, _implementation_raw = _load_bound(
        implementation_path, implementation_sha256,
        name="independent implementation contract")
    implementation_value = implementation.verify_contract(
        repo_root=repo_root,
        contract_path=implementation_path,
        expected_sha256=implementation_sha256,
        adaptive_contract=adaptive_implementation_path,
    )
    calibration_job_id = adaptive_analysis.get(
        "persistent_allocation_job_id")
    _require(type(calibration_job_id) is str and calibration_job_id.strip(),
             "adaptive analysis lacks calibration Slurm job ID")
    source_workload_path = manifest_builder._bound_path(
        manifest.get("source_workload"), name="source workload")
    preregistration_path = preregistration_path.resolve()
    value: dict[str, object] = {
        "schema": SCHEMA,
        "purpose": "one-shot held-out validation of frozen TEMPO candidate",
        "candidate": dict(candidate),
        "preregistration": _binding(preregistration_path),
        "independent_manifest": _binding(
            manifest_path,
            fingerprint_sha256=str(manifest["fingerprint_sha256"])),
        "source_workload": _binding(source_workload_path),
        "adaptive_screen_analysis": _binding(
            adaptive_analysis_path,
            fingerprint_sha256=str(adaptive_analysis["fingerprint_sha256"]),
        ),
        "candidate_screen_analysis": _binding(
            adaptive_analysis_path,
            fingerprint_sha256=str(adaptive_analysis["fingerprint_sha256"]),
        ),
        "adaptive_run_contract": _binding(
            adaptive_run_contract_path,
            fingerprint_sha256=str(
                adaptive_run_contract["fingerprint_sha256"]),
        ),
        "candidate_run_contract": _binding(
            adaptive_run_contract_path,
            fingerprint_sha256=str(
                adaptive_run_contract["fingerprint_sha256"]),
        ),
        "source_calibrated_elastic_profile": _binding(
            source_elastic_path,
            fingerprint_sha256=str(
                adaptive_run_contract["elastic_profile"][
                    "fingerprint_sha256"]),
        ),
        "source_calibrated_endpoint_profile": _binding(
            source_endpoint_path,
            fingerprint_sha256=str(
                adaptive_run_contract["endpoint_service_profile"][
                    "fingerprint_sha256"]),
        ),
        "source_calibrated_profile_receipt": _binding(
            source_receipt_path,
            fingerprint_sha256=str(
                adaptive_run_contract["profile_receipt"][
                    "fingerprint_sha256"]),
        ),
        "promoted_elastic_profile": _binding(
            elastic_path,
            fingerprint_sha256=elastic.fingerprint_sha256),
        "promoted_endpoint_service_profile": _binding(
            endpoint_path,
            fingerprint_sha256=endpoint.fingerprint_sha256),
        "profile_promotion_receipt": _binding(
            promotion_receipt_path,
            fingerprint_sha256=str(
                promotion_receipt["fingerprint_sha256"]),
        ),
        "adaptive_implementation_contract": _binding(
            adaptive_implementation_path,
            fingerprint_sha256=str(
                adaptive_run_contract["adaptive_implementation_contract"][
                    "fingerprint_sha256"]),
        ),
        "candidate_implementation_contract": _binding(
            candidate_implementation_path,
            fingerprint_sha256=str(
                adaptive_run_contract[str(candidate["implementation_entry"])][
                    "fingerprint_sha256"]),
        ),
        "independent_implementation_contract": _binding(
            implementation_path,
            fingerprint_sha256=str(
                implementation_value["fingerprint_sha256"]),
        ),
        "calibration_slurm_job_id": calibration_job_id,
        "validation_must_use_different_slurm_job": True,
        "fixed_runtime_environment": dict(sorted(
            independent_runtime_environment(candidate).items())),
        "success_gates": manifest["success_gates"],
        "slurm": manifest["slurm"],
        "transport": "LMCacheConnectorV1:UCX",
        "unchanged_pd_data_plane": True,
        "controller_parameters_unchanged": True,
        "controller_parameter_search_allowed": False,
        "post_validation_tuning_allowed": False,
        "independent_validation_authorized": True,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "claim_boundary": manifest["claim_boundary"],
    }
    if candidate.get("kind") == "candidate_b_semantic_epoch_v1":
        value.update({
            "endpoint_routing_policy": "semantic_epoch_v1",
            "passive_external_credit": True,
            "semantic_credit_contract": adaptive_run_contract[
                "semantic_credit_contract"],
            "endpoint_service_profile": dict(
                value["promoted_endpoint_service_profile"]),
        })
    value["fingerprint_sha256"] = contract_fingerprint(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest", "preregistration", "elastic",
        "endpoint", "promotion-receipt", "implementation",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument(
        "--candidate-analysis", "--adaptive-analysis",
        dest="adaptive_analysis", type=Path, required=True)
    parser.add_argument(
        "--candidate-analysis-sha256", "--adaptive-analysis-sha256",
        dest="adaptive_analysis_sha256", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(),
             "refusing to overwrite independent run contract")
    value = build_run_contract(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        adaptive_analysis_path=args.adaptive_analysis,
        adaptive_analysis_sha256=args.adaptive_analysis_sha256,
        preregistration_path=args.preregistration,
        preregistration_sha256=args.preregistration_sha256,
        elastic_path=args.elastic,
        elastic_sha256=args.elastic_sha256,
        endpoint_path=args.endpoint,
        endpoint_sha256=args.endpoint_sha256,
        promotion_receipt_path=args.promotion_receipt,
        promotion_receipt_sha256=args.promotion_receipt_sha256,
        implementation_path=args.implementation,
        implementation_sha256=args.implementation_sha256,
        repo_root=args.repo_root,
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
