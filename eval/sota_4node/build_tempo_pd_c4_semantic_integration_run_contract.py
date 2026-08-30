#!/usr/bin/env python3
"""Bind an approved semantic policy to newly calibrated C4 endpoint rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_semantic_epoch_screen as semantic_analysis
from eval.sota_4node import build_tempo_pd_c4_adaptive_run_contract as adaptive
from eval.sota_4node import build_tempo_pd_semantic_epoch_endpoint_profile as profile_builder
from eval.sota_4node import verify_tempo_pd_c4_implementation as fixed
from eval.sota_4node import verify_tempo_pd_c4_semantic_integration_implementation as implementation
from tempo.pd_endpoint_profile import (
    SCHEMA_V1,
    SCHEMA_V2,
    load_endpoint_service_profile,
)


SCHEMA = "tempo-pd-c4-semantic-integration-screen-run-contract-v1"
RUN_CONTRACT_ENV = "TEMPO_PD_C4_SEMANTIC_INTEGRATION_RUN_CONTRACT"
RUN_CONTRACT_SHA_ENV = (
    "TEMPO_PD_C4_SEMANTIC_INTEGRATION_RUN_CONTRACT_SHA256")
SEMANTIC_FIXED_RUNTIME_ENVIRONMENT = {
    **{
        name: value
        for name, value in adaptive.ADAPTIVE_FIXED_RUNTIME_ENVIRONMENT.items()
        if name != "TEMPO_PD_C4_ADAPTIVE_APPROVED"
    },
    "TEMPO_PD_C4_SEMANTIC_INTEGRATION_APPROVED": "YES",
    "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "semantic_epoch_v1",
    "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "1",
}


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
    fixed._canonical_sha(expected_sha256, name=f"{name} SHA-256")
    _require(path.is_file() and _sha256(path) == expected_sha256,
             f"{name} digest differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be an object")
    return path, value


def _bound_entry(
    contract: Mapping[str, object], name: str, *, repo_root: Path,
) -> tuple[Path, Mapping[str, object]]:
    entry = contract.get(name)
    _require(isinstance(entry, Mapping), f"base adaptive contract lacks {name}")
    _require(
        type(entry.get("path")) is str
        and type(entry.get("sha256")) is str,
        f"base adaptive {name} binding differs",
    )
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    _require(path.is_file() and _sha256(path) == entry["sha256"],
             f"base adaptive {name} digest differs")
    return path, entry


def _binding(
    path: Path, *, fingerprint_sha256: str | None = None,
    **extra: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path.resolve()),
    }
    if fingerprint_sha256 is not None:
        value["fingerprint_sha256"] = fingerprint_sha256
    value.update(extra)
    return value


def _validate_base_adaptive_contract(
    path: Path, value: Mapping[str, object], *, repo_root: Path,
) -> None:
    _require(
        value.get("schema") == adaptive.SCHEMA
        and value.get("fingerprint_sha256")
        == adaptive.contract_fingerprint(value)
        and value.get("offline_replay_authorized") is True
        and value.get("calibration_only") is True
        and value.get("performance_claim_allowed") is False,
        "base adaptive run contract differs",
    )
    arguments: dict[str, object] = {}
    for argument, entry_name in (
        ("analysis", "analysis"),
        ("manifest", "phase_manifest"),
        ("elastic", "elastic_profile"),
        ("endpoint", "endpoint_service_profile"),
        ("receipt", "profile_receipt"),
        ("replay", "offline_replay"),
        ("implementation", "adaptive_implementation_contract"),
    ):
        artifact, entry = _bound_entry(value, entry_name, repo_root=repo_root)
        arguments[f"{argument}_path"] = artifact
        arguments[f"{argument}_sha256"] = str(entry["sha256"])
    rebuilt = adaptive.build_run_contract(**arguments, repo_root=repo_root)
    _require(rebuilt == value,
             f"base adaptive run contract does not reproduce: {path}")


def _validate_semantic_authorization(
    path: Path, value: Mapping[str, object], *, repo_root: Path,
) -> tuple[Path, dict[str, object], str]:
    _require(
        value.get("schema") == semantic_analysis.SCHEMA
        and value.get("authorizes_candidate_for_final_c4_integration") is True
        and value.get("semantic_correctness_and_exercise_pass") is True
        and value.get("original_screen_performance_gate_pass") is True
        and value.get("post_screen_parameter_tuning_allowed") is False
        and value.get("performance_claim_allowed") is False,
        "semantic exploratory analysis does not authorize integration",
    )
    result_path = Path(str(value.get("source_result", ""))).resolve()
    base_analysis_path = Path(str(value.get("base_phase_analysis", ""))).resolve()
    reproduced = semantic_analysis.analyze(
        result_path=result_path,
        expected_result_sha256=str(value.get("source_result_sha256", "")),
        base_analysis_path=base_analysis_path,
        expected_base_analysis_sha256=str(
            value.get("base_phase_analysis_sha256", "")),
    )
    _require(reproduced == value,
             f"semantic exploratory analysis does not reproduce: {path}")
    exploratory_contract_path = Path(str(value.get("run_contract", "")))
    if not exploratory_contract_path.is_absolute():
        exploratory_contract_path = repo_root / exploratory_contract_path
    exploratory_contract_path = exploratory_contract_path.resolve()
    _require(
        exploratory_contract_path.is_file()
        and _sha256(exploratory_contract_path)
        == value.get("run_contract_sha256"),
        "semantic exploratory run-contract binding differs",
    )
    exploratory_contract = json.loads(
        exploratory_contract_path.read_text(encoding="utf-8"))
    _require(
        exploratory_contract.get("semantic_credit_contract")
        == profile_builder.SEMANTIC_ROUTING_POLICY
        and exploratory_contract.get("endpoint_routing_policy")
        == "semantic_epoch_v1"
        and exploratory_contract.get("passive_external_credit") is True,
        "semantic exploratory policy differs from the frozen policy object",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    job_id = result.get("slurm_job_id")
    _require(type(job_id) is str and bool(job_id.strip()),
             "semantic exploratory result lacks a Slurm job ID")
    return exploratory_contract_path, exploratory_contract, job_id


def build_run_contract(
    *, adaptive_contract_path: Path, adaptive_contract_sha256: str,
    semantic_analysis_path: Path, semantic_analysis_sha256: str,
    semantic_endpoint_path: Path, semantic_endpoint_sha256: str,
    implementation_path: Path, implementation_sha256: str,
    repo_root: Path,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    adaptive_contract_path, base = _load_bound(
        adaptive_contract_path, adaptive_contract_sha256,
        name="base adaptive run contract",
    )
    _validate_base_adaptive_contract(
        adaptive_contract_path, base, repo_root=repo_root)
    semantic_analysis_path, authorization = _load_bound(
        semantic_analysis_path, semantic_analysis_sha256,
        name="semantic exploratory analysis",
    )
    exploratory_contract_path, exploratory_contract, semantic_job_id = (
        _validate_semantic_authorization(
            semantic_analysis_path, authorization, repo_root=repo_root)
    )

    source_result_path, _source_result_entry = _bound_entry(
        base, "source_node_result", repo_root=repo_root)
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    _require(
        source_result.get("slurm_job_id") == semantic_job_id,
        "semantic exploratory and C4 calibration must reuse one Slurm job",
    )
    source_endpoint_path, source_endpoint_entry = _bound_entry(
        base, "endpoint_service_profile", repo_root=repo_root)
    source_endpoint = load_endpoint_service_profile(source_endpoint_path)
    _require(
        source_endpoint.schema == SCHEMA_V1
        and source_endpoint.routing_policy is None
        and source_endpoint.fingerprint_sha256
        == source_endpoint_entry.get("fingerprint_sha256"),
        "base adaptive endpoint profile is not a calibrated v1 profile",
    )
    semantic_endpoint_path, semantic_endpoint_raw = _load_bound(
        semantic_endpoint_path, semantic_endpoint_sha256,
        name="semantic integration endpoint profile",
    )
    rebuilt_endpoint = profile_builder.build_profile(
        source_endpoint_path,
        expected_base_sha256=str(source_endpoint_entry["sha256"]),
        profile_id=str(semantic_endpoint_raw.get("profile_id", "")),
    )
    semantic_endpoint = load_endpoint_service_profile(semantic_endpoint_path)
    _require(
        semantic_endpoint_raw == rebuilt_endpoint
        and semantic_endpoint.schema == SCHEMA_V2
        and semantic_endpoint.routing_policy is not None
        and semantic_endpoint.routing_policy.as_dict()
        == profile_builder.SEMANTIC_ROUTING_POLICY
        == exploratory_contract.get("semantic_credit_contract")
        and semantic_endpoint.elastic_profile_fingerprint_sha256
        == source_endpoint.elastic_profile_fingerprint_sha256
        and semantic_endpoint.workload_manifest_sha256
        == source_endpoint.workload_manifest_sha256,
        "semantic endpoint is not an unchanged-policy derivation of C4 v1",
    )

    adaptive_implementation_path, _adaptive_impl_entry = _bound_entry(
        base, "adaptive_implementation_contract", repo_root=repo_root)
    implementation_path, _implementation_raw = _load_bound(
        implementation_path, implementation_sha256,
        name="semantic integration implementation contract",
    )
    implementation_value = implementation.verify_contract(
        repo_root=repo_root,
        contract_path=implementation_path,
        expected_sha256=implementation_sha256,
        adaptive_contract=adaptive_implementation_path,
    )

    copied = {}
    for name in (
        "source_node_result", "source_workload", "analysis", "phase_manifest",
        "elastic_profile", "profile_receipt", "offline_replay",
        "fixed_c4_implementation_contract", "adaptive_implementation_contract",
    ):
        artifact, entry = _bound_entry(base, name, repo_root=repo_root)
        copied[name] = {
            **_binding(artifact),
            **{
                key: item for key, item in entry.items()
                if key not in {"path", "sha256"}
            },
        }
    value: dict[str, object] = {
        "schema": SCHEMA,
        "purpose": (
            "calibration-only four-arm integration screen of the approved "
            "semantic-epoch policy on newly measured C4 service rows"),
        **copied,
        "base_adaptive_run_contract": _binding(
            adaptive_contract_path,
            fingerprint_sha256=str(base["fingerprint_sha256"])),
        "semantic_exploratory_analysis": _binding(semantic_analysis_path),
        "semantic_exploratory_run_contract": _binding(
            exploratory_contract_path,
            fingerprint_sha256=str(exploratory_contract["fingerprint_sha256"])),
        "source_endpoint_service_profile": _binding(
            source_endpoint_path,
            fingerprint_sha256=source_endpoint.fingerprint_sha256,
            schema=source_endpoint.schema,
        ),
        "endpoint_service_profile": _binding(
            semantic_endpoint_path,
            fingerprint_sha256=semantic_endpoint.fingerprint_sha256,
            schema=semantic_endpoint.schema,
            derived_from_sha256=str(source_endpoint_entry["sha256"]),
        ),
        "semantic_integration_implementation_contract": _binding(
            implementation_path,
            fingerprint_sha256=str(
                implementation_value["fingerprint_sha256"])),
        "profile_fit_formula": base["profile_fit_formula"],
        "semantic_credit_contract": dict(
            profile_builder.SEMANTIC_ROUTING_POLICY),
        "endpoint_routing_policy": "semantic_epoch_v1",
        "passive_external_credit": True,
        "fixed_runtime_environment": dict(sorted(
            SEMANTIC_FIXED_RUNTIME_ENVIRONMENT.items())),
        "slurm": base["slurm"],
        "transport": "LMCacheConnectorV1:UCX",
        "unchanged_pd_data_plane": True,
        "offline_replay_authorized": True,
        "semantic_policy_authorized": True,
        "same_allocation_calibration_required": True,
        "controller_parameter_search_allowed": False,
        "calibration_only": True,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "independent_validation_required": True,
    }
    value["fingerprint_sha256"] = contract_fingerprint(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "adaptive-contract", "semantic-analysis", "semantic-endpoint",
        "implementation",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(),
             "refusing to overwrite semantic integration run contract")
    value = build_run_contract(
        adaptive_contract_path=args.adaptive_contract,
        adaptive_contract_sha256=args.adaptive_contract_sha256,
        semantic_analysis_path=args.semantic_analysis,
        semantic_analysis_sha256=args.semantic_analysis_sha256,
        semantic_endpoint_path=args.semantic_endpoint,
        semantic_endpoint_sha256=args.semantic_endpoint_sha256,
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
