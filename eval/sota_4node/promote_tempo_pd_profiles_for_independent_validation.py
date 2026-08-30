#!/usr/bin/env python3
"""Promote calibrated profiles without changing any controller parameter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import build_tempo_pd_independent_validation_manifest as manifest_builder
from eval.sota_4node import build_tempo_pd_c4_calibrated_profiles as calibrated
from tempo.pd_elastic_profile import load_elastic_profile, require_replicated_profile
from tempo.pd_endpoint_profile import (
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)


SCHEMA = "tempo-pd-independent-profile-promotion-receipt-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object, *, name: str) -> str:
    return manifest_builder._canonical_sha(value, name=name)


def _load_bound(
    path: Path, expected_sha256: str, *, name: str,
) -> tuple[Path, dict[str, object]]:
    path = path.resolve()
    expected_sha256 = _canonical_sha(expected_sha256, name=f"{name} SHA-256")
    _require(path.is_file() and _sha256(path) == expected_sha256,
             f"{name} digest differs")
    return path, manifest_builder._load_object(path, name=name)


def _fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding(
    path: Path, *, fingerprint_sha256: str | None = None,
) -> dict[str, str]:
    value = {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
    if fingerprint_sha256 is not None:
        value["fingerprint_sha256"] = fingerprint_sha256
    return value


def _adaptive_contract_from_manifest(
    manifest: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    path = manifest_builder._bound_path(
        manifest.get("candidate_run_contract"), name="candidate run contract")
    candidate = manifest.get("candidate")
    _require(isinstance(candidate, Mapping),
             "independent manifest candidate binding is missing")
    analyzer = (
        manifest_builder.semantic_analysis
        if candidate.get("kind") == "candidate_b_semantic_epoch_v1"
        else manifest_builder.adaptive_analysis)
    value = analyzer._validate_run_contract(path)
    _require(
        manifest["candidate_run_contract"].get("fingerprint_sha256")
        == value.get("fingerprint_sha256"),
        "independent manifest/candidate contract fingerprint differs",
    )
    return path, value


def _promote_raw_profiles(
    source_elastic_raw: Mapping[str, object],
    source_endpoint_raw: Mapping[str, object],
    *, manifest_sha256: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    """Change only validation identity/scope bindings, never policy values."""

    manifest_sha256 = _canonical_sha(
        manifest_sha256, name="independent manifest SHA-256")
    promoted_elastic = copy.deepcopy(dict(source_elastic_raw))
    promoted_elastic["profile_id"] = (
        str(source_elastic_raw["profile_id"]) + "-frozen-validation")
    promoted_elastic["deployment_scope"] = "replicated"
    encoded_elastic = json.dumps(
        promoted_elastic, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    promoted_elastic_fingerprint = hashlib.sha256(encoded_elastic).hexdigest()

    promoted_endpoint = copy.deepcopy(dict(source_endpoint_raw))
    promoted_endpoint["profile_id"] = (
        str(source_endpoint_raw["profile_id"]) + "-frozen-validation")
    promoted_endpoint["elastic_profile_fingerprint_sha256"] = (
        promoted_elastic_fingerprint)
    promoted_endpoint["workload_manifest_sha256"] = manifest_sha256
    promoted_endpoint["deployment_scope"] = "frozen_validation"
    promoted_endpoint["fingerprint_sha256"] = (
        endpoint_service_profile_fingerprint(promoted_endpoint))
    _require(
        promoted_elastic["identity"] == source_elastic_raw["identity"]
        and promoted_elastic["controller"] == source_elastic_raw["controller"]
        and promoted_elastic["rows"] == source_elastic_raw["rows"]
        and promoted_endpoint["default_e2e_deadline_ms"]
        == source_endpoint_raw["default_e2e_deadline_ms"]
        and promoted_endpoint["controller"] == source_endpoint_raw["controller"]
        and promoted_endpoint["rows"] == source_endpoint_raw["rows"]
        and promoted_endpoint.get("schema") == source_endpoint_raw.get("schema")
        and promoted_endpoint.get("routing_policy")
        == source_endpoint_raw.get("routing_policy"),
        "profile promotion changed controller, identity, rows, or numeric values",
    )
    return promoted_elastic, promoted_endpoint, promoted_elastic_fingerprint


def _contract_source(
    contract: Mapping[str, object], name: str, supplied: Path,
    supplied_sha256: str,
) -> Path:
    expected = contract.get(name)
    _require(isinstance(expected, Mapping),
             f"adaptive contract lacks {name}")
    path = supplied.resolve()
    _require(
        path == Path(str(expected.get("path"))).resolve()
        and _sha256(path) == supplied_sha256 == expected.get("sha256"),
        f"supplied {name} differs from adaptive contract",
    )
    return path


def promote_profiles(
    *, manifest_path: Path, manifest_sha256: str,
    adaptive_analysis_path: Path, adaptive_analysis_sha256: str,
    preregistration_path: Path, preregistration_sha256: str,
    source_elastic_path: Path, source_elastic_sha256: str,
    source_endpoint_path: Path, source_endpoint_sha256: str,
    source_receipt_path: Path, source_receipt_sha256: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
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
        and manifest.get("schema") == manifest_builder.SCHEMA
        and manifest.get("fingerprint_sha256")
        == manifest_builder.manifest_fingerprint(manifest)
        and manifest.get("profile_promotion") == {
            "controller_parameter_changes_allowed": False,
            "elastic_numeric_or_row_changes_allowed": False,
            "endpoint_numeric_or_row_changes_allowed": False,
            "elastic_deployment_scope": "replicated",
            "endpoint_deployment_scope": "frozen_validation",
            "endpoint_workload_binding": "exact_independent_manifest_sha256",
        },
        "independent manifest does not authorize exact profile promotion",
    )
    adaptive_contract_path, adaptive_contract = (
        _adaptive_contract_from_manifest(manifest)
    )
    source_elastic_sha256 = _canonical_sha(
        source_elastic_sha256, name="source Elastic profile SHA-256")
    source_endpoint_sha256 = _canonical_sha(
        source_endpoint_sha256, name="source endpoint profile SHA-256")
    source_receipt_sha256 = _canonical_sha(
        source_receipt_sha256, name="source profile receipt SHA-256")
    source_elastic_path = _contract_source(
        adaptive_contract, "elastic_profile", source_elastic_path,
        source_elastic_sha256)
    source_endpoint_path = _contract_source(
        adaptive_contract, "endpoint_service_profile", source_endpoint_path,
        source_endpoint_sha256)
    source_receipt_path = _contract_source(
        adaptive_contract, "profile_receipt", source_receipt_path,
        source_receipt_sha256)
    source_elastic_raw = manifest_builder._load_object(
        source_elastic_path, name="source Elastic profile")
    source_endpoint_raw = manifest_builder._load_object(
        source_endpoint_path, name="source endpoint profile")
    source_receipt = manifest_builder._load_object(
        source_receipt_path, name="source profile receipt")
    source_elastic = load_elastic_profile(source_elastic_path)
    source_endpoint = load_endpoint_service_profile(source_endpoint_path)
    _require(
        source_elastic.deployment_scope == "screen_only"
        and source_endpoint.deployment_scope == "calibration_only"
        and source_endpoint.elastic_profile_fingerprint_sha256
        == source_elastic.fingerprint_sha256
        and source_receipt.get("schema") == calibrated.SCHEMA
        and source_receipt.get("fingerprint_sha256")
        == calibrated._receipt_fingerprint(source_receipt)
        and source_receipt.get("calibration_only") is True
        and source_receipt.get("performance_claim_allowed") is False,
        "source calibrated profile lineage differs",
    )

    promoted_elastic, promoted_endpoint, promoted_elastic_fingerprint = (
        _promote_raw_profiles(
            source_elastic_raw,
            source_endpoint_raw,
            manifest_sha256=manifest_sha256,
        )
    )
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "purpose": "metadata-only promotion after authorized candidate screen",
        "candidate": manifest["candidate"],
        "independent_manifest": _binding(
            manifest_path,
            fingerprint_sha256=str(manifest["fingerprint_sha256"])),
        "adaptive_screen_analysis": _binding(
            adaptive_analysis_path,
            fingerprint_sha256=str(
                manifest["adaptive_screen_analysis"]["fingerprint_sha256"]),
        ),
        "candidate_screen_analysis": _binding(
            adaptive_analysis_path,
            fingerprint_sha256=str(
                manifest["candidate_screen_analysis"]["fingerprint_sha256"]),
        ),
        "adaptive_run_contract": _binding(
            adaptive_contract_path,
            fingerprint_sha256=str(adaptive_contract["fingerprint_sha256"]),
        ),
        "candidate_run_contract": _binding(
            adaptive_contract_path,
            fingerprint_sha256=str(adaptive_contract["fingerprint_sha256"]),
        ),
        "source_elastic_profile": _binding(
            source_elastic_path,
            fingerprint_sha256=source_elastic.fingerprint_sha256),
        "source_endpoint_profile": _binding(
            source_endpoint_path,
            fingerprint_sha256=source_endpoint.fingerprint_sha256),
        "source_profile_receipt": _binding(
            source_receipt_path,
            fingerprint_sha256=str(source_receipt["fingerprint_sha256"]),
        ),
        "promoted_elastic_profile": {
            "profile_id": promoted_elastic["profile_id"],
            "fingerprint_sha256": promoted_elastic_fingerprint,
            "deployment_scope": "replicated",
        },
        "promoted_endpoint_profile": {
            "profile_id": promoted_endpoint["profile_id"],
            "fingerprint_sha256": promoted_endpoint["fingerprint_sha256"],
            "deployment_scope": "frozen_validation",
            "workload_manifest_sha256": manifest_sha256,
        },
        "changed_fields": {
            "elastic": ["profile_id", "deployment_scope"],
            "endpoint": [
                "profile_id", "elastic_profile_fingerprint_sha256",
                "workload_manifest_sha256", "deployment_scope",
                "fingerprint_sha256",
            ],
        },
        "controller_parameters_unchanged": True,
        "identity_and_rows_unchanged": True,
        "controller_parameter_search_allowed": False,
        "post_validation_tuning_allowed": False,
        "performance_claim_allowed": False,
    }
    receipt["fingerprint_sha256"] = _fingerprint(receipt)
    return promoted_elastic, promoted_endpoint, receipt


def validate_promoted_profiles(
    *, elastic_path: Path, endpoint_path: Path,
    expected_manifest_sha256: str,
) -> tuple[str, str]:
    elastic = load_elastic_profile(elastic_path.resolve())
    require_replicated_profile(elastic)
    endpoint = load_endpoint_service_profile(endpoint_path.resolve())
    _require(
        endpoint.deployment_scope == "frozen_validation"
        and endpoint.elastic_profile_fingerprint_sha256
        == elastic.fingerprint_sha256
        and endpoint.workload_manifest_sha256 == expected_manifest_sha256,
        "promoted profiles failed strict validation binding",
    )
    return elastic.fingerprint_sha256, endpoint.fingerprint_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest", "preregistration", "source-elastic",
        "source-endpoint", "source-receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument(
        "--candidate-analysis", "--adaptive-analysis",
        dest="adaptive_analysis", type=Path, required=True)
    parser.add_argument(
        "--candidate-analysis-sha256", "--adaptive-analysis-sha256",
        dest="adaptive_analysis_sha256", required=True)
    parser.add_argument("--elastic-output", type=Path, required=True)
    parser.add_argument("--endpoint-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.elastic_output, args.endpoint_output, args.receipt_output)
    _require(len({path.resolve() for path in outputs}) == 3,
             "promotion outputs must be distinct")
    _require(all(not path.exists() for path in outputs),
             "refusing to overwrite promoted profile output")
    elastic, endpoint, receipt = promote_profiles(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        adaptive_analysis_path=args.adaptive_analysis,
        adaptive_analysis_sha256=args.adaptive_analysis_sha256,
        preregistration_path=args.preregistration,
        preregistration_sha256=args.preregistration_sha256,
        source_elastic_path=args.source_elastic,
        source_elastic_sha256=args.source_elastic_sha256,
        source_endpoint_path=args.source_endpoint,
        source_endpoint_sha256=args.source_endpoint_sha256,
        source_receipt_path=args.source_receipt,
        source_receipt_sha256=args.source_receipt_sha256,
    )
    for path, value in zip(outputs, (elastic, endpoint, receipt), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elastic_fingerprint, endpoint_fingerprint = validate_promoted_profiles(
        elastic_path=args.elastic_output,
        endpoint_path=args.endpoint_output,
        expected_manifest_sha256=args.manifest_sha256,
    )
    _require(
        elastic_fingerprint
        == receipt["promoted_elastic_profile"]["fingerprint_sha256"]
        and endpoint_fingerprint
        == receipt["promoted_endpoint_profile"]["fingerprint_sha256"],
        "published profile fingerprints differ from promotion receipt",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "elastic_fingerprint_sha256": elastic_fingerprint,
        "endpoint_fingerprint_sha256": endpoint_fingerprint,
        "receipt_fingerprint_sha256": receipt["fingerprint_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
