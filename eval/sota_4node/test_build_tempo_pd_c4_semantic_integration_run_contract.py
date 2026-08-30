from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from eval.sota_4node import build_tempo_pd_c4_semantic_integration_run_contract as builder
from eval.sota_4node import build_tempo_pd_semantic_epoch_endpoint_profile as profile_builder
from tempo.pd_endpoint_profile import (
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ENDPOINT = (
    REPO_ROOT
    / "eval/sota_4node/real_tempo_pd_endpoint_service_profile_c4_screen_v1.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _entry(path: Path, **extra: object) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha(path), **extra}


def _fixture(root: Path, *, source_job: str = "job-17"):
    source_result = _write(root / "source-result.json", {
        "slurm_job_id": source_job,
    })
    generic = _write(root / "generic.json", {"bound": True})
    source_profile = load_endpoint_service_profile(SOURCE_ENDPOINT)
    base = {
        "schema": builder.adaptive.SCHEMA,
        "fingerprint_sha256": "a" * 64,
        "source_node_result": _entry(source_result),
        "source_workload": _entry(generic),
        "analysis": _entry(generic),
        "phase_manifest": _entry(generic),
        "elastic_profile": _entry(generic),
        "endpoint_service_profile": _entry(
            SOURCE_ENDPOINT,
            fingerprint_sha256=source_profile.fingerprint_sha256,
        ),
        "profile_receipt": _entry(generic),
        "offline_replay": _entry(generic),
        "fixed_c4_implementation_contract": _entry(generic),
        "adaptive_implementation_contract": _entry(generic),
        "profile_fit_formula": "test-formula",
        "slurm": {
            "nodes": 4,
            "gpus": 16,
            "interactive_time_limit": "04:00:00",
            "persistent_allocation_reuse_required": True,
            "login_node_experiment_execution_allowed": False,
        },
    }
    base_path = _write(root / "adaptive-contract.json", base)
    authorization_path = _write(root / "semantic-analysis.json", {
        "schema": builder.semantic_analysis.SCHEMA,
    })
    exploratory_contract_path = _write(root / "exploratory-contract.json", {
        "fingerprint_sha256": "b" * 64,
        "semantic_credit_contract": dict(
            profile_builder.SEMANTIC_ROUTING_POLICY),
    })
    endpoint_raw = profile_builder.build_profile(
        SOURCE_ENDPOINT,
        expected_base_sha256=_sha(SOURCE_ENDPOINT),
        profile_id="synthetic-c4-semantic-integration",
    )
    endpoint_path = _write(root / "semantic-endpoint.json", endpoint_raw)
    implementation_path = _write(root / "implementation.json", {})
    inputs = {
        "adaptive_contract_path": base_path,
        "adaptive_contract_sha256": _sha(base_path),
        "semantic_analysis_path": authorization_path,
        "semantic_analysis_sha256": _sha(authorization_path),
        "semantic_endpoint_path": endpoint_path,
        "semantic_endpoint_sha256": _sha(endpoint_path),
        "implementation_path": implementation_path,
        "implementation_sha256": _sha(implementation_path),
        "repo_root": REPO_ROOT,
    }
    return inputs, exploratory_contract_path


def _build(inputs, exploratory_contract_path, *, semantic_job="job-17"):
    with (
        patch.object(builder, "_validate_base_adaptive_contract"),
        patch.object(
            builder,
            "_validate_semantic_authorization",
            return_value=(
                exploratory_contract_path,
                json.loads(exploratory_contract_path.read_text(encoding="utf-8")),
                semantic_job,
            ),
        ),
        patch.object(
            builder.implementation,
            "verify_contract",
            return_value={"fingerprint_sha256": "c" * 64, "files": []},
        ),
    ):
        return builder.build_run_contract(**inputs)


def test_approved_policy_is_bound_unchanged_to_new_c4_rows():
    with tempfile.TemporaryDirectory() as directory:
        inputs, exploratory = _fixture(Path(directory))
        value = _build(inputs, exploratory)
        assert value["schema"] == builder.SCHEMA
        assert value["fingerprint_sha256"] == builder.contract_fingerprint(value)
        assert value["semantic_credit_contract"] == (
            profile_builder.SEMANTIC_ROUTING_POLICY)
        assert value["endpoint_routing_policy"] == "semantic_epoch_v1"
        assert value["passive_external_credit"] is True
        assert value["same_allocation_calibration_required"] is True
        assert value["fixed_runtime_environment"][
            "TEMPO_PD_ENDPOINT_ROUTING_POLICY"] == "semantic_epoch_v1"
        assert value["fixed_runtime_environment"][
            "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK"] == "1"
        assert value["endpoint_service_profile"]["derived_from_sha256"] == (
            value["source_endpoint_service_profile"]["sha256"])
        assert value["performance_claim_allowed"] is False


def test_semantic_and_c4_must_share_the_persistent_allocation():
    with tempfile.TemporaryDirectory() as directory:
        inputs, exploratory = _fixture(Path(directory), source_job="job-18")
        with pytest.raises(ValueError, match="reuse one Slurm job"):
            _build(inputs, exploratory, semantic_job="job-17")


def test_policy_drift_in_the_derived_profile_fails_closed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inputs, exploratory = _fixture(root)
        endpoint = json.loads(
            inputs["semantic_endpoint_path"].read_text(encoding="utf-8"))
        drifted = copy.deepcopy(endpoint)
        drifted["routing_policy"]["decoder_high_water_numerator"] = 2
        drifted["fingerprint_sha256"] = endpoint_service_profile_fingerprint(
            drifted)
        drifted_path = _write(root / "drifted-endpoint.json", drifted)
        inputs["semantic_endpoint_path"] = drifted_path
        inputs["semantic_endpoint_sha256"] = _sha(drifted_path)
        with pytest.raises(ValueError, match="unchanged-policy derivation"):
            _build(inputs, exploratory)
