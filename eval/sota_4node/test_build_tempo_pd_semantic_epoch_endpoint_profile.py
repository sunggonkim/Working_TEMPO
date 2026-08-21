from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from eval.sota_4node import build_tempo_pd_semantic_epoch_endpoint_profile as builder
from tempo.pd_endpoint_profile import SCHEMA_V2, load_endpoint_service_profile


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "eval/sota_4node/real_tempo_pd_endpoint_service_profile_c4_screen_v1.json"
FROZEN = ROOT / "eval/sota_4node/real_tempo_pd_endpoint_service_profile_c4_semantic_epoch_v2.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_semantic_profile_is_exact_deterministic_derivation() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    rebuilt = builder.build_profile(
        BASE, expected_base_sha256=_sha256(BASE))
    assert rebuilt == frozen
    assert rebuilt["controller"] == base["controller"]
    assert rebuilt["rows"] == base["rows"]
    assert rebuilt["default_e2e_deadline_ms"] == base[
        "default_e2e_deadline_ms"]
    profile = load_endpoint_service_profile(FROZEN)
    assert profile.schema == SCHEMA_V2
    assert profile.routing_policy is not None
    assert profile.routing_policy.as_dict() == builder.SEMANTIC_ROUTING_POLICY


def test_credit_epoch_profile_adds_only_profile_bound_policy_fields() -> None:
    source = json.loads(BASE.read_text(encoding="utf-8"))
    value = builder.build_profile(
        BASE,
        expected_base_sha256=_sha256(BASE),
        profile_id=builder.CREDIT_EPOCH_PROFILE_ID,
    )
    assert value["controller"] == source["controller"]
    assert value["rows"] == source["rows"]
    assert value["routing_policy"] == {
        **builder.SEMANTIC_ROUTING_POLICY,
        **builder.CREDIT_EPOCH_POLICY_ADDITIONS,
    }


def test_semantic_profile_derivation_rejects_source_digest_drift() -> None:
    with pytest.raises(ValueError, match="digest differs"):
        builder.build_profile(BASE, expected_base_sha256="0" * 64)


def test_semantic_profile_policy_is_part_of_fingerprint() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(frozen)
    mutated["routing_policy"]["epoch_confirmation_requests"] = 3
    assert builder.endpoint_service_profile_fingerprint(mutated) != frozen[
        "fingerprint_sha256"]
