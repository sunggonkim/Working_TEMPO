from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "paper/tempo_go/current_evidence_manifest.json"
BUILDER_PATH = ROOT / "paper/tempo_go/build_current_evidence_manifest.py"
SPEC = importlib.util.spec_from_file_location("current_evidence_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def value() -> dict[str, object]:
    loaded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_current_manifest_is_exact_builder_output() -> None:
    assert value() == builder.build_payload()


def test_current_manifest_preserves_native_and_failclosed_business_semantics() -> None:
    manifest = value()
    candidate = manifest["campaigns"]["candidate_o"]
    assert candidate["native_analysis"]["sha256"] == (
        "1d5f9c5a785ffd0b3a35c9bc2709fbb13b45af86d099d869e9fef0da9341b9a5"
    )
    assert candidate["posthoc_business_analysis"]["sha256"] == (
        "850c2858a30473235fb9bbfaa0c8940abf94405e4243607db96e2e1935cd3590"
    )
    assert candidate["business"]["foreground"]["completed"] == 207
    assert candidate["business"]["foreground"]["failures"] == 3
    assert candidate["business"]["background"]["completed"] == 2004
    assert candidate["business"]["background"]["failures"] == 40
    assert candidate["business"]["background"]["global_rejects"] == 704
    assert candidate["receipt_integrity"] is True
    assert candidate["causal_discovery_positive"] is False
    assert candidate["performance_claim_allowed"] is False


def test_current_manifest_fails_candidate_mechanism_and_cross_run_claim_closed() -> None:
    manifest = value()
    mechanism = manifest["candidate_o_changed_mechanism"]
    assert mechanism == {
        "activated": False,
        "all_global_decisions": 1614,
        "causal_mechanism_positive": False,
        "changed_input": "telemetry_failure_quarantine_scope: pair -> route",
        "nonzero_route_failure_counter_decisions": 0,
        "route_failure_quarantine_rejections": 0,
    }
    context = manifest["candidate_o_vs_m_cross_allocation_context"]
    assert context["foreground_completed_delta"] == 4
    assert context["background_completed_delta"] == 526
    assert context["causal_comparison_allowed"] is False
    assert manifest["claim_state"][
        "candidate_o_changed_mechanism_activated"] is False
    preregistered = manifest["preregistered_next_diagnostic"]
    assert preregistered["policy_delta_from_candidate_o"] is False
    assert preregistered["native_result_exists"] is True
    assert preregistered["performance_claim_allowed"] is False
    assert preregistered["causal_discovery_positive"] is False
    result = preregistered["result"]
    assert result["allocation"] == "57740736"
    assert result["status"] == "complete"
    assert result["business"]["foreground"]["completed"] == 210
    assert result["business"]["background"]["completed"] == 1898
    assert result["business"]["background"]["global_rejects"] == 705
    assert result["observer"] == {
        "supported": 102,
        "total": 210,
        "fraction": 102 / 210,
    }
