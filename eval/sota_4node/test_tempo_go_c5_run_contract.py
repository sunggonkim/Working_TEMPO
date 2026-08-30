from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.sota_4node import tempo_go_c5_run_contract as contract
from eval.sota_4node import replay_tempo_go_c5_five_arm as replay


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / (
    "results/tempo_go_c5_candidate_i_telemetry_survivor_v1/"
    "native_run_contract.json"
)
WORKLOAD_INPUT = ROOT / "results/tempo_go_c5_heldout_output128_v1"
MANIFEST = WORKLOAD_INPUT / "tempo_go_workload_manifest.json"
WORKLOAD = WORKLOAD_INPUT / "workloads/validation.jsonl"
MODEL = ROOT / "models/Qwen2.5-7B-Instruct"
GUARD_PROFILE = ROOT / (
    "results/tempo_go_c5_candidate_i_telemetry_survivor_v1/"
    "frozen_global_profile.json"
)
DISABLED_PROFILE = ROOT / (
    "results/tempo_go_c5_heldout_frozen_proxy_v1/"
    "frozen_global_profile.json"
)
ELASTIC_PROFILE = ROOT / (
    "results/tempo_go_c5_anchor_priors_c12_v3_retry1/"
    "real_tempo_pd_elastic_profile_c12_anchor_output2_screen_v3.json"
)
ENDPOINT_PROFILE = ROOT / (
    "results/tempo_go_c5_heldout_frozen_proxy_v1/"
    "frozen_endpoint_service_profile.json"
)


def test_historical_frozen_contract_rejects_current_source_drift() -> None:
    """A retired immutable C5 run must not be rebound to newer source."""

    with pytest.raises(ValueError, match=r"source .* digest differs"):
        contract.verify_contract(
            CONTRACT,
            contract.sha256(CONTRACT),
            repo_root=ROOT,
            workload_input=WORKLOAD_INPUT,
        )


def test_contract_sha_and_fingerprint_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="digest differs"):
        contract.verify_contract(
            CONTRACT, "0" * 64, repo_root=ROOT,
            workload_input=WORKLOAD_INPUT,
        )
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["candidate"]["revision"] = "tampered"
    assert value["fingerprint_sha256"] != contract.contract_fingerprint(value)


def test_contract_rejects_a_different_workload(tmp_path: Path) -> None:
    other = tmp_path / "validation.jsonl"
    other.write_text(
        Path(value_path()).read_text(encoding="utf-8"), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="supplied C5 workload differs"):
        contract.verify_contract(
            CONTRACT, contract.sha256(CONTRACT), repo_root=ROOT,
            workload_input=other,
        )


def value_path() -> str:
    return str(
        ROOT / "results/tempo_go_c5_heldout_output128_v1"
        / "workloads/validation.jsonl"
    )


def test_arm_environment_requires_exact_frozen_values() -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    expected = contract.expected_environment(value, "tempo")
    contract.validate_environment(value, "tempo", expected)
    expected["TEMPO_PD_PRESSURE_MODE"] = "enabled"
    with pytest.raises(ValueError, match="TEMPO_PD_PRESSURE_MODE"):
        contract.validate_environment(value, "tempo", expected)


def test_failure_replay_rejects_quarantine_disabled_profile() -> None:
    with pytest.raises(ValueError, match="route_failure_quarantine_mode"):
        replay.replay(
            manifest_path=MANIFEST,
            workload_path=WORKLOAD,
            model_path=MODEL,
            global_profile_path=DISABLED_PROFILE,
            elastic_profile_path=ELASTIC_PROFILE,
            endpoint_profile_path=ENDPOINT_PROFILE,
            failure_index=0,
        )
