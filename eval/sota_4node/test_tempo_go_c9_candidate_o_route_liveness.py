from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tempo.pd_global_profile import load_global_profile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/tempo_go_c9_candidate_o_route_liveness_v1"
PROFILE = OUTPUT / "real_tempo_go_c9_candidate_o_route_liveness_profile_v1.json"
BASE = OUTPUT / "tempo_go_c8_candidate_o_route_liveness_contract.json"
POPULATION = OUTPUT / "tempo_go_c9_candidate_o_route_liveness_population_contract.json"
LAUNCHER = ROOT / "eval/sota_4node/run_tempo_go_c9_candidate_o_route_liveness_in_allocation.sh"
ATTACH = ROOT / "eval/sota_4node/attach_tempo_go_c9_candidate_o_route_liveness_to_allocation.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_o_is_route_scoped_and_keeps_m_joint_controls() -> None:
    loaded = load_global_profile(PROFILE)
    config = loaded.orchestrator_config()
    assert config.telemetry_failure_quarantine_mode == "deny_until_probe"
    assert config.telemetry_failure_quarantine_scope == "route"
    assert config.route_failure_quarantine_mode == "deny_until_probe"
    assert config.business_clean_pair_pressure_fraction == 0.5
    assert config.decoder_business_admission_mode == "priority_drain_v1"
    assert config.shared_fabric_control_mode == "global_budget_v3"
    assert config.mesh_control_mode == "receiver_credit_pxd_v1"


def test_candidate_o_completed_contract_is_immutable_and_historical() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    population = json.loads(POPULATION.read_text(encoding="utf-8"))
    assert base["joint_control"]["global_profile"]["sha256"] == sha256(PROFILE)
    assert population["system_under_test"]["base_contract_sha256"] == sha256(BASE)
    assert population["claim_boundary"]["performance_claim_allowed"] is False
    assert population["execution"]["one_campaign_no_retry"] is True
    assert [item["arm"] for item in population["execution"]["order"]] == [
        "fixed_local_d0",
        "fixed_local_d1",
        "fixed_remote_p0d1",
        "fixed_remote_p1d0",
        "predictor",
        "queue_gpu",
        "full_c7_managed_background",
    ]
    assert sha256(POPULATION) == (
        "9936a4a980d23250ce6604494d9fa545da189de8508b931d8ffb1952c35a8cc5"
    )
    mismatches = {
        relative
        for relative, expected in population["provenance"]["source_inventory"].items()
        if sha256(ROOT / relative) != expected
    }
    # The native campaign completed against the pinned analyzer.  Its raw
    # contract must not be rewritten after the post-hoc business-terminal
    # accounting fix; future campaigns require a newly built contract.
    assert mismatches == {
        "eval/sota_4node/analyze_tempo_go_c9_causal_burst_discovery.py"
    }


def test_candidate_o_launcher_is_pinned_and_never_submits() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert sha256(POPULATION) in text
    assert "TEMPO_GO_C9_CAUSAL_BURST_APPROVED" in text
    for forbidden in ("salloc", "sbatch", "scancel"):
        assert forbidden not in text


def test_candidate_o_attach_leaves_gpus_for_overlapping_native_children() -> None:
    text = ATTACH.read_text(encoding="utf-8")
    assert '--network=no_vni' in text
    assert '--gpus=0 --gres=none' in text
    assert '--cpus-per-task=128' in text
    assert 'ACTIVE_STEPS' in text
    assert 'QOS=gpu_interactive' in text
    assert 'Network=job_vni' in text
    assert '--wait=' not in text
    assert '--kill-on-bad-exit' not in text
    assert '--time=' not in text
    for forbidden in (
        '--gpus-per-task',
        '--gpus-per-node',
        'salloc',
        'sbatch',
        'scancel',
    ):
        assert forbidden not in text
