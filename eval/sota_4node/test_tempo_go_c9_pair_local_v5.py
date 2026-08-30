from __future__ import annotations

import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_go_c9_causal_burst_discovery as analyzer
from tempo.pd_global_profile import load_global_profile


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_contract_v5.json"
BASE = ROOT / "eval/sota_4node/tempo_go_c9_pair_local_base_contract_v5.json"
PROFILE = ROOT / (
    "results/tempo_go_c9_pair_local_campaign_v5/"
    "real_tempo_go_c9_pair_local_profile_v5.json"
)
LAUNCHER = ROOT / (
    "eval/sota_4node/"
    "run_tempo_go_c9_causal_burst_discovery_in_allocation.sh"
)


def test_v5_is_bootstrap_observer_gated_and_source_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    base = json.loads(BASE.read_text(encoding="utf-8"))
    assert contract["claim_boundary"]["discovery_only"] is True
    assert contract["claim_boundary"]["performance_claim_allowed"] is False
    assert contract["mechanism"]["observer_gate"] == (
        "active_correct_sequence_at_least_one_before_victim_release"
    )
    assert contract["mechanism"]["bootstrap_policy"] == (
        "cojob_runs_before_victim_start_file; victim_release_waits_for_active_snapshot"
    )
    assert contract["system_under_test"]["base_contract_sha256"] == (
        analyzer._sha256(BASE)
    )
    assert base["joint_control"]["global_profile"]["path"] == str(
        PROFILE.relative_to(ROOT)
    )
    assert load_global_profile(PROFILE).fingerprint_sha256 == base[
        "joint_control"
    ]["global_profile"]["fingerprint_sha256"]
    for relative, expected in base["source_inventory"].items():
        assert analyzer._sha256(ROOT / relative) == expected, relative
    for relative, expected in contract["source_inventory"].items():
        assert analyzer._sha256(ROOT / relative) == expected, relative


def test_v5_launcher_breaks_start_file_cycle() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    start = text.index('TEMPO_GO_CROSS_LAYER_COMPONENT_APPROVED=YES')
    end = text.index('  ready=0', start)
    launch_window = text[start:end]
    assert 'TEMPO_GO_CROSS_LAYER_START_FILE="${start_file}"' not in launch_window
    assert 'observer_ready=0' in text
    assert 'C9 causal burst observer readiness failed' in text
