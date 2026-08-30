from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/tempo_go_c9_candidate_p_bounded_observer_v1"
CONTRACT = OUTPUT / "tempo_go_c9_candidate_p_bounded_observer_contract.json"
PARENT = ROOT / (
    "results/tempo_go_c9_candidate_o_route_liveness_v1/"
    "tempo_go_c9_candidate_o_route_liveness_population_contract.json"
)
LAUNCHER = ROOT / (
    "eval/sota_4node/run_tempo_go_c9_candidate_p_bounded_observer_in_allocation.sh"
)
ATTACH = ROOT / (
    "eval/sota_4node/attach_tempo_go_c9_candidate_p_bounded_observer_to_allocation.sh"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_p_changes_load_lifecycle_not_candidate_o_policy() -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    assert value["candidate"]["policy_delta_from_candidate_o"] == "none"
    assert value["system_under_test"]["base_contract"] == (
        parent["system_under_test"]["base_contract"]
    )
    assert value["system_under_test"]["base_contract_sha256"] == (
        parent["system_under_test"]["base_contract_sha256"]
    )
    assert value["burst"]["cojob_pair_count"] == 2
    assert value["burst"]["requests_per_source"] == 1
    assert value["burst"]["kv_mib_per_request"] == 8
    assert value["burst"]["token_iters"] == 256
    assert value["burst"]["block_delay_s"] == 0.25
    assert value["burst"]["minimum_active_duration_s"] == 600
    assert value["burst"]["maximum_blocks"] == 2048
    assert value["gates"]["minimum_full_supported_observer_fraction"] == 1.0
    assert value["claim_boundary"]["performance_claim_allowed"] is False
    assert value["execution"]["one_campaign_no_retry"] is True
    assert len(value["execution"]["order"]) == 7


def test_candidate_p_contract_is_current_source_bound() -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for relative, expected in value["source_inventory"].items():
        assert sha256(ROOT / relative) == expected
    assert value["source_inventory"] == value["provenance"]["source_inventory"]


def test_candidate_p_launcher_and_attach_are_pinned_and_safe() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    attach = ATTACH.read_text(encoding="utf-8")
    assert sha256(CONTRACT) in launcher
    assert "TEMPO_GO_C9_CAUSAL_BURST_APPROVED" in launcher
    for forbidden in ("salloc", "sbatch", "scancel"):
        assert forbidden not in launcher
    assert "--gpus=0 --gres=none" in attach
    assert "--network=no_vni" in attach
    assert "ACTIVE_STEPS" in attach
    assert "QOS=gpu_interactive" in attach
    assert "Network=job_vni" in attach
    assert "--wait=" not in attach
    assert "--kill-on-bad-exit" not in attach
    assert "--time=" not in attach
