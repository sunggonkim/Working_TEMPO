from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from eval.sota_4node import analyze_tempo_go_c9_causal_burst_discovery as analyzer
from eval.sota_4node import run_tempo_go_c8_dual_regime_client as c8_client


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c9_causal_burst_discovery_contract.json"
LAUNCHER = ROOT / "eval/sota_4node/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh"
COJOB_LAUNCHER = ROOT / "eval/sota_4node/run_lmcache_nixl_contention_2node_in_allocation.sh"
ANALYZER = ROOT / "eval/sota_4node/analyze_tempo_go_c9_causal_burst_discovery.py"
BUSINESS_BASE = (
    ROOT / "eval/sota_4node/tempo_go_c9_business_lane_base_contract_v14.json")
BUSINESS_CONTRACT = (
    ROOT / "eval/sota_4node/tempo_go_c9_business_lane_followup_contract_v11.json")


def test_contract_is_abba_discovery_with_frozen_overload_gate() -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert value["claim_boundary"] == {
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "discovery_only": True,
        "reason": (
            "The overload geometry and timeout-as-outcome rule were selected "
            "from a prior partial pilot; this campaign is frozen before "
            "allocation but remains discovery and cannot replace the fresh "
            "C9 independent result"
        ),
    }
    assert [item["arm"] for item in value["execution"]["order"]] == [
        "app_global_only",
        "full_c7_managed_background",
        "full_c7_managed_background",
        "app_global_only",
    ]
    assert value["burst"]["requests_per_source"] == 4
    assert value["burst"]["kv_mib_per_request"] == 128
    assert value["burst"]["nixl_transfer_timeout_s"] == 60


def test_launcher_preserves_exact_timeout_and_continues_without_retry() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "cojob_outcome=overload_timeout" in text
    assert "official LMCache/NIXL batched_write exceeded" in text
    # Perlmutter does not provide ripgrep by default.  The overload receipt
    # check must use a baseline utility available in the native environment;
    # otherwise a valid co-job timeout aborts the campaign before its block
    # receipt is written.
    assert "grep -Eq 'official LMCache/NIXL batched_write exceeded [0-9.]+s'" in text
    assert "rg -q" not in text
    assert "measured_arm_retried:false" in text
    assert '[[ "${inference_rc}" -ne 0 || ! -s' in text
    assert "block_failure_receipt.json" in text
    assert "failed_attempt.json" in text
    assert 'kill -TERM "${cojob_pid}"' in text
    assert 'TEMPO_GO_C9_COJOB_HOSTS_CSV="${pair_hosts}"' in text
    assert 'cojob_pair_count:2' in text
    assert "nvidia-smi --query-compute-apps=pid" in text
    assert '[[ "${inference_rc}" -eq 0 && "${cojob_rc}" -eq 0 ]]' not in text
    assert 'scancel "${step_id}"' in text
    assert 'scancel "${SLURM_JOB_ID}"' not in text
    assert "--jobid=\"${SLURM_JOB_ID}\" --overlap \\" in text
    assert "--gpus-per-node=4" in text
    assert "--network=job_vni" in text
    assert 'TEMPO_GO_SRUN_NETWORK_MODE="${TEMPO_GO_SRUN_NETWORK_MODE:-job_vni}"' in COJOB_LAUNCHER.read_text(encoding="utf-8")
    assert "--overlap --exact" not in text
    assert "--wait=10" not in text


def test_nested_cjob_inherits_job_vni_network_mode() -> None:
    text = COJOB_LAUNCHER.read_text(encoding="utf-8")
    assert 'TEMPO_GO_SRUN_NETWORK_MODE="${TEMPO_GO_SRUN_NETWORK_MODE:-job_vni}"' in text
    assert '"${TEMPO_GO_SRUN_NETWORK_MODE}" == "job_vni"' in text
    assert '"--network=${TEMPO_GO_SRUN_NETWORK_MODE}"' in text


def test_analyzer_accepts_only_receipted_overload_timeout() -> None:
    text = ANALYZER.read_text(encoding="utf-8")
    assert '{"complete", "overload_timeout"}' in text
    assert "cojob_step_failed" in text
    assert "official LMCache/NIXL batched_write exceeded" in text
    assert "cojob_outcome_valid" in text
    assert 'raw.get("c8_dual_regime_contract")' in text
    assert 'raw.get("c7_joint_control_contract")' in text


def test_signal_contribution_parser_handles_wire_shapes() -> None:
    assert analyzer._contribution_names({
        "signal_contributions": [
            ["nccl_collective_p99_ms", 0.5],
            {"name": "lmcache_transfer_p99_ms", "value": 1.0},
        ]
    }) == {"nccl_collective_p99_ms", "lmcache_transfer_p99_ms"}


def test_business_terminal_outcome_does_not_count_valid_503_as_complete() -> None:
    assert analyzer._business_terminal_outcome({
        "valid": True,
        "http_status": 200,
        "done_seen": True,
        "transport_error": None,
        "output_token_values": [0, 1],
    }, expected_output_tokens=2) == "completed"
    assert analyzer._business_terminal_outcome({
        "valid": True,
        "http_status": 503,
        "done_seen": False,
        "terminal_kind": "service_lane_failure",
        "transport_error": "HTTPError: 503",
        "output_token_values": [],
    }, expected_output_tokens=2) == "failure"
    assert analyzer._business_terminal_outcome({
        "valid": True,
        "http_status": 503,
        "done_seen": False,
        "terminal_kind": "global_reject",
        "transport_error": "HTTPError: 503",
        "output_token_values": [],
    }, expected_output_tokens=2) == "global_reject"


def test_shared_fabric_reject_provenance_is_supported_and_actuated() -> None:
    decision = {
        "schema": "tempo-go-global-orchestrator-v1",
        "kind": "reject",
        "telemetry_provenance": {
            "-1": {
                "groups": {
                    "epoch|topology|communicator": {
                        "contributions": [
                            {
                                "name": "shared.requests.nccl_collective_p99_ms",
                                "pressure": 0.05,
                            },
                            {
                                "name": "shared.kv_bytes.lmcache_transfer_p99_ms",
                                "pressure": 1.0,
                            },
                        ],
                        "dispatch_stagger_us": 2000,
                        "suppress_pair_activation": True,
                        "limited": False,
                    }
                }
            }
        },
    }
    assert analyzer._provenance_cross_layer_names(decision) == {
        "nccl_collective_p99_ms",
        "lmcache_transfer_p99_ms",
    }
    assert analyzer._shared_provenance_controls(decision) == {
        "dispatch_stagger": True,
        "pair_activation_suppressed": True,
        "limited": False,
    }


def test_historical_business_lane_contract_is_stale_and_not_reusable() -> None:
    base = json.loads(BUSINESS_BASE.read_text(encoding="utf-8"))
    contract = json.loads(BUSINESS_CONTRACT.read_text(encoding="utf-8"))
    assert contract["claim_boundary"]["discovery_only"] is True
    assert contract["claim_boundary"]["performance_claim_allowed"] is False
    assert contract["system_under_test"]["node_entry"] == (
        "eval/sota_4node/c9_gate_node_entry.sh")
    assert contract["system_under_test"]["base_contract_sha256"] == (
        analyzer._sha256(BUSINESS_BASE))
    assert contract["gates"]["observer_support_scope"] == (
        "remote_favorable_victim_global_decisions")
    assert contract["gates"][
        "minimum_remote_background_completion_fraction"] == 0.99
    assert base["candidate"]["id"] == (
        "tempo-go-c9-dual-route-business-lane-v2")
    assert base["candidate"][
        "failed_preflight_was_not_a_performance_result"] is True
    assert base["joint_control"]["remote_activation"][
        "priority_service_lane_mode"] == (
            "vllm_priority_business_dual_route_v2")
    assert len(base["source_inventory"]) == 44
    mismatches = set()
    for relative, expected in base["source_inventory"].items():
        source = ROOT / relative
        assert source.is_file(), relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            mismatches.add(relative)
    # This v14 contract predates the current source-bound C8/C9 contract
    # lineage.  It must remain visibly stale; treating it as a valid frozen
    # contract was the source-drift mistake that invalidated the prior C9
    # attempt.  New campaigns must use a freshly generated current-source
    # contract instead.
    assert mismatches == {
        "eval/sota_4node/analyze_tempo_go_c9_causal_burst_discovery.py",
        "eval/sota_4node/analyze_tempo_go_c8_dual_regime.py",
        "eval/sota_4node/analyze_tempo_go_c7_joint_control.py",
        "eval/sota_4node/build_tempo_go_c8_priority_service_lane_profile.py",
        "eval/sota_4node/require_perlmutter_4node_4h_interactive.sh",
        "eval/sota_4node/run_tempo_go_c8_independent_validation_in_allocation.sh",
        "eval/sota_4node/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh",
        "eval/sota_4node/tempo_pd_elastic_frontend.py",
        "eval/sota_4node/tempo_pd_elastic_router.py",
        "eval/sota_4node/vllm_lmcache_tempo_pd_perf_node_v1.py",
        "eval/sota_4node/run_tempo_pd_stream_metrics_v1.py",
        "tempo/pd_global_orchestrator.py",
        "tempo/pd_global_profile.py",
        "tempo/pd_global_telemetry.py",
    }


def test_business_dual_route_mode_passes_native_node_binding(monkeypatch) -> None:
    base = json.loads(BUSINESS_BASE.read_text(encoding="utf-8"))
    profile = ROOT / base["joint_control"]["global_profile"]["path"]
    monkeypatch.setenv(c8_client.ARM_ENV, "full_c7_managed_background")
    monkeypatch.setattr(
        c8_client.c7, "configure_node_environment", lambda **_: None)
    with patch.dict(c8_client.os.environ, {}, clear=False):
        c8_client.configure_node_environment(
            repo_root=ROOT,
            qualification=base,
            hosts=["n0", "n1", "n2", "n3"],
            port_slot=2600,
            elastic_profile=profile,
        )
        assert c8_client.os.environ[
            "TEMPO_VLLM_SCHEDULING_POLICY"] == "priority"
        assert c8_client.os.environ[
            "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY"] == "-2"
