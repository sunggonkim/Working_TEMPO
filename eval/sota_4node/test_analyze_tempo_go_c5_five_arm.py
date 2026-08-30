from __future__ import annotations

import hashlib
import json

import pytest

from eval.sota_4node import analyze_tempo_go_c5_five_arm as analyzer


def _request(request_id: str, *, valid: bool, rejected: bool = False) -> dict:
    value = {
        "request_id": request_id,
        "valid": valid,
        "scheduled_dispatch_offset_ns": 0,
        "dispatch_offset_ns": 1_000_000,
        "token_arrival_offsets_ns": [2_000_000, 3_000_000],
        "stream_end_offset_ns": 4_000_000,
        "output_token_values": ["A", "B"] if valid else [],
        "router": {
            "route": "decoder_local_chunked_prefill",
        } if valid and not rejected else None,
    }
    if rejected:
        value.update({
            "terminal_kind": "global_reject",
            "http_status": 503,
        })
        value["token_arrival_offsets_ns"] = []
    return value


def test_service_metrics_separate_rejects_from_starvation_and_report_business_metrics():
    latency_id = "epd-tempo-latency-c0_cool-cache-miss-measured-r00-foreground-000000"
    background_id = "epd-tempo-background-c0_cool-cache-miss-measured-r00-background-000001"
    requests = [
        _request(latency_id, valid=True),
        _request(background_id, valid=True, rejected=True),
    ]
    decisions = [
        {
            "request_id": latency_id,
            "phase": "complete",
            "error": None,
            "route": "decoder_local_chunked_prefill",
            "frontend_pair_index": 0,
            "endpoint_request_local_e2e_prior_ms": 100.0,
            "endpoint_request_remote_e2e_prior_ms": 200.0,
            "frontend_tempo_go_admission_wait_ns": 2_000_000,
        },
        {
            "request_id": background_id,
            "phase": "rejected",
            "global_decision_kind": "reject",
            "tempo_go_rejected": True,
            "frontend_tempo_go_admission_wait_ns": 8_000_000,
        },
    ]
    manifest = {
        "tenant_contract": {
            "latency": {
                "weight": 4.0,
                "ttft_slo_ms": 1000.0,
                "tpot_slo_ms": 100.0,
                "e2e_slo_ms": 4000.0,
                "minimum_service_fraction": 0.15,
            },
            "background": {
                "weight": 0.5,
                "ttft_slo_ms": 5000.0,
                "tpot_slo_ms": 400.0,
                "e2e_slo_ms": 30000.0,
                "minimum_service_fraction": 0.05,
            },
        },
    }
    value = analyzer._service_metrics(
        requests,
        decisions,
        manifest=manifest,
        client_window_ns=1_000_000_000,
    )
    assert value["global"]["completed_count"] == 1
    assert value["global"]["rejected_count"] == 1
    assert value["global"]["output_token_goodput_per_s"] == 2.0
    assert abs(value["global"]["global_admission_wait_ms"]["p99"] - 8.0) < 0.1
    assert value["by_tenant"]["latency"]["slo_good_count"] == 1
    assert value["by_tenant"]["background"]["rejection_only_no_completion"]
    assert not value["by_tenant"]["background"]["starvation"]
    assert value["selected_route_counterfactual"][
        "measured_same_request_counterfactual"] is False


def test_service_metrics_counts_global_scheduler_provenance_separately():
    request_id = (
        "epd-tempo-background-c0_cool-cache-miss-measured-r00-"
        "foreground-000000")
    requests = [_request(request_id, valid=True)]
    scheduler = {
        "schema": "tempo-go-vllm-scheduler-snapshot-v1",
        "source": "router_local_vllm_prometheus_observe_only",
        "running_requests": 3,
        "waiting_requests": 2,
        "kv_cache_usage_fraction": 0.25,
    }
    cross_layer = {
        "schema": "tempo-go-cross-layer-envelope-v1",
        "communicator_id": "c5-test-communicator",
        "source_epoch": "slurm-test-cross-layer",
        "topology_fingerprint_sha256": "a" * 64,
        "signals": [{"name": "nccl_collective_p99_ms", "support": "supported"}],
    }
    decisions = [{
        "request_id": request_id,
        "phase": "complete",
        "error": None,
        "route": "decoder_local_chunked_prefill",
        "frontend_pair_index": 0,
        "frontend_tempo_go_decision": {
            "telemetry_provenance": {
                "0": {"scheduler": scheduler, "cross_layer": cross_layer},
                "1": {
                    "scheduler": dict(scheduler),
                    "cross_layer": dict(cross_layer),
                },
            },
        },
    }]
    value = analyzer._service_metrics(
        requests,
        decisions,
        manifest={"tenant_contract": {}},
        client_window_ns=1_000_000_000,
    )
    overhead = value["telemetry_overhead"]
    assert overhead["scheduler_observation_count"] == 2
    assert overhead["scheduler_observation_payload_count"] == 1
    assert overhead["scheduler_observation_invalid_count"] == 0
    assert overhead["scheduler_pair_observation_counts"] == {"0": 1, "1": 1}
    assert overhead["scheduler_source_counts"] == {
        "router_local_vllm_prometheus_observe_only": 2,
    }
    assert overhead["cross_layer_observer_provenance"] == {
        "payload_count": 1,
        "observation_count": 2,
        "invalid_count": 0,
        "pair_observation_counts": {"0": 1, "1": 1},
        "schema_counts": {"tempo-go-cross-layer-envelope-v1": 2},
        "source_epoch_counts": {"slurm-test-cross-layer": 2},
        "communicator_counts": {"c5-test-communicator": 2},
    }


def test_service_metrics_ignores_aggregate_provenance_for_global_reject():
    request_id = (
        "epd-tempo-background-c0_cool-cache-miss-measured-r00-"
        "foreground-000004")
    requests = [_request(request_id, valid=False, rejected=True)]
    decisions = [{
        "request_id": request_id,
        "phase": "rejected",
        "global_decision_kind": "reject",
        "tempo_go_rejected": True,
        "frontend_tempo_go_decision": {
            "telemetry_provenance": {
                "-1": {
                    "schema": "tempo-go-shared-fabric-provenance-v1",
                    "groups": {},
                },
            },
        },
    }]
    value = analyzer._service_metrics(
        requests,
        decisions,
        manifest={"tenant_contract": {}},
        client_window_ns=1_000_000_000,
    )
    overhead = value["telemetry_overhead"]
    assert overhead["scheduler_observation_payload_count"] == 1
    assert overhead["scheduler_observation_count"] == 0
    assert overhead["scheduler_observation_invalid_count"] == 0
    assert overhead["cross_layer_observer_provenance"]["payload_count"] == 1
    assert overhead["cross_layer_observer_provenance"]["observation_count"] == 0
    assert overhead["cross_layer_observer_provenance"]["invalid_count"] == 0


def test_service_metrics_counts_explicit_global_failure_without_calling_it_starvation():
    request_id = (
        "epd-tempo-background-c0_cool-cache-miss-measured-r00-"
        "foreground-000002")
    requests = [_request(request_id, valid=False)]
    decisions = [{
        "request_id": request_id,
        "phase": "failed",
        "error": "upstream_transport_error",
        "frontend_pair_index": 0,
        "frontend_tempo_go_failure": {
            "schema": "tempo-go-global-failure-v1",
            "request_id": request_id,
            "terminal_phase": "failed",
            "failure_kind": "upstream_transport_error",
            "quarantine_scope": "pair",
        },
    }]
    value = analyzer._service_metrics(
        requests,
        decisions,
        manifest={"tenant_contract": {}},
        client_window_ns=1_000_000_000,
    )
    assert value["global"]["failed_count"] == 1
    assert value["global"]["rejected_count"] == 0
    assert not value["global"]["starvation"]


def test_service_metrics_counts_service_lane_reservation_as_explicit_failure():
    request_id = (
        "epd-tempo-background-c0_cool-cache-miss-measured-r00-"
        "foreground-000003")
    requests = [_request(request_id, valid=False)]
    decisions = [{
        "request_id": request_id,
        "phase": "failed",
        "error": "endpoint service-lane reservation unavailable",
        "frontend_pair_index": 0,
        "frontend_tempo_go_reservation_failure": {
            "schema": "tempo-go-service-lane-reservation-v1",
            "request_id": request_id,
            "terminal_phase": "failed",
            "pair_index": 0,
            "failure_kind": "endpoint_service_lane_reservation_unavailable",
        },
    }]
    value = analyzer._service_metrics(
        requests,
        decisions,
        manifest={"tenant_contract": {}},
        client_window_ns=1_000_000_000,
    )
    assert value["global"]["failed_count"] == 1
    assert value["global"]["service_lane_reservation_failure_count"] == 1
    assert not value["global"]["starvation"]


def test_service_metrics_counts_valid_service_lane_receipt_as_explicit_failure():
    request_id = (
        "epd-tempo-background-c0_cool-cache-miss-measured-r00-"
        "foreground-000005")
    requests = [_request(request_id, valid=True)]
    requests[0]["terminal_kind"] = "service_lane_failure"
    decisions = [{
        "request_id": request_id,
        "phase": "failed",
        "error": "endpoint_bounded_queue_lease_timeout",
        "frontend_pair_index": 0,
        "frontend_tempo_go_reservation_failure": {
            "schema": "tempo-go-service-lane-reservation-v1",
            "request_id": request_id,
            "terminal_phase": "failed",
            "pair_index": 0,
            "failure_kind": "endpoint_bounded_queue_lease_timeout",
        },
    }]
    value = analyzer._service_metrics(
        requests,
        decisions,
        manifest={"tenant_contract": {}},
        client_window_ns=1_000_000_000,
    )
    assert value["global"]["failed_count"] == 1
    assert value["global"]["service_lane_reservation_failure_count"] == 1


def test_single_arm_receipt_analysis_does_not_require_five_arm_permutation(
    tmp_path, monkeypatch,
):
    (tmp_path / "arm_order.txt").write_text("tempo\n", encoding="utf-8")
    monkeypatch.setattr(
        analyzer,
        "_analyze_arm",
        lambda root, arm: {
            "arm": arm,
            "raw_validation": {
                "router_decisions_exact": True,
                "terminal_contract_valid": True,
            },
        },
    )
    value = analyzer.analyze(tmp_path)
    assert value["schema"] == "tempo-go-c5-native-single-arm-analysis-v1"
    assert value["gates"]["single_arm_receipt"] is True
    assert value["gates"]["performance_claim_allowed"] is False


def test_signal_failure_receipt_is_analyzable_and_keeps_signal_provenance(
    tmp_path,
):
    workload = tmp_path / "validation.jsonl"
    manifest = tmp_path / "tempo_go_workload_manifest.json"
    workload.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    receipt = {
        "schema": "tempo-go-c5-native-arm-signal-failure-v1",
        "arm": "tempo",
        "failure": "native_arm_step_signal",
        "signal": "TERM",
        "native_only": True,
        "node_count": 4,
        "gpu_count": 16,
        "transport": "LMCacheConnectorV1:UCX",
        "workload": str(workload.resolve()),
        "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
        "workload_manifest": str(manifest.resolve()),
        "workload_manifest_sha256": hashlib.sha256(
            manifest.read_bytes()).hexdigest(),
    }
    failure_path = tmp_path / "failure.json"
    failure_path.write_text(json.dumps(receipt), encoding="utf-8")

    value = analyzer._analyze_native_arm_failure(
        tmp_path, "tempo", failure_path)

    assert value["execution_failure"] == "native_arm_step_signal"
    assert value["raw_validation"]["failure_schema"] == (
        "tempo-go-c5-native-arm-signal-failure-v1")
    assert value["raw_validation"]["signal"] == "TERM"
    assert value["performance_claim_allowed"] is False


def test_raw_backed_native_failure_is_analyzed_without_claiming_performance(
    tmp_path,
):
    workload = tmp_path / "validation.jsonl"
    manifest = tmp_path / "tempo_go_workload_manifest.json"
    workload.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    workload_sha = hashlib.sha256(workload.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    complete_id = (
        "epd-tempo-latency-c0_cool-cache-miss-measured-r00-"
        "foreground-000000")
    rejected_id = (
        "epd-tempo-background-c0_cool-cache-miss-measured-r00-"
        "background-000001")
    failed_id = (
        "epd-tempo-batch-c0_cool-cache-miss-measured-r00-"
        "foreground-000002")
    requests = [
        _request(complete_id, valid=True),
        _request(rejected_id, valid=True, rejected=True),
        _request(failed_id, valid=False),
    ]
    decisions = [
        {
            "request_id": complete_id,
            "phase": "complete",
            "error": None,
            "route": "decoder_local_chunked_prefill",
        },
        {
            "request_id": rejected_id,
            "phase": "rejected",
            "global_decision_kind": "reject",
            "tempo_go_rejected": True,
        },
        {
            "request_id": failed_id,
            "phase": "failed",
            "error": "upstream_transport_error",
            "frontend_tempo_go_failure": {
                "schema": "tempo-go-global-failure-v1",
                "request_id": failed_id,
                "terminal_phase": "failed",
                "failure_kind": "upstream_transport_error",
                "quarantine_scope": "pair",
            },
        },
    ]
    arm_dir = tmp_path / "tempo"
    raw_dir = arm_dir / "tempo_go_c5_discovery"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "raw.json"
    raw_path.write_text(json.dumps({
        "schema": "tempo-go-c5-native-discovery-v1",
        "workload": {
            "explicit_path": str(workload.resolve()),
            "sha256": workload_sha,
        },
        "validation": {
            "router_decisions_exact": False,
            "terminal_contract_valid": False,
            "performance_claim_allowed": False,
        },
        "requests": requests,
        "router_decisions": decisions,
        "run": {"client_window_ns": 1_000_000_000},
    }), encoding="utf-8")
    failure = {
        "schema": "tempo-go-c5-native-arm-failure-v1",
        "arm": "tempo",
        "failure": "native_arm_process_failed",
        "exit_code": 143,
        "native_only": True,
        "node_count": 4,
        "gpu_count": 16,
        "transport": "LMCacheConnectorV1:UCX",
        "result_dir": str(arm_dir.resolve()),
        "workload": str(workload.resolve()),
        "workload_sha256": workload_sha,
        "workload_manifest": str(manifest.resolve()),
        "workload_manifest_sha256": manifest_sha,
    }
    (arm_dir / "failure.json").write_text(
        json.dumps(failure), encoding="utf-8")
    (tmp_path / "arm_order.txt").write_text("tempo\n", encoding="utf-8")

    value = analyzer.analyze(tmp_path)
    arm = value["arms"]["tempo"]
    assert arm["request_count"] == 3
    assert arm["valid_count"] == 2
    assert arm["global_failure_receipts"] == 1
    assert arm["execution_failure"] == "native_arm_process_failed"
    assert arm["failure_artifact"] == str((arm_dir / "failure.json").resolve())
    assert arm["raw_validation"]["router_decisions_exact"] is False
    assert arm["raw_validation"]["execution_failure_raw_backed"] is True
    assert arm["service_metrics"]["global"]["completed_count"] == 1
    assert arm["service_metrics"]["global"]["rejected_count"] == 1
    assert arm["service_metrics"]["global"]["failed_count"] == 1
    assert value["gates"]["performance_claim_allowed"] is False

    # TEMPO owns this terminal path: an upstream failure without its signed
    # controller receipt must fail closed instead of being misclassified as a
    # valid TEMPO terminal event.
    decisions[-1].pop("frontend_tempo_go_failure")
    raw_path.write_text(json.dumps({
        "schema": "tempo-go-c5-native-discovery-v1",
        "workload": {
            "explicit_path": str(workload.resolve()),
            "sha256": workload_sha,
        },
        "validation": {
            "router_decisions_exact": False,
            "terminal_contract_valid": False,
            "performance_claim_allowed": False,
        },
        "requests": requests,
        "router_decisions": decisions,
        "run": {"client_window_ns": 1_000_000_000},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks global or service-lane"):
        analyzer.analyze(tmp_path)


def test_raw_backed_fixed_baseline_failure_needs_no_tempo_receipt(
    tmp_path,
):
    """A fixed-arm HTTP/process failure remains evidence, not a TEMPO reject."""

    workload = tmp_path / "validation.jsonl"
    manifest = tmp_path / "tempo_go_workload_manifest.json"
    workload.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    workload_sha = hashlib.sha256(workload.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    request_id = (
        "epd-remote-latency-c0_cool-cache-miss-measured-r00-"
        "foreground-000000")
    request = _request(request_id, valid=False)
    decision = {
        "request_id": request_id,
        "phase": "failed",
        "error": "HTTPError: HTTP Error 502 Bad Gateway",
        "route": "official_lmcache_remote_prefill",
    }
    arm_dir = tmp_path / "remote"
    raw_dir = arm_dir / "tempo_go_c5_discovery"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "raw.json"
    raw_path.write_text(json.dumps({
        "schema": "tempo-go-c5-native-discovery-v1",
        "workload": {
            "explicit_path": str(workload.resolve()),
            "sha256": workload_sha,
        },
        "validation": {
            "router_decisions_exact": False,
            "terminal_contract_valid": False,
            "performance_claim_allowed": False,
        },
        "requests": [request],
        "router_decisions": [decision],
        "run": {"client_window_ns": 1_000_000_000},
    }), encoding="utf-8")
    (arm_dir / "failure.json").write_text(json.dumps({
        "schema": "tempo-go-c5-native-arm-failure-v1",
        "arm": "remote",
        "failure": "native_arm_process_failed",
        "exit_code": 143,
        "native_only": True,
        "node_count": 4,
        "gpu_count": 16,
        "transport": "LMCacheConnectorV1:UCX",
        "result_dir": str(arm_dir.resolve()),
        "workload": str(workload.resolve()),
        "workload_sha256": workload_sha,
        "workload_manifest": str(manifest.resolve()),
        "workload_manifest_sha256": manifest_sha,
    }), encoding="utf-8")
    (tmp_path / "arm_order.txt").write_text("remote\n", encoding="utf-8")

    value = analyzer.analyze(tmp_path)
    arm = value["arms"]["remote"]
    assert arm["execution_failure"] == "native_arm_process_failed"
    assert arm["router_execution_failure_receipts"] == 1
    assert arm["global_failure_receipts"] == 0
    assert arm["raw_validation"]["execution_failure_raw_backed"] is True
    assert value["gates"]["performance_claim_allowed"] is False
