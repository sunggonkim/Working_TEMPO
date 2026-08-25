from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from eval.sota_4node import analyze_tempo_go_c8_dual_regime as analyzer
from eval.sota_4node import run_tempo_go_c8_dual_regime_client as client
from eval.sota_4node import tempo_pd_elastic_frontend as frontend
from eval.sota_4node.tempo_pd_frontend_v1 import pair_index


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c8_dual_regime_contract_v45.json"
CONTRACT_SHA256 = (
    "1521d855b8dbddde58afff0a92050969123c0218004288b33b756498f88ca260")


def _section() -> tuple[dict[str, object], dict[str, object]]:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    section = value["joint_control"]
    spec = next(
        row for row in section["blocks"]
        if row.get("pressure_regime") == analyzer.REMOTE_REGIME
    )
    return section, spec


def test_current_v45_contract_matches_source_revision() -> None:
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == CONTRACT_SHA256
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    inventory = value["source_inventory"]
    assert len(inventory) == 33
    for relative, expected in inventory.items():
        source = ROOT / relative
        assert source.is_file(), relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, relative


def test_remote_regime_has_exact_dual_decoder_population(monkeypatch) -> None:
    monkeypatch.setenv(client.ARM_ENV, "full_c7_managed_background")
    section, spec = _section()
    schedule, identities = client._materialize_remote_schedule(
        spec=spec, section=section)
    roles = {}
    for value in identities.values():
        roles[value["role"]] = roles.get(value["role"], 0) + 1
    assert roles == {"victim": 30, "local_aggressor": 1344}
    targets = {0: 0, 1: 0}
    for value in identities.values():
        if value["role"] == "local_aggressor":
            targets[value["target_decoder_index"]] += 1
            assert value["managed_by_tempo_go"] is False
            assert value["expected_cache"] == "miss"
    assert targets == {0: 672, 1: 672}
    victims = [
        value for value in identities.values() if value["role"] == "victim"
    ]
    assert all(value["expected_cache"] == "p_only" for value in victims)
    assert {value["p_only_owner"] for value in victims} == {0, 1}
    assert len(schedule) == 1374


def test_fixed_remote_identity_preserves_source_and_destination(monkeypatch) -> None:
    monkeypatch.setenv(client.ARM_ENV, "fixed_remote_p0d1")
    request_id, value = client._victim_identity(block="remote", ordinal=7)
    assert "cache-p-only-measured" in request_id
    assert request_id.endswith("-0")
    assert value == {
        "expected_route": client.REMOTE_ROUTE,
        "expected_source": 0,
        "expected_decoder": 1,
        "p_only_owner": 0,
    }


def test_physical_preseed_ids_support_replication_and_owner_pin(monkeypatch) -> None:
    monkeypatch.setenv(client.ARM_ENV, "full_c7_managed_background")
    for owner in (0, 1):
        request_id = client._physical_preseed_request_id(
            name="05_p_only_dual_decoder_hot", pool_index=3, owner=owner,
        )
        seed_id = request_id.replace("-warm-", "-warm-seed-o128-", 1)
        assert frontend.c4_physical_pair_pin(seed_id, "tempo") is True
        assert pair_index(seed_id, 2) == owner
        shadow_id = frontend.affinity_shadow_request_id(seed_id, 1 - owner)
        assert f"-affinity-shadow-p{1 - owner}-item-" in shadow_id
        assert pair_index(shadow_id, 2) == owner

    monkeypatch.setenv(client.ARM_ENV, "queue_gpu")
    for owner in (0, 1):
        request_id = client._physical_preseed_request_id(
            name="05_p_only_dual_decoder_hot", pool_index=3, owner=owner,
        )
        seed_id = request_id.replace("-warm-", "-warm-seed-o128-", 1)
        assert frontend.c4_physical_pair_pin(seed_id, "queue_gpu") is True
        assert pair_index(seed_id, 2) == owner


def test_physical_preseed_uses_fixed_arm_cache_namespace(monkeypatch) -> None:
    monkeypatch.setenv(client.ARM_ENV, "fixed_remote_p0d1")
    request_id = client._physical_preseed_request_id(
        name="05_p_only_dual_decoder_hot", pool_index=3, owner=0)
    assert request_id.startswith("epd-remote-interactive-")
    assert request_id.endswith("-owner-0-item-000000")


def test_contract_freezes_remote_activation_without_oracle_input() -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    section = value["joint_control"]
    remote = section["remote_activation"]
    assert value["schema"] == analyzer.CONTRACT_SCHEMA
    assert remote["local_rate_per_decoder"] == 22.4
    assert remote["total_local_rate_per_s"] == 44.8
    assert remote["physical_preseed_outside_measurement"] is True
    assert remote["exact_source_hit_required"] is True
    assert remote["controller_does_not_receive_regime_label"] is True
    assert remote["vllm_scheduling_policy"] == "priority"
    assert remote["managed_remote_priority"] == -2
    assert remote["baseline_request_priority"] == 0
    assert remote["priority_service_lane_mode"] == (
        "vllm_priority_remote_cache_v1")
    assert remote["priority_service_lane_capacity_per_decoder"] == 8
    assert remote["decoder_business_admission_mode"] == "priority_drain_v1"
    assert remote["decoder_background_concurrency_limits"] == [8, 8]
    assert remote["decoder_background_max_wait_ns"] == 60_000_000_000
    assert remote["decoder_background_requests_are_delayed_not_dropped"] is True
    assert remote["mesh_near_tie_source_balance_mode"] == (
        "telemetry_uncertainty_virtual_service_v1")
    assert remote["mesh_near_tie_source_balance_uncertainty_fraction"] == 1.0
    assert remote["mesh_near_tie_source_balance_is_not_a_route_quota"] is True
    assert remote["telemetry_freshness_ns"] == 500_000_000
    assert remote["telemetry_refresh_timeout_ns"] == 400_000_000
    assert remote["telemetry_maximum_collection_span_ns"] == 400_000_000
    assert remote["telemetry_per_fetch_timeout_ns"] == 200_000_000
    assert section["remote_activation_gates"] == {
        "minimum_full_remote_fraction": 0.5,
        "minimum_cross_pair_remote_fraction": 0.1,
        "best_fixed_slo_retention_fraction": 0.95,
        "best_fixed_p99_ratio_ceiling": 1.1,
    }


def test_global_edge_uses_replicated_prefill_not_destination_pair() -> None:
    decision = {
        "frontend_pair_index": 1,
        "frontend_pair_affinity_owner_indices": [0, 1],
        "local_decoder_index": 0,
        "remote_decoder_index": 1,
        "tempo_go_global_commit_pair_index": 1,
        "tempo_go_global_commit_prefill_index": 0,
        "tempo_go_global_commit_decoder_index": 1,
        "tempo_go_global_commit_edge_id": "remote:p0->d1",
    }
    assert client._validated_global_edge(
        decision, route=client.REMOTE_ROUTE) == (0, 1)


def test_c7_shared_analyzer_uses_explicit_cross_mesh_edge() -> None:
    decision = {
        "route": client.REMOTE_ROUTE,
        "frontend_pair_index": 0,
        "remote_decoder_index": 0,
        "tempo_go_global_commit_prefill_index": 1,
        "tempo_go_global_commit_decoder_index": 0,
        "tempo_go_global_commit_edge_id": "remote:p1->d0",
        "tempo_go_global_commit_route": client.REMOTE_ROUTE,
    }
    assert analyzer.c7._edge(decision) == "remote:p1->d0"


def test_decoder_business_gate_drains_background_for_protected_work() -> None:
    async def scenario() -> None:
        gate = frontend.DecoderBusinessAdmissionGate(
            background_limits=(2, 2),
            background_max_wait_ns=1_000_000_000,
            protected_tenants=frozenset({"interactive", "latency"}),
        )
        first = await gate.acquire(
            request_id="b0", pair_index=0, tenant_id="background",
            globally_committed=False)
        second = await gate.acquire(
            request_id="b1", pair_index=0, tenant_id="background",
            globally_committed=False)
        assert first["admission_class"] == "background"
        assert second["background_active_after"] == 2

        pending = asyncio.create_task(gate.acquire(
            request_id="b2", pair_index=0, tenant_id="background",
            globally_committed=False))
        await asyncio.sleep(0)
        assert not pending.done()

        protected = await gate.acquire(
            request_id="p0", pair_index=0, tenant_id="interactive",
            globally_committed=True)
        assert protected["admission_class"] == "protected"
        await gate.release("b0")
        await gate.release("b1")
        await asyncio.sleep(0)
        assert not pending.done()
        released = await gate.release("p0")
        assert released["status"] == "released"
        third = await pending
        assert third["starvation_escape"] is False
        await gate.release("b2")
        snapshot = await gate.snapshot()
        assert snapshot["foreground_active"] == [0, 0]
        assert snapshot["background_active"] == [0, 0]
        assert snapshot["foreground_admitted"] == 1
        assert snapshot["background_admitted"] == 3

    asyncio.run(scenario())


def test_node_environment_binds_identical_priority_scheduler(monkeypatch) -> None:
    monkeypatch.setenv(client.ARM_ENV, "full_c7_managed_background")
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    monkeypatch.setattr(client.c7, "configure_node_environment", lambda **_: None)
    # configure_node_environment intentionally mutates the process environment
    # because the native launcher consumes it.  Snapshot the whole mapping so
    # this unit test cannot leak C8 priority settings into later router tests.
    with patch.dict(client.os.environ, {}, clear=False):
        client.configure_node_environment(
            repo_root=ROOT,
            qualification=value,
            hosts=["n0", "n1", "n2", "n3"],
            port_slot=2000,
            elastic_profile=ROOT / value["joint_control"]["profile"]["path"],
        )
        assert client.os.environ[
            "TEMPO_VLLM_SCHEDULING_POLICY"] == "priority"
        assert client.os.environ[
            "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY"] == "-2"
        assert client.os.environ[
            "TEMPO_PD_REMOTE_CATCHUP_PRIORITY"] == "0"


def test_regime_effect_uses_offered_slo_and_tail() -> None:
    full = {
        "remote_favorable": {
            "slo_good_victims": 29,
            "victim": {"e2e_ms": {"p50": 2.0, "p99": 3.0}},
        },
    }
    baseline = {
        "remote_favorable": {
            "slo_good_victims": 20,
            "victim": {"e2e_ms": {"p50": 4.0, "p99": 6.0}},
        },
    }
    effect = analyzer._regime_effect(full, baseline, "remote_favorable")
    assert effect["slo_good_ratio"] == 1.45
    assert effect["e2e_p50_reduction_fraction"] == 0.5
    assert effect["e2e_p99_reduction_fraction"] == 0.5
