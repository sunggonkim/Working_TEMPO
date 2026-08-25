from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from eval.sota_4node import analyze_tempo_go_c8_independent_validation as analyzer
from eval.sota_4node import run_tempo_go_c8_independent_validation_client as client
from eval.sota_4node import vllm_lmcache_pd_contention_node as contention_node
from eval.sota_4node import vllm_lmcache_tempo_go_c8_independent_validation_node as node


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "eval/sota_4node/tempo_go_c8_independent_validation_contract_v3.json"
)
CONTRACT_SHA256 = (
    "e2d07e8c50316620cee29a82ae06bbb4e3efd5e8c18c07347a34a4f532f07a76"
)
PARENT_SHA256 = (
    "1521d855b8dbddde58afff0a92050969123c0218004288b33b756498f88ca260"
)


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_independent_contract_is_source_bound_and_counterbalanced() -> None:
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == CONTRACT_SHA256
    value = _contract()
    heldout = value["independent_validation"]
    assert heldout["parent_discovery"]["contract_sha256"] == PARENT_SHA256
    assert heldout["one_shot_no_retry"] is True
    assert heldout["fresh_allocation_required"] is True
    assert [row["slurm_job_id"] for row in heldout["prior_failed_attempts"]] == [
        "57585888", "57586085"]
    assert all(row["performance_result"] is False
               for row in heldout["prior_failed_attempts"])
    arm_specs = value["joint_control"]["arms"]
    ports = heldout["runtime_port_schedule"]
    assert ports["maximum_port_slot"] == (
        ports["port_slot_base"]
        + ports["port_slot_stride_per_arm"] * (len(arm_specs) - 1)
    )
    assert ports["maximum_endpoint_probe_port"] < ports["exclusive_upper_bound"]
    for index in range(len(arm_specs)):
        slot = (
            ports["port_slot_base"]
            + index * ports["port_slot_stride_per_arm"]
        )
        assert contention_node._probe_port(slot) == 30_000 + slot
    assert value["claim_boundary"] == {
        "controller_performance_claim_allowed": True,
        "performance_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "purpose": (
            "Preregistered held-out execution; no claim is authorized before "
            "the fresh-allocation analyzer passes"
        ),
    }
    arms = [row["name"] for row in arm_specs]
    blocks = [row["name"] for row in value["joint_control"]["blocks"]]
    assert arms[0] == "full_c7_managed_background"
    assert arms[-1] == "fixed_local_d0"
    assert blocks[1] == "05_p_only_dual_decoder_hot"
    assert blocks == list(reversed(
        heldout["counterbalance"]["discovery_block_order"]))
    for relative, expected in value["source_inventory"].items():
        source = ROOT / relative
        assert source.is_file(), relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, relative


def test_heldout_arrival_jitter_is_deterministic_bounded_and_nonuniform() -> None:
    heldout = {
        "request_seed": 20260825,
        "arrival_jitter": {
            "algorithm": "sha256_centered_subspacing_v1",
            "maximum_spacing_fraction": 0.25,
        },
    }
    client._HELDOUT = heldout
    client._SCHEDULE_CONTEXT = "block-a"
    client._SCHEDULE_STREAM_INDEX = 0
    first = client._jittered_uniform_offsets(30_000.0, 7.8)
    client._SCHEDULE_STREAM_INDEX = 0
    second = client._jittered_uniform_offsets(30_000.0, 7.8)
    base = client._ORIGINAL_UNIFORM_OFFSETS(30_000.0, 7.8)
    assert first == second
    assert len(first) == len(base) == 234
    assert first != base
    assert all(0.0 < value < 30_000.0 for value in first)
    assert all(left < right for left, right in zip(first, first[1:]))
    spacing = 1000.0 / 7.8
    assert max(abs(left - right) for left, right in zip(first, base)) <= 0.25 * spacing


def test_heldout_p_only_prompt_uses_disjoint_marker_offset() -> None:
    client._HELDOUT = {
        "p_only_prompt_namespace": {
            "algorithm": "c8_marker_offset_v1",
            "base_marker": 240_000,
            "marker_offset": 8_192,
        },
    }
    tokenizer = object()
    with patch.object(
        client.frozen.c7.fixed, "_unique_prompt", return_value="heldout"
    ) as unique:
        assert client._p_only_prompt(
            tokenizer, (1, 2), sequence=1, owner=1,
            pool_index=3, pool_size=8,
        ) == "heldout"
    unique.assert_called_once_with(tokenizer, (1, 2), 248_235)


def test_signal_status_never_converts_unavailable_to_zero() -> None:
    decision = {
        "frontend_tempo_go_decision": {
            "telemetry_provenance": {
                "0": {
                    "cross_layer": {
                        "signals": [
                            {
                                "name": "nccl_collective_p99_ms",
                                "support": "not_collected",
                                "value": None,
                            },
                            {
                                "name": "lmcache_remote_kv_bytes_inflight",
                                "support": "supported",
                                "value": 0,
                            },
                        ],
                    },
                },
            },
        },
    }
    assert analyzer._signal_status(
        decision, "nccl_collective_p99_ms") == (False, True)
    assert analyzer._signal_status(
        decision, "lmcache_remote_kv_bytes_inflight") == (True, True)
    assert analyzer._signal_status(decision, "missing") == (False, False)


def test_jain_fairness_is_rate_normalized() -> None:
    assert analyzer._jain([0.8, 0.8]) == 1.0
    assert 0.89 < analyzer._jain([1.0, 0.5]) < 0.91


def test_node_binds_the_preregistered_request_seed(monkeypatch) -> None:
    monkeypatch.setenv(node.CONTRACT_ENV, str(CONTRACT))
    monkeypatch.setattr(
        node, "_FROZEN_CLIENT_COMMAND", lambda *args, **kwargs: ["python"])
    command = node._client_command()
    assert command == ["python", "--seed", "20260825"]


def test_node_rejects_discovery_allocation_and_accepts_fresh_job(monkeypatch) -> None:
    monkeypatch.setenv(node.CONTRACT_ENV, str(CONTRACT))
    monkeypatch.setenv("SLURM_JOB_ID", "99999999")
    path, value = node._qualification(ROOT)
    assert path == CONTRACT.resolve()
    assert value["independent_validation"]["fresh_allocation_required"] is True
    monkeypatch.setenv("SLURM_JOB_ID", "57583281")
    with pytest.raises(ValueError, match="fresh Slurm allocation"):
        node._qualification(ROOT)


@pytest.mark.parametrize(
    ("arm", "expected_prompt_count"),
    [
        ("full_c7_managed_background", 16),
        ("fixed_local_d0", 8),
    ],
)
def test_execution_receipt_uses_explicit_order_after_sorted_json(
    tmp_path, monkeypatch, arm: str, expected_prompt_count: int,
) -> None:
    value = _contract()
    heldout = value["independent_validation"]
    expected_blocks = [
        row["name"] for row in value["joint_control"]["blocks"]
    ]
    remote_name = heldout["remote_favorable_block"]
    artifacts = {}
    for name in sorted(expected_blocks):
        raw_path = tmp_path / f"{name}.json"
        raw = {
            "workload": {
                "seed": heldout["request_seed"],
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
            },
        }
        if name == remote_name:
            raw["c8_dual_regime_contract"] = {
                "request_index": {
                    f"victim-{index}": {
                        "role": "victim",
                        "prompt_sha256": f"{index:064x}",
                    }
                    for index in range(expected_prompt_count)
                },
            }
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        artifacts[name] = str(raw_path)
    output = tmp_path / "bundle.json"
    output.write_text(json.dumps({
        "schema": client.SCHEMA,
        "arm": arm,
        "qualification_contract_sha256": hashlib.sha256(
            CONTRACT.read_bytes()).hexdigest(),
        # json sort_keys deliberately serializes artifacts alphabetically;
        # this explicit list is the causal execution order.
        "artifacts": artifacts,
        "block_order": value["joint_control"]["blocks"],
    }, sort_keys=True), encoding="utf-8")
    monkeypatch.setenv(client.ARM_ENV, arm)
    client._execution_receipt(
        output=output, contract_path=CONTRACT, contract=value)
    receipt = json.loads(output.read_text(encoding="utf-8"))[
        "independent_validation_execution"]
    assert receipt["block_order"] == expected_blocks
    assert len(receipt["remote_p_only_prompt_sha256s"]) == expected_prompt_count
