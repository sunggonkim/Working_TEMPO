from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import run_tempo_pd_contention_fixed_client as client
from tempo.pd_contention_workload import ContentionState, ForegroundArm


_OUTPUT_SHA256 = hashlib.sha256(b"forced-output").hexdigest()


def _request_index() -> dict[str, dict[str, object]]:
    return {
        "foreground": {
            "tenant": "foreground",
            "arm": "local",
            "pair_key": "c1_decoder_hot:foreground:000000",
        },
        "background": {
            "tenant": "decoder_hot",
            "arm": "local",
            "pair_key": None,
        },
    }


def _raw(index: dict[str, dict[str, object]]) -> dict[str, object]:
    requests = [
        {
            "request_id": request_id,
            "valid": True,
            "router": {"route": client.LOCAL_ROUTE},
            "dispatch_offset_ns": 1_000_000,
            "stream_end_offset_ns": 3_000_000,
            "output_text_sha256": _OUTPUT_SHA256,
        }
        for request_id in index
    ]
    decisions = [
        {
            "request_id": request_id,
            "route": client.LOCAL_ROUTE,
            "benchmark_cold_measured": True,
            "decision_cache_residency": "unknown",
            "cache_residency": "confirmed_miss",
            "completion_cache_residency": "confirmed_miss",
            "lmcache_source_cached_tokens": None,
            "lmcache_source_full_hit_observed": None,
        }
        for request_id in index
    ]
    return {
        "validation": {"performance_claim_allowed": True},
        "requests": requests,
        "router_decisions": decisions,
    }


def _endpoint_evidence() -> dict[str, object]:
    identities = (
        "pair0-prefill", "pair0-decoder", "pair1-prefill", "pair1-decoder",
    )
    stages = {}
    for sequence, stage in enumerate(("before", "midpoint", "after"), 1):
        stages[stage] = {
            "schema": client.ENDPOINT_EVIDENCE_SCHEMA,
            "stage": stage,
            "snapshots": [
                {
                    "probe": {
                        "endpoint": {"endpoint_id": endpoint_id,
                                     "sequence": sequence},
                        "cassini": {"sequence": sequence},
                    },
                }
                for endpoint_id in identities
            ],
        }
    return {
        "schema": client.ENDPOINT_EVIDENCE_SCHEMA,
        "sampling_policy": "on_demand_block_boundary_and_midpoint",
        "cross_endpoint_clock_subtraction_allowed": False,
        **stages,
    }


def _cadenced_endpoint_evidence() -> dict[str, object]:
    value = _endpoint_evidence()
    value["sampling_policy"] = (
        "on_demand_block_boundary_midpoint_and_cassini_bridges"
    )
    value["cassini_bridge_interval_s"] = 5.0
    value["midpoint_target_elapsed_s"] = 30.0
    value["child_elapsed_s"] = 60.0
    before = value["before"]
    midpoint = value["midpoint"]
    after = value["after"]

    def stage(name: str, sequence: int) -> dict[str, object]:
        result = json.loads(json.dumps(before))
        result["stage"] = name
        for row in result["snapshots"]:
            row["probe"]["endpoint"]["sequence"] = sequence
            row["probe"]["cassini"] = {"sequence": sequence, "valid": True}
        return result

    for row in before["snapshots"]:
        row["probe"]["cassini"]["valid"] = False
    value["cassini_bridges"] = [
        {
            "ordinal": 0,
            "segment": "before_midpoint",
            "captured_elapsed_s": 5.0,
            "evidence": stage("bridge-000", 2),
        },
        {
            "ordinal": 1,
            "segment": "after_midpoint",
            "captured_elapsed_s": 35.0,
            "evidence": stage("bridge-001", 4),
        },
    ]
    for row in midpoint["snapshots"]:
        row["probe"]["endpoint"]["sequence"] = 3
        row["probe"]["cassini"] = {"sequence": 3, "valid": True}
    for row in after["snapshots"]:
        row["probe"]["endpoint"]["sequence"] = 5
        row["probe"]["cassini"] = {"sequence": 5, "valid": True}
    return value


class ContentionFixedClientTest(unittest.TestCase):
    def test_explicit_miss_cold_contract_requires_confirmed_decision(self) -> None:
        decision = _raw(_request_index())["router_decisions"][0]
        decision["decision_cache_residency"] = "confirmed_miss"
        decision["request_cache_contract"] = "miss"
        self.assertFalse(client._cold_completion_valid(decision))
        self.assertTrue(client._cold_completion_valid(
            decision, require_explicit_miss=True,
        ))
        decision["request_cache_contract"] = None
        self.assertFalse(client._cold_completion_valid(
            decision, require_explicit_miss=True,
        ))

    def test_preflight_is_cold_route_check_not_p_only_warm_seed(self) -> None:
        self.assertEqual(
            client.PREFLIGHT_REQUESTS,
            (
                (ForegroundArm.LOCAL, "epd-local-ct-preflight-local"),
                (ForegroundArm.REMOTE, "epd-remote-ct-preflight-remote"),
            ),
        )
        self.assertTrue(all(
            "-warm-" not in request_id
            for _arm, request_id in client.PREFLIGHT_REQUESTS
        ))

    def test_block_order_is_exact_balanced_two_replicate_matrix(self) -> None:
        self.assertEqual(len(client.BLOCK_ORDER), 8)
        self.assertEqual(
            client.BLOCK_ORDER,
            (
                (ContentionState.C1, ForegroundArm.LOCAL, 0),
                (ContentionState.C1, ForegroundArm.REMOTE, 0),
                (ContentionState.C2, ForegroundArm.REMOTE, 0),
                (ContentionState.C2, ForegroundArm.LOCAL, 0),
                (ContentionState.C2, ForegroundArm.LOCAL, 1),
                (ContentionState.C2, ForegroundArm.REMOTE, 1),
                (ContentionState.C1, ForegroundArm.REMOTE, 1),
                (ContentionState.C1, ForegroundArm.LOCAL, 1),
            ),
        )

    def test_augment_block_requires_exact_pinned_routes(self) -> None:
        index = _request_index()
        schedule_sha = hashlib.sha256(b"schedule").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(_raw(index)), encoding="utf-8")
            contract = client._augment_block(
                path,
                phase=ContentionState.C1,
                arm=ForegroundArm.LOCAL,
                replicate=0,
                load_fraction=0.5,
                schedule_sha256=schedule_sha,
                request_index=index,
                endpoint_evidence=_endpoint_evidence(),
            )
            self.assertEqual(contract["request_counts"], {
                "decoder_hot": 1,
                "foreground": 1,
            })
            self.assertEqual(contract["semantic_schedule_sha256"], schedule_sha)

    def test_augment_block_fails_closed_on_wrong_route(self) -> None:
        index = _request_index()
        raw = _raw(index)
        raw["requests"][0]["router"]["route"] = client.REMOTE_ROUTE
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned route"):
                client._augment_block(
                    path,
                    phase=ContentionState.C1,
                    arm=ForegroundArm.LOCAL,
                    replicate=0,
                    load_fraction=0.5,
                    schedule_sha256=hashlib.sha256(b"schedule").hexdigest(),
                    request_index=index,
                    endpoint_evidence=_endpoint_evidence(),
                )

    def test_endpoint_evidence_rejects_nonincreasing_sequence(self) -> None:
        evidence = _endpoint_evidence()
        evidence["midpoint"]["snapshots"][0]["probe"]["cassini"]["sequence"] = 1
        with self.assertRaisesRegex(ValueError, "Cassini evidence sequence"):
            client._validate_endpoint_evidence_bundle(evidence)

    def test_endpoint_evidence_accepts_cadenced_cassini_bridges(self) -> None:
        client._validate_endpoint_evidence_bundle(_cadenced_endpoint_evidence())

    def test_cadenced_endpoint_evidence_rejects_invalid_bridge(self) -> None:
        evidence = _cadenced_endpoint_evidence()
        evidence["cassini_bridges"][0]["evidence"]["snapshots"][0][
            "probe"
        ]["cassini"]["valid"] = False
        with self.assertRaisesRegex(ValueError, "invalid delta"):
            client._validate_endpoint_evidence_bundle(evidence)

    def test_observation_uses_dispatch_to_stream_end_and_background_exactness(self) -> None:
        index = _request_index()
        raw = _raw(index)
        raw["contention_fixed_contract"] = {
            "phase": ContentionState.C1.value,
            "load_fraction": 0.5,
            "replicate": 0,
            "foreground_arm": ForegroundArm.LOCAL.value,
            "semantic_schedule_sha256": hashlib.sha256(b"schedule").hexdigest(),
            "request_index": index,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            observation = client._observation(path)
        self.assertEqual(observation.foreground[0].e2e_ms, 2.0)
        self.assertEqual(observation.background_offered, 1)
        self.assertEqual(observation.background_completed, 1)
        self.assertEqual(observation.background_errors, 0)


if __name__ == "__main__":
    unittest.main()
