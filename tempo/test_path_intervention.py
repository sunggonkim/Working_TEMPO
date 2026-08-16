from __future__ import annotations

import unittest

from tempo.path_intervention import (
    PathInterventionEvidence,
    build_causal_domain_controller,
    controller_ready_domains,
    path_intervention_candidates,
    validate_path_intervention_artifact,
)
from tempo.resource_domain import ResourceDomain


def p2p_record(**overrides: object) -> PathInterventionEvidence:
    values: dict[str, object] = {
        "domain": ResourceDomain.GPU_LOCAL,
        "intervention_id": "g2-p2p-disable-56858081",
        "control_name": "NCCL_P2P_DISABLE",
        "control_value": "1",
        "baseline_mode": "open_combined",
        "intervention_mode": "open_combined",
        "baseline_step_p99_ns": 70_280_000,
        "intervention_step_p99_ns": 308_831_000,
        "baseline_window_p99_ns": 70_876_000,
        "intervention_window_p99_ns": 311_859_000,
        "baseline_skew_p99_ns": None,
        "intervention_skew_p99_ns": None,
        "sample_count": 8,
        "uncertainty_ns": 1_000_000,
        "source": "job-56857331-vs-56858081",
        "baseline_workload_digest": "same-model-geometry-v1",
        "intervention_workload_digest": "same-model-geometry-v1",
        "baseline_placement_digest": "same-2node-placement-v1",
        "intervention_placement_digest": "same-2node-placement-v1",
        "auxiliary_bytes_baseline": 403_480_576,
        "auxiliary_bytes_intervention": 403_480_576,
        "path_status": "observed",
        "counter_scope": "none",
        "byte_attribution": False,
        "observed_path": "nccl_direct_p2p_disabled_isAllDirectP2p_0",
    }
    values.update(overrides)
    return PathInterventionEvidence(**values)  # type: ignore[arg-type]


class PathInterventionTests(unittest.TestCase):
    def test_p2p_path_is_causal_candidate_but_not_controller_ready(self) -> None:
        record = p2p_record()
        self.assertTrue(record.causal_path_candidate)
        self.assertFalse(record.controller_ready)
        self.assertEqual(path_intervention_candidates([record]), {ResourceDomain.GPU_LOCAL})
        self.assertEqual(controller_ready_domains([record]), frozenset())

    def test_missing_matched_bytes_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            p2p_record(auxiliary_bytes_intervention=1)

    def test_small_effect_is_not_causal(self) -> None:
        record = p2p_record(
            intervention_step_p99_ns=71_000_000,
            intervention_window_p99_ns=71_000_000,
        )
        self.assertFalse(record.causal_path_candidate)

    def test_geometry_mismatch_is_not_causal(self) -> None:
        record = p2p_record(intervention_placement_digest="different-placement")
        self.assertFalse(record.causal_path_candidate)

    def test_declared_path_cannot_be_causal(self) -> None:
        record = p2p_record(path_status="declared")
        self.assertFalse(record.causal_path_candidate)

    def test_byte_counter_is_required_for_controller_ready(self) -> None:
        record = p2p_record(counter_scope="rank", byte_attribution=True)
        self.assertTrue(record.controller_ready)
        self.assertEqual(controller_ready_domains([record]), {ResourceDomain.GPU_LOCAL})

    def test_byte_counter_scope_must_match_domain(self) -> None:
        with self.assertRaises(ValueError):
            p2p_record(domain=ResourceDomain.NVLINK_P2P, counter_scope="rank", byte_attribution=True)

    def test_controller_ready_path_can_enter_shared_observation_only_with_explicit_overlap(self) -> None:
        record = p2p_record(counter_scope="rank", byte_attribution=True)
        observation = record.to_domain_observation(
            foreground_kind="fsdp_collective",
            auxiliary_kind="checkpoint_d2h",
            overlap_ns=5_000_000,
        )
        self.assertEqual(observation.domain, ResourceDomain.GPU_LOCAL)
        self.assertEqual(observation.overlapping_bytes, 403_480_576)
        with self.assertRaises(ValueError):
            p2p_record().to_domain_observation(
                foreground_kind="fsdp_collective",
                auxiliary_kind="checkpoint_d2h",
                overlap_ns=5_000_000,
            )

    def test_controller_bridge_rejects_path_only_evidence(self) -> None:
        from tempo.domain_admission import DomainBudget

        budget = {
            ResourceDomain.GPU_LOCAL: DomainBudget(
                ResourceDomain.GPU_LOCAL, 1_000_000_000, 1_048_576
            )
        }
        with self.assertRaisesRegex(ValueError, "without causal byte evidence"):
            build_causal_domain_controller([p2p_record()], budget, catch_up_slack_ns=0)

    def test_controller_bridge_accepts_only_matching_ready_domain(self) -> None:
        from tempo.domain_admission import DomainBudget

        record = p2p_record(counter_scope="rank", byte_attribution=True)
        budget = {
            ResourceDomain.GPU_LOCAL: DomainBudget(
                ResourceDomain.GPU_LOCAL, 1_000_000_000, 1_048_576
            )
        }
        controller = build_causal_domain_controller([record], budget, catch_up_slack_ns=0)
        self.assertEqual(set(controller.budgets), {ResourceDomain.GPU_LOCAL})

    def test_non_boolean_byte_attribution_rejected(self) -> None:
        with self.assertRaises(TypeError):
            p2p_record(byte_attribution=1)

    def test_json_schema_round_trip_and_derived_flags(self) -> None:
        record = p2p_record()
        restored = PathInterventionEvidence.from_dict(record.to_dict())
        self.assertEqual(restored, record)
        payload = record.to_dict()
        payload["causal_path_candidate"] = False
        with self.assertRaises(ValueError):
            PathInterventionEvidence.from_dict(payload)

    def test_live_path_artifact_is_bound_to_both_raw_manifests(self) -> None:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        artifact = root / "results/sota_4node/g2_p2p_path_job_56858081/path_intervention_evidence.json"
        if not artifact.is_file():
            self.skipTest("local raw path-intervention artifact is not present")
        record = validate_path_intervention_artifact(json.loads(artifact.read_text()), repo_root=root)
        self.assertEqual(record.intervention_id, "g2-p2p-disable-56858081")

    def test_path_artifact_rejects_manifest_source_mismatch(self) -> None:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        artifact = root / "results/sota_4node/g2_p2p_path_job_56858081/path_intervention_evidence.json"
        if not artifact.is_file():
            self.skipTest("local raw path-intervention artifact is not present")
        payload = json.loads(artifact.read_text())
        payload["provenance"]["intervention_source_bundle_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_path_intervention_artifact(payload, repo_root=root)

    def test_path_artifact_rejects_repository_escape(self) -> None:
        from pathlib import Path

        payload = p2p_record().to_dict()
        payload["provenance"] = {
            "baseline_job": 1,
            "intervention_job": 2,
            "baseline_artifact_root": "../outside",
            "intervention_artifact_root": "results",
            "baseline_source_bundle_sha256": "a" * 64,
            "intervention_source_bundle_sha256": "b" * 64,
            "matched_modes": ["combined", "d2h_only", "fg_only", "open_combined", "persist_only"],
            "groups_per_mode": 3,
            "restores_per_mode": 1,
        }
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "inside repo_root"):
            validate_path_intervention_artifact(payload, repo_root=root)


if __name__ == "__main__":
    unittest.main()
