from __future__ import annotations

import unittest
import math

from tempo.causal_gate import (
    CausalGateConfig,
    CausalModeRecord,
    InferenceModeRecord,
    evaluate_causal_matrix,
    evaluate_inference_matrix,
)
from tempo.resource_domain import ResourceDomain


class CausalGateTests(unittest.TestCase):
    def base(self) -> list[CausalModeRecord]:
        return [
            CausalModeRecord("fg_only", None, 100, 100, True, True, 3),
            CausalModeRecord("open_combined", None, 130, 130, True, True, 3),
        ]

    def test_domain_intervention_promotes_only_when_both_metrics_improve(self) -> None:
        records = self.base() + [
            CausalModeRecord("d2h_only", ResourceDomain.PCIE_HOST, 110, 105, True, True, 3),
            CausalModeRecord("host_pressure", ResourceDomain.HOST_NUMA, 135, 140, True, True, 3),
        ]
        result = evaluate_causal_matrix(records)
        self.assertTrue(result.headroom)
        self.assertEqual(result.eligible_domains, frozenset({ResourceDomain.PCIE_HOST}))
        self.assertTrue(result.promote_static_policy)

    def test_no_open_headroom_stops_scheduler_promotion(self) -> None:
        records = [
            CausalModeRecord("fg_only", None, 100, 100, True, True, 3),
            CausalModeRecord("open_combined", None, 102, 101, True, True, 3),
            CausalModeRecord("d2h_only", ResourceDomain.PCIE_HOST, 90, 90, True, True, 3),
        ]
        result = evaluate_causal_matrix(records)
        self.assertFalse(result.headroom)
        self.assertFalse(result.promote_static_policy)

    def test_failed_deadline_cannot_become_a_domain_candidate(self) -> None:
        result = evaluate_causal_matrix(
            self.base()
            + [
                CausalModeRecord(
                    "persist_only", ResourceDomain.PERSISTENT_ENDPOINT, 90, 90, False, True, 3
                )
            ]
        )
        self.assertEqual(result.eligible_domains, frozenset())
        self.assertIn("deadline/correctness failed", " ".join(result.reasons))

    def test_baseline_sample_shortfall_blocks_training_headroom(self) -> None:
        records = self.base()
        records[0] = CausalModeRecord(
            "fg_only", None, 100, 100, True, True, 1
        )
        result = evaluate_causal_matrix(records)
        self.assertFalse(result.headroom)
        self.assertFalse(result.promote_static_policy)
        self.assertIn("insufficient foreground/open samples", result.reasons)

    def test_full_flow_consistency_replicate_is_not_a_placebo(self) -> None:
        result = evaluate_causal_matrix(
            self.base()
            + [
                CausalModeRecord("combined", None, 90, 90, True, True, 3),
                CausalModeRecord("d2h_only", ResourceDomain.PCIE_HOST, 110, 105, True, True, 3),
            ]
        )
        self.assertTrue(result.placebo_clean)
        self.assertTrue(result.promote_static_policy)
        self.assertEqual(result.eligible_domains, frozenset({ResourceDomain.PCIE_HOST}))

    def test_training_domain_map_rejects_new_positive_exposure(self) -> None:
        records = [
            CausalModeRecord(
                "fg_only", None, 100, 100, True, True, 3,
                domain_exposure_ns={ResourceDomain.PCIE_HOST: 0},
            ),
            CausalModeRecord(
                "open_combined", None, 130, 130, True, True, 3,
                domain_exposure_ns={ResourceDomain.PCIE_HOST: 100},
            ),
            CausalModeRecord(
                "d2h_only", ResourceDomain.PCIE_HOST, 110, 105, True, True, 3,
                domain_exposure_ns={
                    ResourceDomain.PCIE_HOST: 40,
                    ResourceDomain.NVLINK_P2P: 20,
                },
            ),
        ]
        result = evaluate_causal_matrix(records)
        self.assertFalse(result.promote_static_policy)
        self.assertIn("unpaired exposed domains", " ".join(result.reasons))

    def test_training_domain_map_rejects_missing_intervention_domain(self) -> None:
        records = [
            CausalModeRecord("fg_only", None, 100, 100, True, True, 3),
            CausalModeRecord(
                "open_combined", None, 130, 130, True, True, 3,
                domain_exposure_ns={
                    ResourceDomain.PCIE_HOST: 100,
                    ResourceDomain.HOST_NUMA: 80,
                },
            ),
            CausalModeRecord(
                "d2h_only", ResourceDomain.PCIE_HOST, 110, 105, True, True, 3,
                domain_exposure_ns={ResourceDomain.HOST_NUMA: 40},
            ),
        ]
        result = evaluate_causal_matrix(records)
        self.assertFalse(result.promote_static_policy)
        self.assertIn("intervention domain exposure is missing", " ".join(result.reasons))

    def test_missing_open_is_fail_closed(self) -> None:
        result = evaluate_causal_matrix([CausalModeRecord("fg_only", None, 100, 100, True, True, 3)])
        self.assertFalse(result.promote_static_policy)
        self.assertIn("missing modes", result.reasons[0])

    def test_duplicate_training_mode_cannot_be_overwritten(self) -> None:
        result = evaluate_causal_matrix(
            self.base()
            + [CausalModeRecord("d2h_only", ResourceDomain.PCIE_HOST, 110, 105, True, True, 3)]
            + [CausalModeRecord("d2h_only", ResourceDomain.HOST_NUMA, 90, 90, True, True, 3)]
        )
        self.assertFalse(result.promote_static_policy)
        self.assertIn("duplicate modes", result.reasons[0])

    def test_baseline_intervention_domain_is_rejected(self) -> None:
        result = evaluate_causal_matrix([
            CausalModeRecord("fg_only", ResourceDomain.PCIE_HOST, 100, 100, True, True, 3),
            CausalModeRecord("open_combined", None, 130, 130, True, True, 3),
        ])
        self.assertFalse(result.headroom)
        self.assertFalse(result.promote_static_policy)
        self.assertIn("baseline must not name", " ".join(result.reasons))

    def test_invalid_open_baseline_cannot_create_training_headroom(self) -> None:
        records = [
            CausalModeRecord("fg_only", None, 100, 100, True, True, 3),
            CausalModeRecord("open_combined", None, 140, 140, False, True, 3),
            CausalModeRecord("d2h_only", ResourceDomain.PCIE_HOST, 90, 90, True, True, 3),
        ]
        result = evaluate_causal_matrix(records)
        self.assertFalse(result.headroom)
        self.assertFalse(result.promote_static_policy)
        self.assertIn("optimized-open baseline deadline/correctness failed", result.reasons)

    def test_training_headroom_uses_exact_inclusive_margin_boundary(self) -> None:
        exact = evaluate_causal_matrix([
            CausalModeRecord("fg_only", None, 100, 100, True, True, 2),
            CausalModeRecord("open_combined", None, 105, 105, True, True, 2),
        ])
        below = evaluate_causal_matrix([
            CausalModeRecord("fg_only", None, 100, 100, True, True, 2),
            CausalModeRecord("open_combined", None, 104, 104, True, True, 2),
        ])
        self.assertTrue(exact.headroom)
        self.assertFalse(below.headroom)

    def test_non_finite_margin_is_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CausalGateConfig(practical_tail_margin=value)

    def inference_base(self) -> list[InferenceModeRecord]:
        return [
            InferenceModeRecord("fg_only", None, 100, 50, 950_000, True, True, 3),
            InferenceModeRecord("open_combined", None, 130, 70, 900_000, True, True, 3),
        ]

    def test_inference_intervention_requires_both_latency_tails_and_preserves_goodput(self) -> None:
        result = evaluate_inference_matrix(
            self.inference_base()
            + [InferenceModeRecord("pcie_cap", ResourceDomain.PCIE_HOST, 110, 60, 900_000, True, True, 3)]
        )
        self.assertTrue(result.headroom)
        self.assertEqual(result.eligible_domains, frozenset({ResourceDomain.PCIE_HOST}))

    def test_inference_no_headroom_stops_promotion(self) -> None:
        result = evaluate_inference_matrix([
            InferenceModeRecord("fg_only", None, 100, 50, 950_000, True, True, 3),
            InferenceModeRecord("open_combined", None, 102, 51, 949_000, True, True, 3),
            InferenceModeRecord("pcie_cap", ResourceDomain.PCIE_HOST, 90, 45, 950_000, True, True, 3),
        ])
        self.assertFalse(result.headroom)
        self.assertFalse(result.promote_static_policy)

    def test_inference_deadline_or_goodput_failure_blocks_domain(self) -> None:
        result = evaluate_inference_matrix(
            self.inference_base()
            + [InferenceModeRecord("pcie_cap", ResourceDomain.PCIE_HOST, 110, 60, 880_000, False, True, 3)]
        )
        self.assertEqual(result.eligible_domains, frozenset())
        self.assertIn("deadline/correctness failed", " ".join(result.reasons))

    def test_baseline_sample_shortfall_blocks_inference_headroom(self) -> None:
        records = self.inference_base()
        records[1] = InferenceModeRecord(
            "open_combined", None, 120, 120, 900_000, True, True, 1
        )
        result = evaluate_inference_matrix(records)
        self.assertFalse(result.headroom)
        self.assertFalse(result.promote_static_policy)
        self.assertIn("insufficient foreground/open samples", result.reasons)

    def test_inference_goodput_range_is_strict(self) -> None:
        with self.assertRaises(ValueError):
            InferenceModeRecord("bad", None, 1, 1, 1_000_001, True, True, 1)

    def test_invalid_open_baseline_cannot_create_inference_headroom(self) -> None:
        records = [
            InferenceModeRecord("fg_only", None, 100, 50, 950_000, True, True, 3),
            InferenceModeRecord("open_combined", None, 140, 70, 900_000, False, True, 3),
            InferenceModeRecord("pcie_cap", ResourceDomain.PCIE_HOST, 90, 40, 900_000, True, True, 3),
        ]
        result = evaluate_inference_matrix(records)
        self.assertFalse(result.headroom)
        self.assertFalse(result.promote_static_policy)
        self.assertIn("optimized-open baseline deadline/correctness failed", result.reasons)

    def test_inference_goodput_drop_uses_exact_inclusive_boundary(self) -> None:
        result = evaluate_inference_matrix([
            InferenceModeRecord("fg_only", None, 100, 100, 1_000_000, True, True, 2),
            InferenceModeRecord("open_combined", None, 100, 100, 950_000, True, True, 2),
        ])
        self.assertTrue(result.headroom)

    def test_duplicate_inference_mode_cannot_be_overwritten(self) -> None:
        result = evaluate_inference_matrix(
            self.inference_base()
            + [InferenceModeRecord("pcie_cap", ResourceDomain.PCIE_HOST, 110, 60, 900_000, True, True, 2)]
            + [InferenceModeRecord("pcie_cap", ResourceDomain.HOST_NUMA, 90, 40, 950_000, True, True, 2)]
        )
        self.assertFalse(result.promote_static_policy)
        self.assertIn("duplicate modes", result.reasons[0])

    def test_inference_domain_map_blocks_hidden_bottleneck_shift(self) -> None:
        records = [
            InferenceModeRecord(
                "fg_only", None, 100, 50, 950_000, True, True, 3,
                domain_exposure_ns={},
            ),
            InferenceModeRecord(
                "open_combined", None, 130, 70, 900_000, True, True, 3,
                max_domain_exposure_ns=200,
                domain_exposure_ns={
                    ResourceDomain.GPU_LOCAL: 200,
                    ResourceDomain.NIC_FABRIC: 100,
                },
            ),
            InferenceModeRecord(
                "remote_fabric", ResourceDomain.NIC_FABRIC, 110, 60, 900_000,
                True, True, 3,
                max_domain_exposure_ns=150,
                domain_exposure_ns={ResourceDomain.NIC_FABRIC: 150},
            ),
        ]
        result = evaluate_inference_matrix(records)
        self.assertFalse(result.promote_static_policy)
        self.assertIn("domain exposure", " ".join(result.reasons))

    def test_inference_rejects_new_route_domains_without_matched_open(self) -> None:
        # A lower scalar maximum on a newly added NIC route is not evidence of
        # orchestration: the endpoint/path changed.  Such a mode needs its
        # own open_combined baseline with the identical route domain set.
        records = [
            InferenceModeRecord(
                "fg_only", None, 100, 100, 950_000, True, True, 3,
                domain_exposure_ns={},
            ),
            InferenceModeRecord(
                "open_combined", None, 130, 130, 900_000, True, True, 3,
                max_domain_exposure_ns=200,
                domain_exposure_ns={ResourceDomain.GPU_LOCAL: 200},
            ),
            InferenceModeRecord(
                "remote_fabric", ResourceDomain.NIC_FABRIC, 100, 100, 900_000,
                True, True, 3,
                max_domain_exposure_ns=50,
                domain_exposure_ns={ResourceDomain.NIC_FABRIC: 50},
            ),
        ]
        result = evaluate_inference_matrix(records)
        self.assertFalse(result.promote_static_policy)
        self.assertNotIn(ResourceDomain.NIC_FABRIC, result.eligible_domains)
        self.assertIn("matched endpoint baseline required", " ".join(result.reasons))


if __name__ == "__main__":
    unittest.main()
