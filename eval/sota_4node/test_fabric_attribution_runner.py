from __future__ import annotations

import unittest

from eval.sota_4node.fabric_attribution_runner import (
    build_g2_manifest,
    build_g2_matrix,
    validate_g2_manifest,
)
from tempo.causal_gate import CausalPromotion
from tempo.resource_domain import ResourceDomain
from tempo.tier_attribution import TierEvaluation


def successful_g1() -> TierEvaluation:
    return TierEvaluation(
        promotion=CausalPromotion(
            frozenset({ResourceDomain.PCIE_HOST}), True, True, ()
        ),
        evidence_ready=True,
        reasons=(),
    )


class FabricAttributionRunnerTests(unittest.TestCase):
    def test_g2_requires_g1_promotion(self) -> None:
        with self.assertRaisesRegex(ValueError, "G2 requires"):
            build_g2_manifest(
                promoted_domain=ResourceDomain.PCIE_HOST,
                placebo_domain=ResourceDomain.HOST_NUMA,
                g1_evaluation=TierEvaluation(
                    promotion=CausalPromotion(frozenset(), False, False, ("no",)),
                    evidence_ready=False,
                    reasons=("no",),
                ),
                state_bytes_per_rank=402_705_672,
                deadline_ns=1_000_000_000,
                checkpoint_steps=[16, 52],
            )

    def test_g2_matrix_has_open_static_placebo_and_combined_modes(self) -> None:
        runs = build_g2_matrix(ResourceDomain.PCIE_HOST, ResourceDomain.NIC_FABRIC)
        self.assertEqual([run.mode for run in runs], [
            "fg_only", "open_combined", "causal_domain_static_cap",
            "unrelated_domain_placebo", "combined",
        ])
        self.assertEqual(runs[2].auxiliary_domains, ("pcie_host",))
        self.assertEqual(runs[3].auxiliary_domains, ("nic_fabric",))

    def test_g2_manifest_is_non_submitting_and_splits_fabric_paths(self) -> None:
        manifest = build_g2_manifest(
            promoted_domain=ResourceDomain.PCIE_HOST,
            placebo_domain=ResourceDomain.NIC_FABRIC,
            g1_evaluation=successful_g1(),
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16, 52],
        )
        self.assertEqual((manifest["world_size"], manifest["nodes"]), (8, 2))
        self.assertEqual(manifest["collective_slices"], ["intra_node", "inter_node"])
        self.assertEqual(manifest["fabric_splits"], ["gdr_gpu_originated", "host_originated", "pfs_endpoint"])
        self.assertFalse(manifest["slurm_submitted"])
        self.assertEqual(manifest["evidence_state"], "design_only")

    def test_g2_rejects_boolean_instead_of_g1_evaluation(self) -> None:
        with self.assertRaisesRegex(TypeError, "TierEvaluation"):
            build_g2_manifest(
                promoted_domain=ResourceDomain.PCIE_HOST,
                placebo_domain=ResourceDomain.NIC_FABRIC,
                g1_evaluation=True,
                state_bytes_per_rank=402_705_672,
                deadline_ns=1_000_000_000,
                checkpoint_steps=[16],
            )

    def test_g2_rejects_same_promoted_and_placebo_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ"):
            build_g2_matrix(ResourceDomain.PCIE_HOST, ResourceDomain.PCIE_HOST)

    def test_g2_validator_rejects_promoted_domain_not_in_g1(self) -> None:
        candidate = build_g2_manifest(
            promoted_domain=ResourceDomain.PCIE_HOST,
            placebo_domain=ResourceDomain.NIC_FABRIC,
            g1_evaluation=successful_g1(),
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16, 52],
        )
        candidate["g1_eligible_domains"] = []
        with self.assertRaisesRegex(ValueError, "include the promoted"):
            validate_g2_manifest(candidate)

    def test_g2_validator_rejects_live_or_coerced_manifest(self) -> None:
        candidate = build_g2_manifest(
            promoted_domain=ResourceDomain.PCIE_HOST,
            placebo_domain=ResourceDomain.NIC_FABRIC,
            g1_evaluation=successful_g1(),
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16],
        )
        candidate["slurm_submitted"] = 0
        with self.assertRaisesRegex(ValueError, "never submit"):
            validate_g2_manifest(candidate)

    def test_g2_validator_rejects_unknown_eligible_domain(self) -> None:
        candidate = build_g2_manifest(
            promoted_domain=ResourceDomain.PCIE_HOST,
            placebo_domain=ResourceDomain.NIC_FABRIC,
            g1_evaluation=successful_g1(),
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16],
        )
        candidate["g1_eligible_domains"] = ["made_up_domain", "pcie_host"]
        with self.assertRaisesRegex(ValueError, "unknown resource domain"):
            validate_g2_manifest(candidate)


if __name__ == "__main__":
    unittest.main()
