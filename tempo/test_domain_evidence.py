from __future__ import annotations

import unittest

from tempo.domain_evidence import (
    CounterSupport,
    assess_domain_coverage,
    DomainEvidence,
    PathStatus,
    controller_candidates,
    detect_bottleneck_shift,
)
from tempo.resource_domain import EvidenceLevel, ResourceDomain, domain_contract


class DomainEvidenceTests(unittest.TestCase):
    def record(self, *, domain: ResourceDomain, evidence: EvidenceLevel, support: CounterSupport, delta: int, uncertainty: int = 10) -> DomainEvidence:
        return DomainEvidence(
            domain=domain,
            mode="d2h_only",
            foreground_kind="fsdp_all_gather",
            auxiliary_kind="checkpoint_d2h",
            overlapping_bytes=4096,
            overlap_ns=1000,
            tail_delta_ns=delta,
            evidence=evidence,
            counter_support=support,
            path_status=PathStatus.OBSERVED,
            uncertainty_ns=uncertainty,
            source="fixture",
            path_evidence=domain_contract(domain).path_evidence,
            counter_family=domain_contract(domain).counter_family,
        )

    def test_only_interventional_supported_above_uncertainty_is_candidate(self) -> None:
        records = [
            self.record(domain=ResourceDomain.PCIE_HOST, evidence=EvidenceLevel.OBSERVATIONAL, support=CounterSupport.SUPPORTED, delta=100),
            self.record(domain=ResourceDomain.PCIE_HOST, evidence=EvidenceLevel.INTERVENTIONAL, support=CounterSupport.SUPPORTED, delta=100),
            self.record(domain=ResourceDomain.NIC_FABRIC, evidence=EvidenceLevel.INTERVENTIONAL, support=CounterSupport.AMBIGUOUS, delta=100),
        ]
        self.assertEqual(controller_candidates(records), frozenset({ResourceDomain.PCIE_HOST}))

    def test_missing_counter_or_uncertainty_blocks_promotion(self) -> None:
        record = self.record(
            domain=ResourceDomain.HOST_NUMA,
            evidence=EvidenceLevel.INTERVENTIONAL,
            support=CounterSupport.NOT_COLLECTED,
            delta=100,
        )
        self.assertFalse(controller_candidates([record]))
        uncertain = self.record(
            domain=ResourceDomain.PCIE_HOST,
            evidence=EvidenceLevel.INTERVENTIONAL,
            support=CounterSupport.SUPPORTED,
            delta=10,
            uncertainty=10,
        )
        self.assertFalse(controller_candidates([uncertain]))

    def test_intervention_without_observed_path_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            DomainEvidence(
                domain=ResourceDomain.NVLINK_P2P,
                mode="p2p_only",
                foreground_kind="inference",
                auxiliary_kind="kv_migrate",
                overlapping_bytes=1,
                overlap_ns=1,
                tail_delta_ns=1,
                evidence=EvidenceLevel.INTERVENTIONAL,
                counter_support=CounterSupport.SUPPORTED,
                path_status=PathStatus.DECLARED,
                uncertainty_ns=0,
                source="fixture",
                path_evidence=domain_contract(ResourceDomain.NVLINK_P2P).path_evidence,
                counter_family=domain_contract(ResourceDomain.NVLINK_P2P).counter_family,
            )

    def test_bottleneck_shift_is_reported_when_cap_moves_exposure(self) -> None:
        before = {
            ResourceDomain.PCIE_HOST: 100,
            ResourceDomain.HOST_NUMA: 80,
            ResourceDomain.PERSISTENT_ENDPOINT: 120,
        }
        after = {
            ResourceDomain.PCIE_HOST: 40,
            ResourceDomain.HOST_NUMA: 110,
            ResourceDomain.PERSISTENT_ENDPOINT: 120,
        }
        self.assertEqual(
            detect_bottleneck_shift(before, after, controlled_domain=ResourceDomain.PCIE_HOST),
            (ResourceDomain.HOST_NUMA,),
        )

    def test_bottleneck_shift_requires_exact_snapshots_and_control_reduction(self) -> None:
        before = {ResourceDomain.PCIE_HOST: 100, ResourceDomain.HOST_NUMA: 80}
        with self.assertRaises(ValueError):
            detect_bottleneck_shift(
                before,
                {ResourceDomain.PCIE_HOST: 40},
                controlled_domain=ResourceDomain.PCIE_HOST,
            )
        self.assertEqual(
            detect_bottleneck_shift(
                before,
                {ResourceDomain.PCIE_HOST: 100, ResourceDomain.HOST_NUMA: 120},
                controlled_domain=ResourceDomain.PCIE_HOST,
            ),
            (),
        )

    def test_domain_coverage_keeps_missing_and_unsupported_tiers_visible(self) -> None:
        pcie = self.record(
            domain=ResourceDomain.PCIE_HOST,
            evidence=EvidenceLevel.INTERVENTIONAL,
            support=CounterSupport.SUPPORTED,
            delta=100,
        )
        declared_storage = DomainEvidence(
            domain=ResourceDomain.PERSISTENT_ENDPOINT,
            mode="open_combined",
            foreground_kind="fsdp_all_gather",
            auxiliary_kind="checkpoint_persist",
            overlapping_bytes=4096,
            overlap_ns=1000,
            tail_delta_ns=0,
            evidence=EvidenceLevel.OBSERVATIONAL,
            counter_support=CounterSupport.NOT_COLLECTED,
            path_status=PathStatus.DECLARED,
            uncertainty_ns=10,
            source="fixture",
            path_evidence=domain_contract(ResourceDomain.PERSISTENT_ENDPOINT).path_evidence,
            counter_family=domain_contract(ResourceDomain.PERSISTENT_ENDPOINT).counter_family,
        )
        coverage = assess_domain_coverage(
            [pcie, declared_storage],
            [ResourceDomain.PCIE_HOST, ResourceDomain.NIC_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT],
        )
        self.assertEqual(
            coverage.missing_domains,
            frozenset({ResourceDomain.NIC_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT}),
        )
        self.assertEqual(coverage.observed_domains, frozenset({ResourceDomain.PCIE_HOST}))
        self.assertEqual(coverage.supported_domains, frozenset({ResourceDomain.PCIE_HOST}))
        self.assertEqual(coverage.causal_domains, frozenset({ResourceDomain.PCIE_HOST}))
        self.assertFalse(coverage.coverage_complete)
        self.assertFalse(coverage.causal_ready)

    def test_supported_observation_without_intervention_is_not_causal_ready(self) -> None:
        observed_only = self.record(
            domain=ResourceDomain.PCIE_HOST,
            evidence=EvidenceLevel.OBSERVATIONAL,
            support=CounterSupport.SUPPORTED,
            delta=100,
        )
        coverage = assess_domain_coverage(
            [observed_only], [ResourceDomain.PCIE_HOST]
        )
        self.assertTrue(coverage.coverage_complete)
        self.assertEqual(coverage.supported_domains, frozenset({ResourceDomain.PCIE_HOST}))
        self.assertEqual(coverage.causal_domains, frozenset())
        self.assertFalse(coverage.causal_ready)


if __name__ == "__main__":
    unittest.main()
