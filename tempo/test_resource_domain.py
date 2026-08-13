#!/usr/bin/env python3
"""CPU-only tests for the TEMPO-RD resource-domain contract."""

from __future__ import annotations

import unittest
import tempo

from tempo.resource_domain import (
    DomainObservation,
    EvidenceLevel,
    FlowStage,
    ForegroundOperation,
    ResourceDomain,
    StateFlow,
    aggregate_observations,
    causal_candidate_domains,
    allowed_counter_scopes,
)


class ResourceDomainTests(unittest.TestCase):
    def test_checkpoint_path_does_not_infer_nvlink(self) -> None:
        flow = StateFlow(
            flow_id="checkpoint:event32:rank0",
            version="v1",
            deadline_ns=1_000_000_000,
            stages=(
                FlowStage(
                    stage_id="d2h",
                    bytes=64 * 1024 * 1024,
                    domains=(ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA),
                    deadline_ns=200_000_000,
                    max_residual_bytes=1 * 1024 * 1024,
                ),
                FlowStage(
                    stage_id="persist",
                    bytes=64 * 1024 * 1024,
                    domains=(ResourceDomain.NIC_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT),
                    deadline_ns=1_000_000_000,
                    max_residual_bytes=16 * 1024 * 1024,
                ),
            ),
        )
        self.assertNotIn(ResourceDomain.NVLINK_P2P, flow.domains)
        self.assertEqual(flow.total_bytes, 128 * 1024 * 1024)

    def test_public_api_exposes_complete_domain_contracts(self) -> None:
        self.assertEqual(set(tempo.DOMAIN_CONTRACTS), set(ResourceDomain))
        for domain in ResourceDomain:
            contract = tempo.domain_contract(domain)
            self.assertIsInstance(contract, tempo.DomainContract)
            self.assertTrue(contract.path_evidence)
            self.assertTrue(contract.counter_family)

    def test_counter_scope_contract_rejects_host_aggregate_for_causal_domains(self) -> None:
        self.assertIn("rank", allowed_counter_scopes(ResourceDomain.PCIE_HOST))
        self.assertNotIn("host", allowed_counter_scopes(ResourceDomain.PCIE_HOST))
        self.assertIn("slice", allowed_counter_scopes(ResourceDomain.SLINGSHOT_FABRIC))
        self.assertIn("endpoint", allowed_counter_scopes(ResourceDomain.PERSISTENT_ENDPOINT))

    def test_foreground_operation_checks_interval_and_domains(self) -> None:
        operation = ForegroundOperation(
            operation_id="ag:step32:rank0",
            kind="fsdp_all_gather",
            group_id="ag:step32",
            domains=(ResourceDomain.NVLINK_P2P, ResourceDomain.NIC_FABRIC),
            start_ns=10,
            end_ns=20,
            bytes=4096,
        )
        self.assertEqual(operation.end_ns - operation.start_ns, 10)
        with self.assertRaises(ValueError):
            ForegroundOperation(
                operation_id="bad",
                kind="ag",
                group_id="g",
                domains=(ResourceDomain.NVLINK_P2P,),
                start_ns=20,
                end_ns=10,
            )

    def test_observational_evidence_is_not_a_control_domain(self) -> None:
        observations = [
            DomainObservation(
                domain=ResourceDomain.PCIE_HOST,
                foreground_kind="fsdp_all_gather",
                auxiliary_kind="checkpoint_d2h",
                overlapping_bytes=1024,
                overlap_ns=100,
                tail_delta_ns=50,
                evidence=EvidenceLevel.OBSERVATIONAL,
                source="fixture",
            ),
            DomainObservation(
                domain=ResourceDomain.NIC_FABRIC,
                foreground_kind="fsdp_all_gather",
                auxiliary_kind="checkpoint_pfs",
                overlapping_bytes=1024,
                overlap_ns=100,
                tail_delta_ns=80,
                evidence=EvidenceLevel.OBSERVATIONAL,
                source="fixture",
            ),
        ]
        aggregates = aggregate_observations(observations)
        self.assertEqual(len(aggregates), 2)
        self.assertFalse(causal_candidate_domains(observations))

    def test_intervention_makes_only_explicit_domain_eligible(self) -> None:
        observations = [
            DomainObservation(
                domain=ResourceDomain.PCIE_HOST,
                foreground_kind="fsdp_all_gather",
                auxiliary_kind="checkpoint_d2h",
                overlapping_bytes=4096,
                overlap_ns=1000,
                tail_delta_ns=300,
                evidence=EvidenceLevel.INTERVENTIONAL,
                source="d2h_only_ablation",
                uncertainty_ns=10,
            ),
            DomainObservation(
                domain=ResourceDomain.NIC_FABRIC,
                foreground_kind="fsdp_all_gather",
                auxiliary_kind="checkpoint_pfs",
                overlapping_bytes=4096,
                overlap_ns=1000,
                tail_delta_ns=100,
                evidence=EvidenceLevel.OBSERVATIONAL,
                source="same_run",
            ),
        ]
        candidates = causal_candidate_domains(observations)
        self.assertEqual(candidates, frozenset({ResourceDomain.PCIE_HOST}))
        aggregate = next(
            value for key, value in aggregate_observations(observations).items()
            if key[0] is ResourceDomain.PCIE_HOST
        )
        self.assertTrue(aggregate.causal_candidate)

    def test_interventional_effect_below_uncertainty_is_not_promoted(self) -> None:
        observations = [
            DomainObservation(
                domain=ResourceDomain.PCIE_HOST,
                foreground_kind="fsdp_all_gather",
                auxiliary_kind="checkpoint_d2h",
                overlapping_bytes=4096,
                overlap_ns=1000,
                tail_delta_ns=5,
                evidence=EvidenceLevel.INTERVENTIONAL,
                source="small_intervention",
                uncertainty_ns=10,
            ),
            # A large observational delta must not rescue the weak
            # interventional sample in the same aggregate.
            DomainObservation(
                domain=ResourceDomain.PCIE_HOST,
                foreground_kind="fsdp_all_gather",
                auxiliary_kind="checkpoint_d2h",
                overlapping_bytes=4096,
                overlap_ns=1000,
                tail_delta_ns=100,
                evidence=EvidenceLevel.OBSERVATIONAL,
                source="observation",
                uncertainty_ns=10,
            ),
        ]
        aggregate = next(iter(aggregate_observations(observations).values()))
        self.assertEqual(aggregate.interventional_above_uncertainty_samples, 0)
        self.assertFalse(causal_candidate_domains(observations))

    def test_invalid_route_contract_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            FlowStage(
                stage_id="bad",
                bytes=4096,
                domains=(ResourceDomain.PCIE_HOST, ResourceDomain.PCIE_HOST),
                deadline_ns=100,
            )
        with self.assertRaises(ValueError):
            StateFlow(
                flow_id="bad",
                stages=(
                    FlowStage(
                        stage_id="same",
                        bytes=4096,
                        domains=(ResourceDomain.HOST_NUMA,),
                        deadline_ns=100,
                    ),
                    FlowStage(
                        stage_id="same",
                        bytes=4096,
                        domains=(ResourceDomain.HOST_NUMA,),
                        deadline_ns=100,
                    ),
                ),
                deadline_ns=200,
            )


if __name__ == "__main__":
    unittest.main()
