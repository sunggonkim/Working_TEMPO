from __future__ import annotations

import unittest

from tempo.domain_admission import DomainAdmissionController, DomainBudget
from tempo.flow_adapters import (
    StateFlowAdmission,
    checkpoint_state_flow,
    flow_route_signature,
    kv_state_flow,
)
from tempo.kv_flow import KVFlowLedger, KVOperation, KVTransferRequest, KVVersion
from tempo.resource_domain import ResourceDomain


class FlowAdapterTests(unittest.TestCase):
    def controller(self, domains):
        return DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 16 * 1024 * 1024)
                for domain in domains
            },
            catch_up_slack_ns=0,
        )

    def test_checkpoint_adapter_preserves_two_stage_bytes_and_routes(self) -> None:
        flow = checkpoint_state_flow(
            flow_id="checkpoint:event32:rank0",
            state_bytes=64 * 1024 * 1024,
            deadline_ns=1_000_000_000,
            d2h_deadline_ns=200_000_000,
            persist_deadline_ns=1_000_000_000,
            d2h_residual_bytes=1 * 1024 * 1024,
            persist_residual_bytes=16 * 1024 * 1024,
        )
        self.assertEqual([stage.bytes for stage in flow.stages], [64 * 1024 * 1024] * 2)
        self.assertEqual(flow.domains, frozenset({
            ResourceDomain.GPU_LOCAL,
            ResourceDomain.PCIE_HOST,
            ResourceDomain.HOST_NUMA,
            ResourceDomain.NIC_FABRIC,
            ResourceDomain.SLINGSHOT_FABRIC,
            ResourceDomain.PERSISTENT_ENDPOINT,
        }))
        self.assertEqual(flow.total_bytes, 128 * 1024 * 1024)

    def test_control_overhead_is_preserved_for_checkpoint_stage_admission(self) -> None:
        flow = checkpoint_state_flow(
            flow_id="checkpoint:controller-cost",
            state_bytes=100,
            deadline_ns=1_000,
            d2h_deadline_ns=1_000,
            persist_deadline_ns=1_000,
            d2h_domains=(ResourceDomain.PCIE_HOST,),
            persist_domains=(ResourceDomain.PERSISTENT_ENDPOINT,),
            d2h_control_overhead_ns=901,
        )
        admission = StateFlowAdmission(
            flow,
            self.controller((ResourceDomain.PCIE_HOST, ResourceDomain.PERSISTENT_ENDPOINT)),
        )
        decision = admission.admit_next(now_ns=0)
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, "deadline")
        self.assertEqual(decision.estimated_completion_ns, 1001)

    def test_checkpoint_adapter_does_not_infer_nvlink(self) -> None:
        flow = checkpoint_state_flow(
            flow_id="checkpoint:host-only",
            state_bytes=4096,
            deadline_ns=1000,
            d2h_deadline_ns=500,
            persist_deadline_ns=1000,
            d2h_domains=(ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA),
            persist_domains=(ResourceDomain.PERSISTENT_ENDPOINT,),
        )
        self.assertNotIn(ResourceDomain.NVLINK_P2P, flow.domains)

    def test_checkpoint_and_kv_can_describe_the_same_full_route_domains(self) -> None:
        """The shared abstraction covers a declared multi-hop route, not topology inference."""

        checkpoint = checkpoint_state_flow(
            flow_id="checkpoint:full-route-contract",
            state_bytes=4096,
            deadline_ns=1_000_000,
            d2h_deadline_ns=500_000,
            persist_deadline_ns=1_000_000,
            d2h_domains=(
                ResourceDomain.GPU_LOCAL,
                ResourceDomain.PCIE_HOST,
                ResourceDomain.HOST_NUMA,
                ResourceDomain.NIC_FABRIC,
                ResourceDomain.SLINGSHOT_FABRIC,
            ),
            persist_domains=(ResourceDomain.PERSISTENT_ENDPOINT,),
        )
        version = KVVersion.from_bytes("session-full-route", 0, b"kv")
        kv = kv_state_flow(KVTransferRequest(
            request_id="kv-full-route",
            version=version,
            operation=KVOperation.PREFETCH,
            bytes=4096,
            source=ResourceDomain.PERSISTENT_ENDPOINT,
            destination=ResourceDomain.GPU_LOCAL,
            route=(
                ResourceDomain.PERSISTENT_ENDPOINT,
                ResourceDomain.SLINGSHOT_FABRIC,
                ResourceDomain.NIC_FABRIC,
                ResourceDomain.HOST_NUMA,
                ResourceDomain.PCIE_HOST,
                ResourceDomain.GPU_LOCAL,
            ),
            deadline_ns=1_000_000,
        ))
        self.assertEqual(checkpoint.domains, kv.domains)
        self.assertEqual(
            checkpoint.domains,
            frozenset({
                ResourceDomain.GPU_LOCAL,
                ResourceDomain.PCIE_HOST,
                ResourceDomain.HOST_NUMA,
                ResourceDomain.NIC_FABRIC,
                ResourceDomain.SLINGSHOT_FABRIC,
                ResourceDomain.PERSISTENT_ENDPOINT,
            }),
        )

    def test_kv_adapter_preserves_version_and_route(self) -> None:
        version = KVVersion.from_bytes("session-a", 4, b"kv")
        request = KVTransferRequest(
            request_id="prefetch-4",
            version=version,
            operation=KVOperation.PREFETCH,
            bytes=4096,
            source=ResourceDomain.GPU_LOCAL,
            destination=ResourceDomain.HOST_NUMA,
            route=(ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA),
            deadline_ns=1_000_000,
            max_residual_bytes=1024,
        )
        flow = kv_state_flow(request)
        self.assertEqual(flow.version, f"session-a:4:{version.content_digest}")
        self.assertEqual(flow.total_bytes, 4096)
        self.assertEqual(
            flow_route_signature(flow),
            (("kv:prefetch", ("gpu_local", "pcie_host", "host_numa")),),
        )

    def test_kv_and_checkpoint_share_route_signature_contract(self) -> None:
        checkpoint = checkpoint_state_flow(
            flow_id="checkpoint:one-stage",
            state_bytes=4096,
            deadline_ns=1_000_000,
            d2h_deadline_ns=500_000,
            persist_deadline_ns=1_000_000,
            d2h_domains=(ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA),
            persist_domains=(ResourceDomain.PERSISTENT_ENDPOINT,),
        )
        version = KVVersion.from_bytes("session-shared", 0, b"kv")
        kv = kv_state_flow(KVTransferRequest(
            request_id="kv-shared",
            version=version,
            operation=KVOperation.PREFETCH,
            bytes=4096,
            source=ResourceDomain.GPU_LOCAL,
            destination=ResourceDomain.HOST_NUMA,
            route=(ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA),
            deadline_ns=1_000_000,
        ))
        self.assertEqual(
            flow_route_signature(checkpoint)[0][1],
            flow_route_signature(kv)[0][1],
        )

    def test_stage_deadline_and_residual_contracts_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            checkpoint_state_flow(
                flow_id="bad",
                state_bytes=4096,
                deadline_ns=100,
                d2h_deadline_ns=101,
                persist_deadline_ns=100,
            )
        with self.assertRaisesRegex(ValueError, "within state_bytes"):
            checkpoint_state_flow(
                flow_id="bad-residual",
                state_bytes=4096,
                deadline_ns=100,
                d2h_deadline_ns=50,
                persist_deadline_ns=100,
                d2h_residual_bytes=4097,
            )

    def test_checkpoint_and_kv_use_the_same_ordered_stage_admission(self) -> None:
        route = (ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA)
        checkpoint = checkpoint_state_flow(
            flow_id="checkpoint:admission",
            state_bytes=4 * 1024 * 1024,
            deadline_ns=20_000_000,
            d2h_deadline_ns=20_000_000,
            persist_deadline_ns=20_000_000,
            d2h_domains=route,
            persist_domains=(ResourceDomain.PERSISTENT_ENDPOINT,),
            d2h_tail_budget_ns=3_000_000,
        )
        checkpoint_admission = StateFlowAdmission(
            checkpoint,
            self.controller(route + (ResourceDomain.PERSISTENT_ENDPOINT,)),
        )
        rejected = checkpoint_admission.admit_next(now_ns=0)
        self.assertEqual((rejected.admitted, rejected.reason), (False, "tail_budget"))
        self.assertEqual(checkpoint_admission.completed_stages, 0)

        version = KVVersion.from_bytes("session-admission", 0, b"kv")
        kv = kv_state_flow(KVTransferRequest(
            request_id="kv-admission",
            version=version,
            operation=KVOperation.PREFETCH,
            bytes=4 * 1024 * 1024,
            source=ResourceDomain.PCIE_HOST,
            destination=ResourceDomain.HOST_NUMA,
            route=route,
            deadline_ns=20_000_000,
            tail_budget_ns=3_000_000,
        ))
        kv_admission = StateFlowAdmission(kv, self.controller(route))
        kv_rejected = kv_admission.admit_next(now_ns=0)
        self.assertEqual((kv_rejected.admitted, kv_rejected.reason), (False, "tail_budget"))

    def test_checkpoint_and_kv_share_live_domain_capacity(self) -> None:
        """A common domain cap must constrain both adapters, not just training."""

        route = (ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA)
        controller = DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 4096)
                for domain in route + (ResourceDomain.PERSISTENT_ENDPOINT,)
            },
            catch_up_slack_ns=0,
        )
        checkpoint = checkpoint_state_flow(
            flow_id="checkpoint:shared-capacity",
            state_bytes=4096,
            deadline_ns=1_000_000_000,
            d2h_deadline_ns=1_000_000_000,
            persist_deadline_ns=1_000_000_000,
            d2h_domains=route,
            persist_domains=(ResourceDomain.PERSISTENT_ENDPOINT,),
        )
        checkpoint_admission = StateFlowAdmission(checkpoint, controller)
        checkpoint_decision = checkpoint_admission.admit_next(now_ns=0)
        self.assertTrue(checkpoint_decision.admitted)

        version = KVVersion.from_bytes("session-shared-cap", 0, b"kv")
        request = KVTransferRequest(
            request_id="kv-shared-capacity",
            version=version,
            operation=KVOperation.PREFETCH,
            bytes=1024,
            source=ResourceDomain.PCIE_HOST,
            destination=ResourceDomain.HOST_NUMA,
            route=route,
            deadline_ns=1_000_000_000,
        )
        kv = KVFlowLedger()
        kv.publish(version)
        blocked = kv.admit_via_domain_controller(
            request, controller, now_ns=0, foreground_domains=route
        )
        self.assertEqual((blocked.status, blocked.reason), ("rejected", "capacity"))
        self.assertEqual(blocked.shared_domains, route)

        checkpoint_admission.complete_active(4096)
        admitted = kv.admit_via_domain_controller(
            request, controller, now_ns=10_000, foreground_domains=route
        )
        self.assertEqual((admitted.status, admitted.reason), ("admitted", "open"))
        self.assertEqual(admitted.shared_domains, route)
        kv.complete(request.request_id, request.bytes)

    def test_state_flow_admission_passes_foreground_footprint_to_shared_controller(self) -> None:
        route = (ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA)
        flow = checkpoint_state_flow(
            flow_id="checkpoint:footprint",
            state_bytes=4 * 1024 * 1024,
            deadline_ns=20_000_000,
            d2h_deadline_ns=20_000_000,
            persist_deadline_ns=20_000_000,
            d2h_domains=route,
            persist_domains=(ResourceDomain.PERSISTENT_ENDPOINT,),
            d2h_tail_budget_ns=3_000_000,
        )
        admission = StateFlowAdmission(
            flow,
            self.controller(route + (ResourceDomain.PERSISTENT_ENDPOINT,)),
            foreground_domains=(ResourceDomain.PCIE_HOST,),
        )
        decision = admission.admit_next(now_ns=0)
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.shared_domains, (ResourceDomain.PCIE_HOST,))

    def test_state_flow_admission_requires_exact_sequential_completion(self) -> None:
        flow = checkpoint_state_flow(
            flow_id="checkpoint:sequential",
            state_bytes=1024,
            deadline_ns=10_000_000,
            d2h_deadline_ns=10_000_000,
            persist_deadline_ns=10_000_000,
            d2h_domains=(ResourceDomain.PCIE_HOST,),
            persist_domains=(ResourceDomain.PERSISTENT_ENDPOINT,),
        )
        admission = StateFlowAdmission(
            flow,
            self.controller((ResourceDomain.PCIE_HOST, ResourceDomain.PERSISTENT_ENDPOINT)),
        )
        first = admission.admit_next(now_ns=0)
        self.assertTrue(first.admitted)
        with self.assertRaisesRegex(ValueError, "previous stage"):
            admission.admit_next(now_ns=1)
        with self.assertRaisesRegex(ValueError, "equal"):
            admission.complete_active(1023)
        admission.complete_active(1024)
        second = admission.admit_next(now_ns=2_000)
        self.assertTrue(second.admitted)
        admission.complete_active(1024)
        self.assertEqual(admission.completed_stages, 2)

    def test_abort_active_releases_domains_and_keeps_stage_uncompleted(self) -> None:
        route = (ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA)
        flow = checkpoint_state_flow(
            flow_id="checkpoint:abort",
            state_bytes=1024,
            deadline_ns=10_000_000,
            d2h_deadline_ns=10_000_000,
            persist_deadline_ns=10_000_000,
            d2h_domains=route,
            persist_domains=(ResourceDomain.PERSISTENT_ENDPOINT,),
        )
        controller = self.controller(route + (ResourceDomain.PERSISTENT_ENDPOINT,))
        admission = StateFlowAdmission(flow, controller)
        self.assertTrue(admission.admit_next(now_ns=0).admitted)
        admission.abort_active()
        self.assertEqual(admission.completed_stages, 0)
        self.assertTrue(admission.admit_next(now_ns=1_000).admitted)


if __name__ == "__main__":
    unittest.main()
