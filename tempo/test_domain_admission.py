from __future__ import annotations

import unittest

from tempo.domain_admission import (
    DomainAdmissionController,
    DomainBudget,
    DomainRequest,
    FlowAdmissionLedger,
)
from tempo.resource_domain import ResourceDomain


class DomainAdmissionTests(unittest.TestCase):
    def controller(self) -> DomainAdmissionController:
        return DomainAdmissionController(
            {
                ResourceDomain.PCIE_HOST: DomainBudget(ResourceDomain.PCIE_HOST, 1_000_000_000, 4 * 1024 * 1024),
                ResourceDomain.NIC_FABRIC: DomainBudget(ResourceDomain.NIC_FABRIC, 500_000_000, 16 * 1024 * 1024),
            },
            catch_up_slack_ns=2_000_000,
        )

    def request(self, *, request_id: str = "r0", deadline_ns: int = 10_000_000) -> DomainRequest:
        return DomainRequest(
            request_id=request_id,
            flow_id="checkpoint-0",
            bytes=4 * 1024 * 1024,
            route=(ResourceDomain.PCIE_HOST, ResourceDomain.NIC_FABRIC),
            now_ns=0,
            deadline_ns=deadline_ns,
            nonpreemptible_residual_bytes=1 * 1024 * 1024,
        )

    def test_explicit_route_uses_slowest_domain_envelope(self) -> None:
        decision = self.controller().admit(self.request())
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.estimated_completion_ns, 8_388_608)
        self.assertEqual(decision.reason, "catch_up")

    def test_capacity_and_completion_are_exact(self) -> None:
        controller = self.controller()
        self.assertTrue(controller.admit(self.request()).admitted)
        second = controller.admit(self.request(request_id="r1"))
        self.assertFalse(second.admitted)
        self.assertEqual(second.reason, "capacity")
        with self.assertRaisesRegex(ValueError, "equal"):
            controller.complete("r0", 1)
        controller.complete("r0", 4 * 1024 * 1024)
        self.assertEqual(controller.inflight_bytes[ResourceDomain.PCIE_HOST], 0)

    def test_deadline_includes_already_reserved_route_bytes(self) -> None:
        controller = DomainAdmissionController(
            {
                ResourceDomain.PCIE_HOST: DomainBudget(
                    ResourceDomain.PCIE_HOST, 1_000_000_000, 8 * 1024 * 1024
                )
            },
            catch_up_slack_ns=0,
        )
        first = DomainRequest(
            request_id="queued-0",
            flow_id="f",
            bytes=4 * 1024 * 1024,
            route=(ResourceDomain.PCIE_HOST,),
            now_ns=0,
            deadline_ns=10_000_000,
        )
        self.assertTrue(controller.admit(first).admitted)
        second = DomainRequest(
            request_id="queued-1",
            flow_id="f",
            bytes=4 * 1024 * 1024,
            route=(ResourceDomain.PCIE_HOST,),
            now_ns=0,
            deadline_ns=5_000_000,
        )
        decision = controller.admit(second)
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, "deadline")

    def test_deadline_includes_controller_control_overhead(self) -> None:
        controller = DomainAdmissionController(
            {
                ResourceDomain.PCIE_HOST: DomainBudget(
                    ResourceDomain.PCIE_HOST,
                    service_rate_bytes_per_second=1_000_000_000,
                    max_inflight_bytes=10_000_000,
                )
            },
            catch_up_slack_ns=0,
        )
        request = DomainRequest(
            request_id="control-overhead",
            flow_id="flow",
            bytes=100,
            route=(ResourceDomain.PCIE_HOST,),
            now_ns=0,
            deadline_ns=1_000,
            control_overhead_ns=901,
        )
        decision = controller.admit(request)
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, "deadline")
        self.assertEqual(decision.estimated_completion_ns, 1001)

    def test_training_and_kv_requests_share_the_same_pcie_reservation(self) -> None:
        """Shared-domain admission is not a global stop-the-world gate."""
        controller = DomainAdmissionController(
            {
                ResourceDomain.PCIE_HOST: DomainBudget(
                    ResourceDomain.PCIE_HOST, 1_000_000_000, 8 * 1024 * 1024
                ),
                ResourceDomain.NIC_FABRIC: DomainBudget(
                    ResourceDomain.NIC_FABRIC, 1_000_000_000, 8 * 1024 * 1024
                ),
            },
            catch_up_slack_ns=0,
        )
        training = DomainRequest(
            request_id="training-checkpoint",
            flow_id="training",
            bytes=4 * 1024 * 1024,
            route=(ResourceDomain.PCIE_HOST,),
            now_ns=0,
            deadline_ns=20_000_000,
        )
        self.assertTrue(controller.admit(training).admitted)

        kv_shared = DomainRequest(
            request_id="kv-prefetch-shared",
            flow_id="inference",
            bytes=4 * 1024 * 1024,
            route=(ResourceDomain.PCIE_HOST,),
            now_ns=0,
            deadline_ns=5_000_000,
        )
        shared = controller.admit(kv_shared)
        self.assertFalse(shared.admitted)
        self.assertEqual(shared.reason, "deadline")

        disjoint = DomainRequest(
            request_id="kv-prefetch-disjoint",
            flow_id="inference",
            bytes=4 * 1024 * 1024,
            route=(ResourceDomain.NIC_FABRIC,),
            now_ns=0,
            deadline_ns=5_000_000,
        )
        self.assertTrue(controller.admit(disjoint).admitted)

    def test_unknown_domain_and_deadline_fail_closed(self) -> None:
        unsupported = DomainRequest(
            request_id="r-unsupported",
            flow_id="f",
            bytes=1,
            route=(ResourceDomain.NVLINK_P2P,),
            now_ns=0,
            deadline_ns=10,
        )
        decision = self.controller().admit(unsupported)
        self.assertFalse(decision.admitted)
        self.assertIn("unsupported", decision.reason)
        late = self.controller().admit(self.request(request_id="late", deadline_ns=1))
        self.assertFalse(late.admitted)
        self.assertEqual(late.reason, "deadline")

    def test_tail_budget_is_separate_from_absolute_flow_deadline(self) -> None:
        request = DomainRequest(
            request_id="tail-limited",
            flow_id="training",
            bytes=4 * 1024 * 1024,
            route=(ResourceDomain.PCIE_HOST,),
            now_ns=0,
            deadline_ns=20_000_000,
            tail_budget_ns=3_000_000,
        )
        decision = self.controller().admit(request)
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, "tail_budget")
        self.assertEqual(decision.estimated_completion_ns, 4_194_304)

    def test_explicit_foreground_footprint_reports_overlap_and_gates_only_shared_route(self) -> None:
        controller = self.controller()
        shared = DomainRequest(
            request_id="shared-footprint",
            flow_id="training",
            bytes=4 * 1024 * 1024,
            route=(ResourceDomain.PCIE_HOST, ResourceDomain.NIC_FABRIC),
            now_ns=0,
            deadline_ns=20_000_000,
            tail_budget_ns=3_000_000,
            foreground_domains=(ResourceDomain.PCIE_HOST,),
        )
        rejected = controller.admit(shared)
        self.assertFalse(rejected.admitted)
        self.assertEqual(rejected.reason, "tail_budget")
        self.assertEqual(rejected.shared_domains, (ResourceDomain.PCIE_HOST,))

        disjoint = DomainRequest(
            request_id="disjoint-footprint",
            flow_id="inference",
            bytes=4 * 1024 * 1024,
            route=(ResourceDomain.PCIE_HOST,),
            now_ns=0,
            deadline_ns=20_000_000,
            tail_budget_ns=3_000_000,
            foreground_domains=(ResourceDomain.NIC_FABRIC,),
        )
        admitted = controller.admit(disjoint)
        self.assertTrue(admitted.admitted)
        self.assertEqual(admitted.shared_domains, ())

    def test_unknown_foreground_footprint_remains_conservative(self) -> None:
        request = self.request(deadline_ns=20_000_000)
        request = DomainRequest(
            request_id=request.request_id,
            flow_id=request.flow_id,
            bytes=request.bytes,
            route=request.route,
            now_ns=request.now_ns,
            deadline_ns=request.deadline_ns,
            nonpreemptible_residual_bytes=request.nonpreemptible_residual_bytes,
            tail_budget_ns=3_000_000,
        )
        decision = self.controller().admit(request)
        self.assertFalse(decision.admitted)
        self.assertIsNone(decision.shared_domains)
        self.assertEqual(decision.reason, "tail_budget")

    def test_positive_minimum_service_is_the_conservative_deadline_rate(self) -> None:
        controller = DomainAdmissionController(
            {
                ResourceDomain.PCIE_HOST: DomainBudget(
                    ResourceDomain.PCIE_HOST,
                    1_000_000_000,
                    8 * 1024 * 1024,
                    minimum_service_bytes_per_second=500_000_000,
                )
            },
            catch_up_slack_ns=0,
        )
        decision = controller.admit(
            DomainRequest(
                request_id="conservative",
                flow_id="f",
                bytes=4 * 1024 * 1024,
                route=(ResourceDomain.PCIE_HOST,),
                now_ns=0,
                deadline_ns=8_500_000,
            )
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.estimated_completion_ns, 8_388_608)

    def test_residual_is_exposed_and_cannot_exceed_domain_capacity(self) -> None:
        controller = self.controller()
        request = self.request()
        decision = controller.admit(request)
        self.assertEqual(decision.nonpreemptible_residual_bytes, 1 * 1024 * 1024)
        self.assertEqual(controller.active_residual_bytes[request.request_id], 1 * 1024 * 1024)
        controller.complete(request.request_id, request.bytes)
        self.assertEqual(controller.active_residual_bytes, {})

        too_large = DomainRequest(
            request_id="too-large-residual",
            flow_id="f",
            bytes=8 * 1024 * 1024,
            route=(ResourceDomain.PCIE_HOST, ResourceDomain.NIC_FABRIC),
            now_ns=0,
            deadline_ns=20_000_000,
            nonpreemptible_residual_bytes=5 * 1024 * 1024,
        )
        rejected = controller.admit(too_large)
        self.assertFalse(rejected.admitted)
        self.assertEqual(rejected.reason, "residual")

    def test_shared_flow_ledger_has_exact_once_release(self) -> None:
        controller = self.controller()
        ledger = FlowAdmissionLedger(controller)
        request = self.request(request_id="ledger-request")
        decision = ledger.admit(request)
        self.assertTrue(decision.admitted)
        self.assertTrue(ledger.is_active(request.request_id))
        with self.assertRaisesRegex(ValueError, "equal"):
            ledger.complete(request.request_id, request.bytes - 1)
        self.assertTrue(ledger.is_active(request.request_id))
        ledger.complete(request.request_id, request.bytes)
        self.assertFalse(ledger.is_active(request.request_id))
        with self.assertRaisesRegex(ValueError, "unknown"):
            ledger.complete(request.request_id, request.bytes)

    def test_cancel_releases_route_reservation_without_completion(self) -> None:
        controller = self.controller()
        request = self.request(request_id="cancel-request")
        self.assertTrue(controller.admit(request).admitted)
        controller.cancel(request.request_id)
        self.assertEqual(controller.inflight_bytes[ResourceDomain.PCIE_HOST], 0)
        self.assertEqual(controller.active_residual_bytes, {})
        with self.assertRaisesRegex(ValueError, "unknown"):
            controller.cancel(request.request_id)


if __name__ == "__main__":
    unittest.main()
