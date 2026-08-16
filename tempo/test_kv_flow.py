from __future__ import annotations

import unittest

from tempo.kv_flow import KVFlowLedger, KVOperation, KVTransferRequest, KVVersion
from tempo.domain_admission import (
    DomainAdmissionController,
    DomainBudget,
    DomainRequest,
    FlowAdmissionLedger,
)
from tempo.resource_domain import ResourceDomain


class KVFlowTests(unittest.TestCase):
    def request(
        self,
        version: KVVersion,
        *,
        deadline_ns: int = 2_000_000_000,
        request_id: str | None = None,
    ) -> KVTransferRequest:
        return KVTransferRequest(
            request_id=request_id or f"req-{version.sequence}",
            version=version,
            operation=KVOperation.PREFETCH,
            bytes=4 * 1024 * 1024,
            source=ResourceDomain.HOST_NUMA,
            destination=ResourceDomain.GPU_LOCAL,
            route=(ResourceDomain.HOST_NUMA, ResourceDomain.PCIE_HOST, ResourceDomain.GPU_LOCAL),
            deadline_ns=deadline_ns,
            max_residual_bytes=1 * 1024 * 1024,
        )

    def test_version_and_route_are_explicit(self) -> None:
        version = KVVersion.from_bytes("session-a", 0, b"kv")
        ledger = KVFlowLedger()
        ledger.publish(version)
        request = self.request(version)
        decision = ledger.admit(
            request,
            now_ns=0,
            service_rate_bytes_per_second=1_000_000_000,
            available_bytes=request.bytes,
        )
        self.assertEqual(decision.status, "admitted")
        ledger.complete(request.request_id, request.bytes)
        self.assertTrue(ledger.is_complete(request.request_id))

    def test_version_and_request_identity_types_are_strict(self) -> None:
        with self.assertRaises(ValueError):
            KVVersion(1, 0, "a" * 64)
        with self.assertRaises(ValueError):
            KVVersion("session", 0, 1)  # type: ignore[arg-type]
        version = KVVersion.from_bytes("session-strict", 0, b"kv")
        with self.assertRaises(ValueError):
            KVTransferRequest(
                request_id=1,  # type: ignore[arg-type]
                version=version,
                operation=KVOperation.PREFETCH,
                bytes=1,
                source=ResourceDomain.HOST_NUMA,
                destination=ResourceDomain.GPU_LOCAL,
                route=(ResourceDomain.HOST_NUMA, ResourceDomain.GPU_LOCAL),
                deadline_ns=10,
            )
        with self.assertRaises(TypeError):
            KVTransferRequest(
                request_id="bad-version",
                version=object(),  # type: ignore[arg-type]
                operation=KVOperation.PREFETCH,
                bytes=1,
                source=ResourceDomain.HOST_NUMA,
                destination=ResourceDomain.GPU_LOCAL,
                route=(ResourceDomain.HOST_NUMA, ResourceDomain.GPU_LOCAL),
                deadline_ns=10,
            )

    def test_stale_version_is_rejected(self) -> None:
        ledger = KVFlowLedger()
        old = KVVersion.from_bytes("session-a", 0, b"old")
        new = KVVersion.from_bytes("session-a", 1, b"new")
        ledger.publish(old)
        ledger.publish(new)
        with self.assertRaisesRegex(ValueError, "stale"):
            ledger.admit(
                self.request(old),
                now_ns=0,
                service_rate_bytes_per_second=1_000_000_000,
                available_bytes=8 * 1024 * 1024,
            )

    def test_inflight_stale_completion_is_rejected_and_releases_capacity(self) -> None:
        ledger = KVFlowLedger()
        old = KVVersion.from_bytes("session-inflight", 0, b"old")
        new = KVVersion.from_bytes("session-inflight", 1, b"new")
        ledger.publish(old)
        request = self.request(old)
        controller = DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 8 * 1024 * 1024)
                for domain in request.route
            },
            catch_up_slack_ns=1,
        )
        decision = ledger.admit_via_domain_controller(request, controller, now_ns=0)
        self.assertEqual(decision.status, "admitted")
        ledger.publish(new)
        with self.assertRaisesRegex(ValueError, "stale or superseded"):
            ledger.complete(request.request_id, request.bytes)
        self.assertTrue(all(value == 0 for value in controller.inflight_bytes.values()))
        self.assertFalse(ledger.is_complete(request.request_id))

    def test_deadline_and_capacity_are_fail_closed(self) -> None:
        version = KVVersion.from_bytes("session-a", 0, b"kv")
        ledger = KVFlowLedger()
        ledger.publish(version)
        request = self.request(version, deadline_ns=1)
        deadline = ledger.admit(
            request,
            now_ns=0,
            service_rate_bytes_per_second=1_000_000,
            available_bytes=request.bytes,
        )
        self.assertEqual((deadline.status, deadline.reason), ("rejected", "deadline"))
        capacity = ledger.admit(
            KVTransferRequest(
                request_id="capacity",
                version=version,
                operation=KVOperation.PREFETCH,
                bytes=request.bytes,
                source=request.source,
                destination=request.destination,
                route=request.route,
                deadline_ns=2_000_000_000,
            ),
            now_ns=0,
            service_rate_bytes_per_second=1_000_000_000,
            available_bytes=0,
        )
        self.assertEqual((capacity.status, capacity.reason), ("rejected", "capacity"))

    def test_partial_completion_is_not_accepted_as_success(self) -> None:
        version = KVVersion.from_bytes("session-a", 0, b"kv")
        ledger = KVFlowLedger()
        ledger.publish(version)
        request = self.request(version)
        ledger.admit(
            request,
            now_ns=0,
            service_rate_bytes_per_second=1_000_000_000,
            available_bytes=request.bytes,
        )
        with self.assertRaisesRegex(ValueError, "exactly"):
            ledger.complete(request.request_id, request.bytes - 1)

    def test_kv_can_use_the_shared_domain_controller(self) -> None:
        version = KVVersion.from_bytes("session-a", 0, b"kv")
        ledger = KVFlowLedger()
        ledger.publish(version)
        controller = DomainAdmissionController(
            {
                ResourceDomain.HOST_NUMA: DomainBudget(ResourceDomain.HOST_NUMA, 1_000_000_000, 8 * 1024 * 1024),
                ResourceDomain.PCIE_HOST: DomainBudget(ResourceDomain.PCIE_HOST, 1_000_000_000, 8 * 1024 * 1024),
                ResourceDomain.GPU_LOCAL: DomainBudget(ResourceDomain.GPU_LOCAL, 1_000_000_000, 8 * 1024 * 1024),
            },
            catch_up_slack_ns=1,
        )
        decision = ledger.admit_via_domain_controller(self.request(version), controller, now_ns=0)
        self.assertEqual(decision.status, "admitted")
        ledger.complete(decision.request_id, decision.admitted_bytes)
        self.assertEqual(
            controller.inflight_bytes[ResourceDomain.HOST_NUMA],
            0,
        )
        self.assertTrue(ledger.is_complete(decision.request_id))

    def test_kv_controller_overhead_is_part_of_shared_deadline(self) -> None:
        version = KVVersion.from_bytes("session-control-cost", 0, b"kv")
        ledger = KVFlowLedger()
        ledger.publish(version)
        controller = DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 8 * 1024 * 1024)
                for domain in self.request(version).route
            },
            catch_up_slack_ns=0,
        )
        decision = ledger.admit_via_domain_controller(
            self.request(version),
            controller,
            now_ns=0,
            control_overhead_ns=2_000_000_000,
        )
        self.assertEqual(decision.status, "rejected")
        self.assertEqual(decision.reason, "deadline")

    def test_training_and_kv_use_the_same_route_envelope(self) -> None:
        version = KVVersion.from_bytes("session-shared", 0, b"kv")
        ledger = KVFlowLedger()
        ledger.publish(version)
        budgets = {
            ResourceDomain.HOST_NUMA: DomainBudget(ResourceDomain.HOST_NUMA, 1_000_000_000, 8 * 1024 * 1024),
            ResourceDomain.PCIE_HOST: DomainBudget(ResourceDomain.PCIE_HOST, 1_000_000_000, 8 * 1024 * 1024),
            ResourceDomain.GPU_LOCAL: DomainBudget(ResourceDomain.GPU_LOCAL, 1_000_000_000, 8 * 1024 * 1024),
        }
        kv_controller = DomainAdmissionController(budgets, catch_up_slack_ns=1)
        training_controller = DomainAdmissionController(budgets, catch_up_slack_ns=1)
        kv_decision = ledger.admit_via_domain_controller(
            self.request(version), kv_controller, now_ns=0
        )
        training_ledger = FlowAdmissionLedger(training_controller)
        training_decision = training_ledger.admit(
            DomainRequest(
                request_id="training-request",
                flow_id="checkpoint-0",
                bytes=4 * 1024 * 1024,
                route=self.request(version).route,
                now_ns=0,
                deadline_ns=2_000_000_000,
                nonpreemptible_residual_bytes=1 * 1024 * 1024,
            )
        )
        self.assertTrue(kv_decision.status == "admitted" and training_decision.admitted)
        self.assertEqual(kv_decision.admitted_bytes, training_decision.admitted_bytes)
        self.assertEqual(kv_decision.estimated_service_ns, max(training_decision.per_domain_service_ns.values()))
        ledger.complete(kv_decision.request_id, kv_decision.admitted_bytes)
        training_ledger.complete("training-request", training_decision.admitted_bytes)
        self.assertEqual(dict(kv_controller.inflight_bytes), dict(training_controller.inflight_bytes))

    def test_controller_owned_completion_is_exactly_once(self) -> None:
        version = KVVersion.from_bytes("session-once", 0, b"kv")
        ledger = KVFlowLedger()
        ledger.publish(version)
        controller = DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 8 * 1024 * 1024)
                for domain in self.request(version).route
            },
            catch_up_slack_ns=1,
        )
        request = self.request(version)
        decision = ledger.admit_via_domain_controller(request, controller, now_ns=0)
        ledger.complete(decision.request_id, decision.admitted_bytes)
        with self.assertRaisesRegex(ValueError, "already complete"):
            ledger.complete(decision.request_id, decision.admitted_bytes)
        self.assertTrue(all(value == 0 for value in controller.inflight_bytes.values()))

    def test_controller_partial_completion_keeps_reservation(self) -> None:
        version = KVVersion.from_bytes("session-atomic", 0, b"kv")
        ledger = KVFlowLedger()
        ledger.publish(version)
        request = self.request(version)
        controller = DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 8 * 1024 * 1024)
                for domain in request.route
            },
            catch_up_slack_ns=1,
        )
        decision = ledger.admit_via_domain_controller(request, controller, now_ns=0)
        with self.assertRaisesRegex(ValueError, "exactly"):
            ledger.complete(decision.request_id, decision.admitted_bytes - 1)
        self.assertTrue(all(value == request.bytes for value in controller.inflight_bytes.values()))
        self.assertFalse(ledger.is_complete(request.request_id))
        ledger.complete(decision.request_id, decision.admitted_bytes)
        self.assertTrue(all(value == 0 for value in controller.inflight_bytes.values()))

    def test_controller_cancel_releases_reservation_for_retry(self) -> None:
        version = KVVersion.from_bytes("session-cancel", 0, b"cancel")
        ledger = KVFlowLedger()
        ledger.publish(version)
        request = self.request(version, request_id="cancel-request")
        controller = DomainAdmissionController(
            {domain: DomainBudget(domain, 1_000_000_000, 8 * 1024 * 1024) for domain in request.route},
            catch_up_slack_ns=0,
        )
        decision = ledger.admit_via_domain_controller(request, controller, now_ns=0)
        self.assertEqual(decision.status, "admitted")
        ledger.cancel(request.request_id)
        self.assertTrue(all(value == 0 for value in controller.inflight_bytes.values()))
        retry = ledger.admit_via_domain_controller(
            self.request(version, request_id="cancel-retry"), controller, now_ns=1_000
        )
        self.assertEqual(retry.status, "admitted")

    def test_full_fabric_persistent_route_uses_shared_controller(self) -> None:
        """A KV persistent route reserves each declared fabric/storage domain."""
        version = KVVersion.from_bytes("session-fabric", 0, b"kv-fabric")
        ledger = KVFlowLedger()
        ledger.publish(version)
        request = KVTransferRequest(
            request_id="kv-persistent-prefetch",
            version=version,
            operation=KVOperation.PREFETCH,
            bytes=4 * 1024 * 1024,
            source=ResourceDomain.GPU_LOCAL,
            destination=ResourceDomain.PERSISTENT_ENDPOINT,
            route=(
                ResourceDomain.GPU_LOCAL,
                ResourceDomain.PCIE_HOST,
                ResourceDomain.HOST_NUMA,
                ResourceDomain.NIC_FABRIC,
                ResourceDomain.SLINGSHOT_FABRIC,
                ResourceDomain.PERSISTENT_ENDPOINT,
            ),
            deadline_ns=20_000_000,
            max_residual_bytes=1 * 1024 * 1024,
        )
        controller = DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 8 * 1024 * 1024)
                for domain in request.route
            },
            catch_up_slack_ns=0,
        )
        decision = ledger.admit_via_domain_controller(request, controller, now_ns=0)
        self.assertEqual(decision.status, "admitted")
        self.assertEqual(decision.admitted_bytes, request.bytes)
        self.assertTrue(
            all(controller.inflight_bytes[domain] == request.bytes for domain in request.route)
        )
        ledger.complete(request.request_id, request.bytes)
        self.assertTrue(all(value == 0 for value in controller.inflight_bytes.values()))

    def test_kv_tail_slo_uses_the_same_budget_as_training(self) -> None:
        version = KVVersion.from_bytes("session-tail", 0, b"kv-tail")
        ledger = KVFlowLedger()
        ledger.publish(version)
        request = KVTransferRequest(
            request_id="kv-tail-limited",
            version=version,
            operation=KVOperation.PREFETCH,
            bytes=4 * 1024 * 1024,
            source=ResourceDomain.HOST_NUMA,
            destination=ResourceDomain.GPU_LOCAL,
            route=(ResourceDomain.HOST_NUMA, ResourceDomain.PCIE_HOST, ResourceDomain.GPU_LOCAL),
            deadline_ns=20_000_000,
            tail_budget_ns=3_000_000,
        )
        controller = DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 8 * 1024 * 1024)
                for domain in request.route
            },
            catch_up_slack_ns=0,
        )
        decision = ledger.admit_via_domain_controller(request, controller, now_ns=0)
        self.assertEqual((decision.status, decision.reason), ("rejected", "tail_budget"))
        self.assertTrue(all(value == 0 for value in controller.inflight_bytes.values()))

    def test_kv_controller_records_explicit_foreground_overlap(self) -> None:
        version = KVVersion.from_bytes("session-footprint", 0, b"kv-footprint")
        ledger = KVFlowLedger()
        ledger.publish(version)
        request = self.request(version, deadline_ns=20_000_000)
        controller = DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 8 * 1024 * 1024)
                for domain in request.route
            },
            catch_up_slack_ns=0,
        )
        decision = ledger.admit_via_domain_controller(
            request,
            controller,
            now_ns=0,
            foreground_domains=(ResourceDomain.PCIE_HOST,),
        )
        self.assertTrue(decision.status == "admitted")
        self.assertEqual(decision.shared_domains, (ResourceDomain.PCIE_HOST,))
        # The public KV decision preserves identity/status; the shared
        # controller owns the explicit overlap in its request ledger.
        ledger.complete(request.request_id, request.bytes)


if __name__ == "__main__":
    unittest.main()
