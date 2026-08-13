from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

from tempo.domain_admission import DomainAdmissionController, DomainBudget
from tempo.kv_flow import KVOperation
from tempo.resource_domain import ResourceDomain
from tempo.torch_kv_backend import TorchKVBackend


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchKVBackendTests(unittest.TestCase):
    PREFETCH_ROUTE = (
        ResourceDomain.PERSISTENT_ENDPOINT,
        ResourceDomain.SLINGSHOT_FABRIC,
        ResourceDomain.NIC_FABRIC,
        ResourceDomain.HOST_NUMA,
        ResourceDomain.PCIE_HOST,
        ResourceDomain.GPU_LOCAL,
    )
    EVICT_ROUTE = tuple(reversed(PREFETCH_ROUTE))

    def controller(self, route):
        return DomainAdmissionController(
            {
                domain: DomainBudget(domain, 1_000_000_000, 1 << 20)
                for domain in route
            },
            catch_up_slack_ns=0,
        )

    def test_prefetch_moves_exact_tensor_and_preserves_version(self) -> None:
        backend = TorchKVBackend()
        tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        version = backend.publish_hbm("session", 0, tensor)
        backend.seed_endpoint(version, tensor)
        backend.drop_hbm("session", version)
        request = backend.make_request(
            request_id="prefetch-0",
            session_id="session",
            operation=KVOperation.PREFETCH,
            route=self.PREFETCH_ROUTE,
            deadline_ns=1_000_000_000,
        )
        decision = backend.admit(
            request,
            self.controller(self.PREFETCH_ROUTE),
            now_ns=0,
            foreground_domains=(ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST),
        )
        self.assertEqual(decision.status, "admitted")
        completion = backend.complete(request.request_id)
        self.assertEqual(completion.version, version)
        self.assertTrue(torch.equal(backend.read_hbm("session"), tensor))

    def test_evict_requires_endpoint_commit_before_hbm_drop(self) -> None:
        backend = TorchKVBackend()
        tensor = torch.arange(16, dtype=torch.float16)
        version = backend.publish_hbm("session", 0, tensor)
        with self.assertRaisesRegex(ValueError, "matching endpoint"):
            backend.drop_hbm("session", version)
        request = backend.make_request(
            request_id="evict-0",
            session_id="session",
            operation=KVOperation.EVICT,
            route=self.EVICT_ROUTE,
            deadline_ns=1_000_000_000,
        )
        decision = backend.admit(request, self.controller(self.EVICT_ROUTE), now_ns=0)
        self.assertEqual(decision.status, "admitted")
        backend.complete(request.request_id)
        backend.drop_hbm("session", version)
        self.assertTrue(torch.equal(backend.read_endpoint("session"), tensor))

    def test_nonmatching_completed_bytes_fail_before_release(self) -> None:
        backend = TorchKVBackend()
        tensor = torch.arange(16, dtype=torch.float32)
        version = backend.publish_hbm("session", 0, tensor)
        backend.seed_endpoint(version, tensor)
        backend.drop_hbm("session", version)
        request = backend.make_request(
            request_id="prefetch-bad",
            session_id="session",
            operation=KVOperation.PREFETCH,
            route=self.PREFETCH_ROUTE,
            deadline_ns=1_000_000_000,
        )
        backend.admit(request, self.controller(self.PREFETCH_ROUTE), now_ns=0)
        with self.assertRaisesRegex(ValueError, "completed tensor bytes"):
            backend.complete(request.request_id, tensor[:1])
        # The caller can cancel after a failed integrity check; the reservation
        # is intentionally not silently released by a malformed completion.
        backend.ledger.cancel(request.request_id)

    def test_prefetch_and_evict_direction_is_explicit(self) -> None:
        backend = TorchKVBackend()
        tensor = torch.ones(8, dtype=torch.float32)
        version = backend.publish_hbm("session", 0, tensor)
        backend.seed_endpoint(version, tensor)
        with self.assertRaisesRegex(ValueError, "prefetch route"):
            backend.make_request(
                request_id="wrong-prefetch",
                session_id="session",
                operation=KVOperation.PREFETCH,
                route=self.EVICT_ROUTE,
                deadline_ns=1_000_000_000,
            )
        with self.assertRaisesRegex(ValueError, "evict route"):
            backend.make_request(
                request_id="wrong-evict",
                session_id="session",
                operation=KVOperation.EVICT,
                route=self.PREFETCH_ROUTE,
                deadline_ns=1_000_000_000,
            )


if __name__ == "__main__":
    unittest.main()
