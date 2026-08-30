import unittest

from tempo.pd_elastic_controller_v442 import (
    CacheResidency,
    ElasticConfig,
    ElasticEstimate,
    ElasticPDController,
    ElasticPhase,
    ElasticRegime,
    ElasticRequest,
    ElasticRoute,
)


def estimate(local=20, remote=40, *, deadline=200, tbt=True,
             backend=True, evidence=True, uncertainty=1):
    return ElasticEstimate(
        local_upper_bound_ms=local,
        remote_upper_bound_ms=remote,
        uncertainty_ms=uncertainty,
        remaining_deadline_ms=deadline,
        local_tbt_safe=tbt,
        remote_backend_available=backend,
        remote_evidence_valid=evidence,
    )


def request(name, now, *, residency=CacheResidency.MISS,
            local_cost=2, remote_bytes=64):
    return ElasticRequest(
        request_id=name,
        arrival_ns=now,
        cache_residency=residency,
        local_compute_cost_us=local_cost,
        remote_kv_bytes=remote_bytes,
    )


class ElasticPDControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = ElasticPDController(ElasticConfig(
            local_compute_budget_us=6,
            remote_kv_budget_bytes=128,
            exit_consecutive_windows=2,
        ))

    def test_high_load_uses_weighted_local_credit_then_queues(self):
        now = 0
        decisions = []
        for index in range(5):
            decisions.append(self.controller.submit(
                request(f"r{index}", now), estimate()))
            now += 10_000_000
        self.assertEqual(self.controller.regime, ElasticRegime.DEFLECT_ACTIVE)
        local = [d for d in decisions if d.route is ElasticRoute.LOCAL]
        self.assertEqual(len(local), 3)
        self.assertEqual(self.controller.local_compute_used_us, 6)
        self.assertEqual(decisions[-1].route, ElasticRoute.QUEUE)
        self.assertEqual(
            decisions[-1].reason, "local_credit_exhausted_remote_too_costly")

        self.controller.mark_started(local[0].request_id)
        self.controller.complete(local[0].request_id)
        admitted = self.controller.retry(decisions[-1].request_id, estimate())
        self.assertEqual(admitted.route, ElasticRoute.LOCAL)
        self.assertEqual(admitted.attempt, 2)

    def test_remote_bytes_are_reserved_and_prestart_fallback_is_one_way(self):
        remote_estimate = estimate(local=50, remote=10)
        first = self.controller.submit(
            request("remote0", 0, residency=CacheResidency.P_ONLY),
            remote_estimate)
        second = self.controller.submit(
            request("remote1", 100_000_000, residency=CacheResidency.P_ONLY),
            remote_estimate)
        third = self.controller.submit(
            request("remote2", 200_000_000, residency=CacheResidency.P_ONLY),
            remote_estimate)
        self.assertEqual(first.route, ElasticRoute.REMOTE)
        self.assertEqual(second.route, ElasticRoute.REMOTE)
        self.assertEqual(third.route, ElasticRoute.QUEUE)
        self.assertEqual(self.controller.remote_kv_used_bytes, 128)

        fallback = self.controller.fallback_remote_before_start(
            "remote0", estimate(local=20, remote=10))
        self.assertEqual(fallback.route, ElasticRoute.LOCAL)
        self.assertEqual(self.controller.remote_kv_used_bytes, 64)
        self.controller.mark_started("remote0")
        with self.assertRaises(ValueError):
            self.controller.fallback_remote_before_start(
                "remote0", estimate(local=20, remote=10))

    def test_explicit_recovery_probe_success_and_failure(self):
        now = 0
        high_ids = []
        for index in range(5):
            name = f"high{index}"
            high_ids.append(name)
            self.controller.submit(request(name, now), estimate(local=10, remote=30))
            now += 10_000_000
        self.assertEqual(self.controller.regime, ElasticRegime.DEFLECT_ACTIVE)
        for name in high_ids[:3]:
            decision = self.controller.decision(name)
            if decision.route is ElasticRoute.LOCAL:
                self.controller.mark_started(name)
                self.controller.complete(name)

        for index in range(5):
            now += 100_000_000
            self.controller.submit(
                request(f"low{index}", now, local_cost=1),
                estimate(local=30, remote=29))
        self.assertEqual(self.controller.regime, ElasticRegime.RECOVERY_PROBE)
        probe = self.controller.submit(
            request("probe", now + 100_000_000, residency=CacheResidency.P_ONLY),
            estimate(local=30, remote=29))
        self.assertEqual(probe.route, ElasticRoute.REMOTE)
        self.assertTrue(probe.remote_probe)
        self.controller.mark_started("probe")
        self.controller.complete("probe", remote_probe_success=True)
        self.assertEqual(self.controller.regime, ElasticRegime.REMOTE_STABLE)

        # A later high-load epoch re-enters deflection. A failed recovery probe
        # returns to DEFLECT_ACTIVE instead of silently declaring recovery.
        for index in range(5):
            now += 10_000_000
            self.controller.submit(
                request(f"high2-{index}", now, local_cost=1),
                estimate(local=10, remote=30))
        for index in range(5):
            now += 100_000_000
            self.controller.submit(
                request(f"low2-{index}", now, local_cost=1),
                estimate(local=30, remote=29))
        failed_probe = self.controller.submit(
            request("failed-probe", now + 100_000_000,
                    residency=CacheResidency.P_ONLY),
            estimate(local=30, remote=29))
        self.assertTrue(failed_probe.remote_probe)
        self.controller.mark_started("failed-probe")
        self.controller.complete("failed-probe", remote_probe_success=False)
        self.assertEqual(self.controller.regime, ElasticRegime.DEFLECT_ACTIVE)

    def test_cache_residency_deadline_and_tbt_are_fail_closed(self):
        local = self.controller.submit(
            request("d-hit", 0, residency=CacheResidency.D_ONLY),
            estimate(local=20, remote=5))
        self.assertEqual(local.route, ElasticRoute.LOCAL)

        queued = self.controller.submit(
            request("unsafe", 100_000_000),
            estimate(local=100, remote=100, deadline=50, tbt=False,
                     backend=False, evidence=False))
        self.assertEqual(queued.route, ElasticRoute.QUEUE)
        self.assertEqual(queued.reason, "both_deadlines_infeasible")

    def test_duplicate_is_idempotent_and_credit_released_once(self):
        req = request("same", 0)
        first = self.controller.submit(req, estimate())
        second = self.controller.submit(req, estimate(local=999, remote=1))
        self.assertIs(first, second)
        self.controller.mark_started("same")
        self.controller.complete("same")
        self.assertEqual(self.controller.local_compute_used_us, 0)
        with self.assertRaises(ValueError):
            self.controller.complete("same")


if __name__ == "__main__":
    unittest.main()
