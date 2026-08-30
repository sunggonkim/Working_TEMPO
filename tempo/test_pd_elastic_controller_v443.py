import unittest

from tempo.pd_elastic_controller_v443 import (
    CacheResidency,
    ElasticConfig,
    ElasticEstimate,
    ElasticPDController,
    ElasticRegime,
    ElasticRequest,
    ElasticRoute,
    POLICY_ID,
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


class ElasticPDControllerCorrectedTest(unittest.TestCase):
    def setUp(self):
        self.controller = ElasticPDController(ElasticConfig(
            local_compute_budget_us=6,
            remote_kv_budget_bytes=128,
            exit_consecutive_windows=2,
        ))

    def test_high_load_weighted_credit_queues_then_retries(self):
        decisions = []
        for index in range(5):
            decisions.append(self.controller.submit(
                request(f"r{index}", index * 10_000_000), estimate()))
        self.assertEqual(self.controller.regime, ElasticRegime.DEFLECT_ACTIVE)
        local = [value for value in decisions if value.route is ElasticRoute.LOCAL]
        self.assertEqual(len(local), 3)
        self.assertEqual(decisions[-1].route, ElasticRoute.QUEUE)
        self.controller.mark_started(local[0].request_id)
        self.controller.complete(local[0].request_id)
        retried = self.controller.retry(decisions[-1].request_id, estimate())
        self.assertEqual(retried.route, ElasticRoute.LOCAL)
        self.assertEqual(retried.attempt, 2)

    def test_remote_bytes_and_safe_prestart_local_fallback(self):
        remote = estimate(local=50, remote=10)
        first = self.controller.submit(
            request("remote0", 0, residency=CacheResidency.P_ONLY), remote)
        second = self.controller.submit(
            request("remote1", 100_000_000, residency=CacheResidency.P_ONLY), remote)
        third = self.controller.submit(
            request("remote2", 200_000_000, residency=CacheResidency.P_ONLY), remote)
        self.assertEqual((first.route, second.route),
                         (ElasticRoute.REMOTE, ElasticRoute.REMOTE))
        self.assertEqual(third.route, ElasticRoute.QUEUE)
        fallback = self.controller.fallback_remote_before_start(
            "remote0", estimate(local=20, remote=10))
        self.assertEqual(fallback.route, ElasticRoute.LOCAL)
        self.assertEqual(fallback.reason, "remote_prestart_failure_to_local")
        self.assertEqual(self.controller.remote_kv_used_bytes, 64)
        self.controller.mark_started("remote0")
        with self.assertRaises(ValueError):
            self.controller.fallback_remote_before_start(
                "remote0", estimate(local=20, remote=10))

    def test_recovery_requires_eligible_probe_and_success_feedback(self):
        now = 0
        for index in range(5):
            decision = self.controller.submit(
                request(f"high{index}", now), estimate(local=10, remote=30))
            now += 10_000_000
            if decision.route is ElasticRoute.LOCAL:
                self.controller.mark_started(decision.request_id)
                self.controller.complete(decision.request_id)
        self.assertEqual(self.controller.regime, ElasticRegime.DEFLECT_ACTIVE)

        ordinary_recovery_decisions = []
        for index in range(5):
            now += 100_000_000
            ordinary_recovery_decisions.append(self.controller.submit(
                request(f"low{index}", now, local_cost=1),
                estimate(local=30, remote=29)))
        self.assertEqual(self.controller.regime, ElasticRegime.RECOVERY_PROBE)
        self.assertFalse(any(value.remote_probe
                             for value in ordinary_recovery_decisions))

        probe = self.controller.submit(
            request("probe", now + 100_000_000,
                    residency=CacheResidency.P_ONLY),
            estimate(local=30, remote=29))
        self.assertEqual(probe.route, ElasticRoute.REMOTE)
        self.assertTrue(probe.remote_probe)
        self.controller.mark_started("probe")
        self.controller.complete("probe", remote_probe_success=True)
        self.assertEqual(self.controller.regime, ElasticRegime.REMOTE_STABLE)

    def test_fail_closed_deadline_tbt_and_idempotency(self):
        req = request("unsafe", 0)
        unsafe = estimate(local=100, remote=100, deadline=50, tbt=False,
                          backend=False, evidence=False)
        first = self.controller.submit(req, unsafe)
        second = self.controller.submit(req, estimate(local=1, remote=1))
        self.assertIs(first, second)
        self.assertEqual(first.route, ElasticRoute.QUEUE)
        self.assertEqual(first.reason, "both_deadlines_infeasible")
        self.assertEqual(first.policy_id, POLICY_ID)

    def test_failure_releases_exact_credit_once(self):
        decision = self.controller.submit(request("local", 0), estimate())
        self.assertEqual(decision.route, ElasticRoute.LOCAL)
        self.controller.mark_started("local")
        self.controller.fail("local")
        self.assertEqual(self.controller.local_compute_used_us, 0)
        with self.assertRaises(ValueError):
            self.controller.fail("local")


if __name__ == "__main__":
    unittest.main()
