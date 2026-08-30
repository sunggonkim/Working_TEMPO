import unittest

from tempo.pd_elastic_controller_v443 import (
    CacheResidency,
    ElasticConfig,
    ElasticEstimate,
    ElasticPDController,
    ElasticRequest,
    ElasticRoute,
)


def _estimate(local: float, remote: float) -> ElasticEstimate:
    return ElasticEstimate(
        local_upper_bound_ms=local,
        remote_upper_bound_ms=remote,
        uncertainty_ms=0.0,
        remaining_deadline_ms=1_000.0,
        local_tbt_safe=True,
        remote_backend_available=True,
        remote_evidence_valid=True,
    )


def _request(residency: CacheResidency) -> ElasticRequest:
    return ElasticRequest(
        request_id=f"request-{residency.value}",
        arrival_ns=1,
        cache_residency=residency,
        local_compute_cost_us=1,
        remote_kv_bytes=1,
    )


class CacheResidencyPolicyInvariantTest(unittest.TestCase):
    def _route(self, residency, local, remote):
        controller = ElasticPDController(ElasticConfig(
            local_compute_budget_us=10,
            remote_kv_budget_bytes=10,
            route_margin_ms=5.0,
            spill_regression_budget_ms=5.0,
        ))
        return controller.submit(
            _request(residency), _estimate(local, remote)).route

    def test_p_only_affinity_selects_remote_within_bounded_regression(self):
        self.assertEqual(
            self._route(CacheResidency.P_ONLY, local=20, remote=24),
            ElasticRoute.REMOTE,
        )

    def test_d_only_and_both_affinity_select_local(self):
        for residency in (CacheResidency.D_ONLY, CacheResidency.BOTH):
            with self.subTest(residency=residency):
                self.assertEqual(
                    self._route(residency, local=20, remote=10),
                    ElasticRoute.LOCAL,
                )

    def test_confirmed_miss_requires_modelled_remote_advantage(self):
        self.assertEqual(
            self._route(CacheResidency.MISS, local=20, remote=16),
            ElasticRoute.LOCAL,
        )
        self.assertEqual(
            self._route(CacheResidency.MISS, local=20, remote=14),
            ElasticRoute.REMOTE,
        )


if __name__ == "__main__":
    unittest.main()
