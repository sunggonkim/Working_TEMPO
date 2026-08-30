import unittest
from unittest.mock import patch

from tempo.pd_elastic_controller import (
    CacheResidency,
    CacheResidencyCatalog,
    ElasticConfig,
    ElasticEstimate,
    ElasticPDController,
    ElasticRequest,
    ElasticRoute,
)


def estimate(local, remote, *, evidence=True, uncertainty=0.0):
    return ElasticEstimate(
        local_upper_bound_ms=local,
        remote_upper_bound_ms=remote,
        uncertainty_ms=uncertainty,
        remaining_deadline_ms=1_000.0,
        local_tbt_safe=True,
        remote_backend_available=True,
        remote_evidence_valid=evidence,
    )


def request(name, residency):
    return ElasticRequest(
        request_id=name,
        arrival_ns=1,
        cache_residency=residency,
        local_compute_cost_us=1,
        remote_kv_bytes=1,
    )


class CanonicalElasticControllerTest(unittest.TestCase):
    def controller(self):
        return ElasticPDController(ElasticConfig(
            local_compute_budget_us=10,
            remote_kv_budget_bytes=10,
            route_margin_ms=5.0,
            spill_regression_budget_ms=5.0,
        ))

    def test_unknown_is_local_until_completion_evidence(self):
        with patch.dict(
            "os.environ",
            {"TEMPO_PD_BENCHMARK_COLD_MEASURED": "0"},
            clear=False,
        ):
            controller = self.controller()
        decision = controller.submit(
            request("unknown", CacheResidency.UNKNOWN), estimate(40, 10))
        self.assertEqual(decision.route, ElasticRoute.LOCAL)
        self.assertEqual(decision.cache_residency, CacheResidency.UNKNOWN)

    def test_explicit_cold_unknown_can_use_strong_remote_evidence(self):
        with patch.dict(
            "os.environ",
            {"TEMPO_PD_BENCHMARK_COLD_MEASURED": "1"},
            clear=False,
        ):
            controller = self.controller()
        cold = request("cold-unknown", CacheResidency.UNKNOWN)
        controller.register_request_geometry(cold.request_id, 512, 16)
        decision = controller.submit(
            cold, estimate(100, 50, uncertainty=10))
        self.assertEqual(decision.route, ElasticRoute.REMOTE)
        self.assertEqual(decision.reason, "cold_unknown_remote_evidence")
        self.assertEqual(decision.cache_residency, CacheResidency.UNKNOWN)
        self.assertEqual(controller.remote_kv_used_bytes, 1)
        evidence = controller.request_credit_evidence(cold.request_id)
        self.assertTrue(evidence["cold_unknown_remote_candidate"])
        self.assertTrue(evidence["cold_unknown_remote_admitted"])
        controller.mark_started(cold.request_id)
        controller.complete(cold.request_id)
        self.assertEqual(controller.remote_kv_used_bytes, 0)

    def test_explicit_cold_unknown_still_requires_remote_evidence(self):
        with patch.dict(
            "os.environ",
            {"TEMPO_PD_BENCHMARK_COLD_MEASURED": "1"},
            clear=False,
        ):
            controller = self.controller()
        cold = request("cold-no-evidence", CacheResidency.UNKNOWN)
        controller.register_request_geometry(cold.request_id, 512, 16)
        decision = controller.submit(
            cold, estimate(100, 50, evidence=False, uncertainty=10))
        self.assertEqual(decision.route, ElasticRoute.LOCAL)

    def test_invalid_cold_measured_setting_fails_closed(self):
        with patch.dict(
            "os.environ",
            {"TEMPO_PD_BENCHMARK_COLD_MEASURED": "yes"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                self.controller()

    def test_cold_high_load_output256_uses_separate_headroom_credit(self):
        with patch.dict(
            "os.environ",
            {
                "TEMPO_PD_BENCHMARK_COLD_MEASURED": "1",
                "TEMPO_PD_EXTERNALITY_SPILL_BUDGET_MS": "220",
                "TEMPO_PD_REMOTE_HEADROOM_KV_BUDGET_BYTES": "20",
            },
            clear=False,
        ):
            controller = self.controller()
        for index in range(5):
            request_id = f"cold-headroom-load-{index}"
            owner = ElasticRequest(
                request_id=request_id,
                arrival_ns=(index + 1) * 1_000_000,
                cache_residency=CacheResidency.D_ONLY,
                local_compute_cost_us=1,
                remote_kv_bytes=1,
            )
            controller.register_request_geometry(request_id, 512, 32)
            decision = controller.submit(owner, estimate(20, 100))
            self.assertEqual(decision.route, ElasticRoute.LOCAL)
            controller.mark_started(request_id)
            controller.complete(request_id)
        self.assertEqual(controller.regime.value, "deflect_active")

        intrinsic = ElasticRequest(
            request_id="cold-headroom-intrinsic",
            arrival_ns=6_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1,
            remote_kv_bytes=9,
        )
        controller.register_request_geometry(intrinsic.request_id, 4094, 16)
        intrinsic_decision = controller.submit(
            intrinsic, estimate(100, 20))
        self.assertEqual(intrinsic_decision.route, ElasticRoute.REMOTE)

        candidate = ElasticRequest(
            request_id="cold-headroom-output256",
            arrival_ns=7_000_000,
            cache_residency=CacheResidency.UNKNOWN,
            local_compute_cost_us=1,
            remote_kv_bytes=6,
        )
        controller.register_request_geometry(candidate.request_id, 1230, 256)
        decision = controller.submit(candidate, estimate(20, 120))
        self.assertEqual(decision.route, ElasticRoute.REMOTE)
        self.assertEqual(
            decision.reason,
            "cold_high_load_remote_headroom_deflection")
        self.assertEqual(decision.cache_residency, CacheResidency.UNKNOWN)
        self.assertEqual(controller.remote_kv_used_bytes, 15)
        evidence = controller.request_credit_evidence(candidate.request_id)
        self.assertTrue(evidence["cold_high_load_headroom_candidate"])
        self.assertTrue(evidence["cold_high_load_headroom_consumed"])

        output128 = ElasticRequest(
            request_id="cold-headroom-output128",
            arrival_ns=8_000_000,
            cache_residency=CacheResidency.UNKNOWN,
            local_compute_cost_us=1,
            remote_kv_bytes=1,
        )
        controller.register_request_geometry(output128.request_id, 512, 128)
        output128_decision = controller.submit(output128, estimate(20, 120))
        self.assertEqual(output128_decision.route, ElasticRoute.LOCAL)
        output128_evidence = controller.request_credit_evidence(
            output128.request_id)
        self.assertFalse(
            output128_evidence["cold_high_load_headroom_candidate"])

        for owner in (intrinsic, candidate, output128):
            controller.mark_started(owner.request_id)
            controller.complete(owner.request_id)
        self.assertEqual(controller.remote_kv_used_bytes, 0)

    def test_p_only_requires_five_ms_benefit(self):
        near = self.controller().submit(
            request("near", CacheResidency.P_ONLY), estimate(20, 16))
        self.assertEqual(near.route, ElasticRoute.LOCAL)
        far = self.controller().submit(
            request("far", CacheResidency.P_ONLY), estimate(20, 14))
        self.assertEqual(far.route, ElasticRoute.REMOTE)

    def test_intrinsic_remote_advantage_must_exceed_uncertainty(self):
        uncertain = self.controller().submit(
            request("uncertain", CacheResidency.P_ONLY),
            estimate(30, 20, uncertainty=6))
        self.assertEqual(uncertain.route, ElasticRoute.LOCAL)
        decisive = self.controller().submit(
            request("decisive", CacheResidency.P_ONLY),
            estimate(32, 20, uncertainty=6))
        self.assertEqual(decisive.route, ElasticRoute.REMOTE)

    def test_short_intrinsic_remote_requires_geometry_and_strong_mean_gain(self):
        baseline = self.controller()
        baseline_request = request("short-baseline", CacheResidency.P_ONLY)
        baseline.register_request_geometry(baseline_request.request_id, 512, 16)
        self.assertEqual(
            baseline.submit(
                baseline_request, estimate(100, 45, uncertainty=60)).route,
            ElasticRoute.LOCAL,
        )

        with patch.dict(
            "os.environ",
            {"TEMPO_PD_SHORT_REMOTE_MIN_ADVANTAGE_MS": "50"},
            clear=False,
        ):
            controller = self.controller()
        short = request("short-evidence", CacheResidency.P_ONLY)
        controller.register_request_geometry(short.request_id, 512, 16)
        decision = controller.submit(
            short, estimate(100, 45, uncertainty=60))
        self.assertEqual(decision.route, ElasticRoute.REMOTE)
        self.assertEqual(decision.reason, "short_intrinsic_remote_evidence")

        longer = ElasticRequest(
            request_id="short-output-guard",
            arrival_ns=2,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1,
            remote_kv_bytes=1,
        )
        controller.register_request_geometry(longer.request_id, 512, 32)
        self.assertEqual(
            controller.submit(
                longer, estimate(100, 45, uncertainty=60)).route,
            ElasticRoute.LOCAL,
        )

        for invalid in ("49", "251", "nan", "not-a-number"):
            with self.subTest(invalid=invalid), patch.dict(
                "os.environ",
                {"TEMPO_PD_SHORT_REMOTE_MIN_ADVANTAGE_MS": invalid},
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    self.controller()

    def test_p_only_near_tie_spills_after_local_credit_exhaustion(self):
        controller = ElasticPDController(ElasticConfig(
            local_compute_budget_us=1,
            remote_kv_budget_bytes=10,
            route_margin_ms=5.0,
            spill_regression_budget_ms=5.0,
        ))
        local = controller.submit(
            ElasticRequest(
                request_id="occupy-local", arrival_ns=1,
                cache_residency=CacheResidency.D_ONLY,
                local_compute_cost_us=1, remote_kv_bytes=1,
            ),
            estimate(20, 50),
        )
        spill = controller.submit(
            ElasticRequest(
                request_id="p-only-spill", arrival_ns=2,
                cache_residency=CacheResidency.P_ONLY,
                local_compute_cost_us=1, remote_kv_bytes=1,
            ),
            estimate(20, 21),
        )
        self.assertEqual(local.route, ElasticRoute.LOCAL)
        self.assertEqual(spill.route, ElasticRoute.REMOTE)
        self.assertEqual(spill.reason, "bounded_spill_to_alternate")
        self.assertEqual(spill.cache_residency, CacheResidency.P_ONLY)

    def test_oversized_local_only_request_borrows_budget_exclusively(self):
        controller = ElasticPDController(ElasticConfig(
            local_compute_budget_us=10,
            remote_kv_budget_bytes=10,
            route_margin_ms=5.0,
            spill_regression_budget_ms=5.0,
        ))
        first_request = ElasticRequest(
            request_id="oversized-first", arrival_ns=1,
            cache_residency=CacheResidency.D_ONLY,
            local_compute_cost_us=15, remote_kv_bytes=1,
        )
        second_request = ElasticRequest(
            request_id="oversized-second", arrival_ns=2,
            cache_residency=CacheResidency.D_ONLY,
            local_compute_cost_us=15, remote_kv_bytes=1,
        )
        first = controller.submit(first_request, estimate(20, 100))
        queued = controller.submit(second_request, estimate(20, 100))
        self.assertEqual(first.route, ElasticRoute.LOCAL)
        self.assertEqual(first.reason, "oversized_singleton_local_borrow")
        self.assertEqual(controller.local_compute_used_us, 10)
        self.assertEqual(queued.route, ElasticRoute.QUEUE)
        controller.mark_started(first_request.request_id)
        controller.complete(first_request.request_id)
        retried = controller.retry(second_request.request_id, estimate(20, 100))
        self.assertEqual(retried.route, ElasticRoute.LOCAL)
        self.assertEqual(retried.reason, "oversized_singleton_local_borrow")

    def test_critical_output_expands_local_credit_to_two_prefills(self):
        controller = ElasticPDController(ElasticConfig(
            local_compute_budget_us=10,
            remote_kv_budget_bytes=10,
            route_margin_ms=5.0,
            spill_regression_budget_ms=5.0,
        ))
        first = ElasticRequest(
            request_id="ordinary-owner", arrival_ns=1,
            cache_residency=CacheResidency.D_ONLY,
            local_compute_cost_us=6, remote_kv_bytes=1,
        )
        critical = ElasticRequest(
            request_id="critical-output", arrival_ns=2,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=6, remote_kv_bytes=1,
        )
        controller.register_request_geometry(first.request_id, 512, 128)
        controller.register_request_geometry(critical.request_id, 2048, 256)
        self.assertEqual(
            controller.submit(first, estimate(20, 100)).route,
            ElasticRoute.LOCAL,
        )
        decision = controller.submit(critical, estimate(20, 21))
        self.assertEqual(decision.route, ElasticRoute.LOCAL)
        self.assertEqual(
            decision.reason, "critical_output_expanded_local_credit")
        self.assertEqual(decision.local_compute_budget_us, 12)
        self.assertEqual(controller.local_compute_used_us, 12)

    def test_noncritical_output_keeps_base_credit_and_spills(self):
        controller = ElasticPDController(ElasticConfig(
            local_compute_budget_us=10,
            remote_kv_budget_bytes=10,
            route_margin_ms=5.0,
            spill_regression_budget_ms=5.0,
        ))
        first = ElasticRequest(
            request_id="ordinary-first", arrival_ns=1,
            cache_residency=CacheResidency.D_ONLY,
            local_compute_cost_us=6, remote_kv_bytes=1,
        )
        second = ElasticRequest(
            request_id="ordinary-second", arrival_ns=2,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=6, remote_kv_bytes=1,
        )
        controller.register_request_geometry(first.request_id, 512, 64)
        controller.register_request_geometry(second.request_id, 512, 64)
        controller.submit(first, estimate(20, 100))
        decision = controller.submit(second, estimate(20, 21))
        self.assertEqual(decision.route, ElasticRoute.REMOTE)
        self.assertEqual(decision.reason, "bounded_spill_to_alternate")
        self.assertEqual(decision.local_compute_budget_us, 10)

    def test_request_geometry_is_idempotent_but_cannot_change(self):
        controller = self.controller()
        controller.register_request_geometry("same", 2048, 256)
        controller.register_request_geometry("same", 2048, 256)
        with self.assertRaises(ValueError):
            controller.register_request_geometry("same", 1230, 256)

    def test_high_load_safe_headroom_deflects_but_low_load_does_not(self):
        controller = self.controller()
        for index in range(5):
            request_id = f"load-{index}"
            owner = ElasticRequest(
                request_id=request_id,
                arrival_ns=(index + 1) * 1_000_000,
                cache_residency=CacheResidency.D_ONLY,
                local_compute_cost_us=1,
                remote_kv_bytes=1,
            )
            controller.register_request_geometry(request_id, 512, 32)
            decision = controller.submit(owner, estimate(20, 100))
            self.assertEqual(decision.route, ElasticRoute.LOCAL)
            controller.mark_started(request_id)
            controller.complete(request_id)
        self.assertEqual(controller.regime.value, "deflect_active")

        donor = ElasticRequest(
            request_id="headroom-donor",
            arrival_ns=6_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1,
            remote_kv_bytes=1,
        )
        controller.register_request_geometry(donor.request_id, 512, 128)
        decision = controller.submit(donor, estimate(20, 21))
        self.assertEqual(decision.route, ElasticRoute.REMOTE)
        self.assertEqual(decision.reason, "high_load_remote_headroom_deflection")
        self.assertEqual(decision.regime.value, "deflect_active")
        self.assertEqual(decision.local_score_ms, 20)
        self.assertEqual(decision.remote_score_ms, 21)

        boundary = ElasticRequest(
            request_id="headroom-2k-boundary",
            arrival_ns=7_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1,
            remote_kv_bytes=1,
        )
        controller.register_request_geometry(boundary.request_id, 2048, 128)
        boundary_decision = controller.submit(boundary, estimate(20, 21))
        self.assertEqual(boundary_decision.route, ElasticRoute.REMOTE)
        self.assertEqual(
            boundary_decision.reason, "high_load_remote_headroom_deflection")

        low_load = self.controller()
        quiet = request("quiet", CacheResidency.P_ONLY)
        low_load.register_request_geometry(quiet.request_id, 512, 128)
        quiet_decision = low_load.submit(quiet, estimate(20, 21))
        self.assertEqual(quiet_decision.route, ElasticRoute.LOCAL)

    def test_remote_byte_and_request_credits_are_independent(self):
        with patch.dict(
            "os.environ",
            {
                "TEMPO_PD_REMOTE_REQUEST_BUDGET": "2",
                "TEMPO_PD_REMOTE_HEADROOM_REQUEST_BUDGET": "1",
            },
            clear=False,
        ):
            controller = self.controller()
        for index in range(5):
            request_id = f"count-load-{index}"
            owner = ElasticRequest(
                request_id=request_id,
                arrival_ns=(index + 1) * 1_000_000,
                cache_residency=CacheResidency.D_ONLY,
                local_compute_cost_us=1,
                remote_kv_bytes=1,
            )
            controller.register_request_geometry(request_id, 512, 32)
            decision = controller.submit(owner, estimate(20, 100))
            controller.mark_started(request_id)
            controller.complete(request_id)
            self.assertEqual(decision.route, ElasticRoute.LOCAL)

        first = ElasticRequest(
            request_id="count-headroom-first", arrival_ns=6_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1, remote_kv_bytes=1,
        )
        controller.register_request_geometry(first.request_id, 512, 128)
        first_decision = controller.submit(first, estimate(20, 21))
        self.assertEqual(first_decision.route, ElasticRoute.REMOTE)
        self.assertEqual(
            first_decision.reason, "high_load_remote_headroom_deflection")

        second = ElasticRequest(
            request_id="count-headroom-second", arrival_ns=7_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1, remote_kv_bytes=1,
        )
        controller.register_request_geometry(second.request_id, 512, 128)
        second_decision = controller.submit(second, estimate(20, 21))
        self.assertEqual(second_decision.route, ElasticRoute.LOCAL)
        self.assertEqual(
            second_decision.reason,
            "remote_headroom_credit_exhausted_to_local")

        intrinsic = ElasticRequest(
            request_id="count-intrinsic-second-slot", arrival_ns=8_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1, remote_kv_bytes=1,
        )
        controller.register_request_geometry(intrinsic.request_id, 4094, 16)
        intrinsic_decision = controller.submit(intrinsic, estimate(20, 10))
        self.assertEqual(intrinsic_decision.route, ElasticRoute.REMOTE)

        exhausted = ElasticRequest(
            request_id="count-total-exhausted", arrival_ns=9_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1, remote_kv_bytes=1,
        )
        controller.register_request_geometry(exhausted.request_id, 4094, 16)
        exhausted_decision = controller.submit(exhausted, estimate(20, 10))
        self.assertEqual(exhausted_decision.route, ElasticRoute.LOCAL)
        self.assertEqual(
            exhausted_decision.reason,
            "remote_request_credit_exhausted_to_local")

        evidence = controller.request_credit_evidence(exhausted.request_id)
        self.assertEqual(evidence["remote_requests_used_before"], 2)
        self.assertEqual(evidence["remote_request_budget"], 2)
        self.assertFalse(evidence["remote_request_credit_available"])

        for variables in (
            {"TEMPO_PD_REMOTE_REQUEST_BUDGET": "0"},
            {"TEMPO_PD_REMOTE_REQUEST_BUDGET": "not-an-integer"},
            {
                "TEMPO_PD_REMOTE_REQUEST_BUDGET": "2",
                "TEMPO_PD_REMOTE_HEADROOM_REQUEST_BUDGET": "3",
            },
        ):
            with self.subTest(variables=variables), patch.dict(
                "os.environ", variables, clear=False,
            ):
                with self.assertRaises(ValueError):
                    self.controller()

    def test_headroom_geometry_split_keeps_short_or_long_output_remote(self):
        with patch.dict(
            "os.environ",
            {"TEMPO_PD_HEADROOM_MEDIUM_MIN_OUTPUT_TOKENS": "256"},
            clear=False,
        ):
            controller = self.controller()
        self.assertEqual(controller.headroom_medium_min_output_tokens, 256)
        for index in range(5):
            request_id = f"split-load-{index}"
            owner = ElasticRequest(
                request_id=request_id,
                arrival_ns=(index + 1) * 1_000_000,
                cache_residency=CacheResidency.D_ONLY,
                local_compute_cost_us=1,
                remote_kv_bytes=1,
            )
            controller.register_request_geometry(request_id, 512, 32)
            controller.submit(owner, estimate(20, 100))
            controller.mark_started(request_id)
            controller.complete(request_id)

        medium = ElasticRequest(
            request_id="split-medium", arrival_ns=6_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1, remote_kv_bytes=1,
        )
        controller.register_request_geometry(medium.request_id, 1230, 128)
        self.assertEqual(
            controller.submit(medium, estimate(20, 21)).route, ElasticRoute.LOCAL)

        short = ElasticRequest(
            request_id="split-short", arrival_ns=7_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1, remote_kv_bytes=1,
        )
        controller.register_request_geometry(short.request_id, 512, 128)
        self.assertEqual(
            controller.submit(short, estimate(20, 21)).route, ElasticRoute.REMOTE)

        long_output = ElasticRequest(
            request_id="split-long-output", arrival_ns=8_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1, remote_kv_bytes=1,
        )
        controller.register_request_geometry(long_output.request_id, 1230, 256)
        self.assertEqual(
            controller.submit(
                long_output, estimate(20, 21)).route, ElasticRoute.REMOTE)

        for invalid in ("127", "129", "not-an-integer"):
            with self.subTest(invalid=invalid), patch.dict(
                "os.environ",
                {"TEMPO_PD_HEADROOM_MEDIUM_MIN_OUTPUT_TOKENS": invalid},
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    self.controller()

    def test_externality_budget_is_bounded_and_request_scoped(self):
        def loaded_controller():
            controller = self.controller()
            for index in range(5):
                request_id = f"externality-load-{index}"
                owner = ElasticRequest(
                    request_id=request_id,
                    arrival_ns=(index + 1) * 1_000_000,
                    cache_residency=CacheResidency.D_ONLY,
                    local_compute_cost_us=1,
                    remote_kv_bytes=1,
                )
                controller.register_request_geometry(request_id, 512, 32)
                decision = controller.submit(owner, estimate(20, 100))
                self.assertEqual(decision.route, ElasticRoute.LOCAL)
                controller.mark_started(request_id)
                controller.complete(request_id)
            self.assertEqual(controller.regime.value, "deflect_active")
            return controller

        baseline = loaded_controller()
        baseline_request = ElasticRequest(
            request_id="externality-default",
            arrival_ns=6_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1,
            remote_kv_bytes=1,
        )
        baseline.register_request_geometry(
            baseline_request.request_id, 2048, 128)
        baseline_decision = baseline.submit(
            baseline_request, estimate(20, 215))
        self.assertEqual(baseline_decision.route, ElasticRoute.LOCAL)

        with patch.dict(
            "os.environ",
            {"TEMPO_PD_EXTERNALITY_SPILL_BUDGET_MS": "220"},
            clear=False,
        ):
            expanded = loaded_controller()
        expanded_request = ElasticRequest(
            request_id="externality-expanded",
            arrival_ns=6_000_000,
            cache_residency=CacheResidency.P_ONLY,
            local_compute_cost_us=1,
            remote_kv_bytes=1,
        )
        expanded.register_request_geometry(
            expanded_request.request_id, 2048, 128)
        expanded_decision = expanded.submit(
            expanded_request, estimate(20, 215))
        self.assertEqual(expanded_decision.route, ElasticRoute.REMOTE)
        self.assertEqual(
            expanded_decision.reason,
            "high_load_remote_headroom_deflection",
        )
        self.assertEqual(expanded.externality_spill_budget_ms, 220)
        self.assertEqual(expanded.config.spill_regression_budget_ms, 5)

    def test_remote_kv_budget_override_is_bounded_and_effective(self):
        config = ElasticConfig(
            local_compute_budget_us=10,
            remote_kv_budget_bytes=10,
            route_margin_ms=5.0,
            spill_regression_budget_ms=5.0,
        )
        with patch.dict(
            "os.environ",
            {
                "TEMPO_PD_REMOTE_KV_BUDGET_BYTES": "15",
                "TEMPO_PD_REMOTE_HEADROOM_KV_BUDGET_BYTES": "18",
            },
            clear=False,
        ):
            controller = ElasticPDController(config)
        self.assertEqual(controller.profile_remote_kv_budget_bytes, 10)
        self.assertEqual(controller.effective_remote_kv_budget_bytes, 15)
        self.assertEqual(controller.remote_headroom_kv_budget_bytes, 18)
        self.assertEqual(controller.config.remote_kv_budget_bytes, 15)

        for invalid in ("9", "21", "not-an-integer"):
            with self.subTest(invalid=invalid), patch.dict(
                "os.environ",
                {"TEMPO_PD_REMOTE_KV_BUDGET_BYTES": invalid},
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    ElasticPDController(config)
        for invalid in ("14", "21", "not-an-integer"):
            with self.subTest(headroom=invalid), patch.dict(
                "os.environ",
                {
                    "TEMPO_PD_REMOTE_KV_BUDGET_BYTES": "15",
                    "TEMPO_PD_REMOTE_HEADROOM_KV_BUDGET_BYTES": invalid,
                },
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    ElasticPDController(config)

    def test_d_only_and_both_are_local_first(self):
        for index, residency in enumerate((CacheResidency.D_ONLY, CacheResidency.BOTH)):
            with self.subTest(residency=residency):
                decision = self.controller().submit(
                    request(f"warm-{index}", residency), estimate(40, 10))
                self.assertEqual(decision.route, ElasticRoute.LOCAL)


class CacheCatalogTest(unittest.TestCase):
    def test_only_completed_event_establishes_residency(self):
        catalog = CacheResidencyCatalog()
        namespace = catalog.namespace(
            arm="tempo", prompt_tokens=2048, output_tokens=32, item="x")
        self.assertEqual(catalog.classify(namespace), CacheResidency.UNKNOWN)
        catalog.record_completion(
            namespace, prefill_resident=True, decode_resident=False,
            actual_kv_bytes=123, completed_ns=5)
        self.assertEqual(catalog.classify(namespace), CacheResidency.P_ONLY)

    def test_repeated_identical_completion_is_idempotent_but_changes_fail(self):
        catalog = CacheResidencyCatalog()
        namespace = catalog.namespace(
            arm="tempo", prompt_tokens=2048, output_tokens=32, item="repeat")
        first = catalog.record_completion(
            namespace, prefill_resident=True, decode_resident=False,
            actual_kv_bytes=123, completed_ns=5)
        second = catalog.record_completion(
            namespace, prefill_resident=True, decode_resident=False,
            actual_kv_bytes=123, completed_ns=6)
        self.assertIs(second, first)
        with self.assertRaises(ValueError):
            catalog.record_completion(
                namespace, prefill_resident=False, decode_resident=False,
                actual_kv_bytes=0, completed_ns=7)

    def test_residency_evidence_can_only_grow_monotonically(self):
        catalog = CacheResidencyCatalog()
        namespace = catalog.namespace(
            arm="tempo", prompt_tokens=2048, output_tokens=32,
            item="transition")
        catalog.record_completion(
            namespace, prefill_resident=True, decode_resident=False,
            actual_kv_bytes=123, completed_ns=5)
        both = catalog.record_completion(
            namespace, prefill_resident=True, decode_resident=True,
            actual_kv_bytes=0, completed_ns=6)
        self.assertEqual(both.residency, CacheResidency.BOTH)
        self.assertIs(catalog.event(namespace), both)
        with self.assertRaisesRegex(ValueError, "regressed"):
            catalog.record_completion(
                namespace, prefill_resident=True, decode_resident=False,
                actual_kv_bytes=123, completed_ns=7)
        with self.assertRaisesRegex(ValueError, "stale"):
            catalog.record_completion(
                namespace, prefill_resident=True, decode_resident=True,
                actual_kv_bytes=0, completed_ns=4)


    def test_namespace_separates_arm_and_geometry(self):
        catalog = CacheResidencyCatalog()
        tempo = catalog.namespace(arm="tempo", prompt_tokens=512,
                                  output_tokens=32, item="x")
        local = catalog.namespace(arm="local", prompt_tokens=512,
                                 output_tokens=32, item="x")
        other_prompt = catalog.namespace(
            arm="tempo", prompt_tokens=2048, output_tokens=32, item="x")
        other_decode_length = catalog.namespace(
            arm="tempo", prompt_tokens=512, output_tokens=256, item="x")
        self.assertNotEqual(tempo, local)
        self.assertNotEqual(tempo, other_prompt)
        self.assertEqual(tempo, other_decode_length)


if __name__ == "__main__":
    unittest.main()
