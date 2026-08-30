from __future__ import annotations

import unittest

from tempo.pd_admission import (
    FrozenPDAdmissionPolicy,
    PDAdmissionLedger,
    PDCalibrationProfile,
    PDDecisionReason,
    PDEvidenceLevel,
    PDPolicyConfig,
    PDRequestContext,
    PDRequestPhase,
    PDRoute,
    PDWorkloadClass,
)


def _workload(*, backend: str = "lmcache-ucx") -> PDWorkloadClass:
    return PDWorkloadClass(
        model_id="Qwen2.5-7B-Instruct",
        model_revision="immutable-local-sha",
        topology_id="perlmutter-4n-2x-tp4-pd",
        remote_backend=backend,
        prompt_bucket="3078",
        output_bucket="2",
        decoder_load_bucket="7x128",
        kv_bytes_bucket="176.5MB",
    )


def _profile(
    *,
    evidence: PDEvidenceLevel = PDEvidenceLevel.REPLICATED,
    samples: int = 3,
    local_lower: float = 220.0,
    remote_upper: float = 200.0,
    equivalent: bool = True,
    failures: int = 0,
) -> PDCalibrationProfile:
    return PDCalibrationProfile(
        workload=_workload(),
        evidence_level=evidence,
        local_samples=samples,
        remote_samples=samples,
        local_latency_p50_ms=225.0,
        remote_latency_p50_ms=195.0,
        local_latency_lower_bound_ms=local_lower,
        remote_latency_upper_bound_ms=remote_upper,
        outputs_equivalent=equivalent,
        remote_transfer_failures=failures,
        valid_from_epoch=10,
        valid_through_epoch=12,
    )


def _request(request_id: str = "r0", **kwargs: object) -> PDRequestContext:
    values = {
        "request_id": request_id,
        "workload": _workload(),
        "policy_epoch": 11,
        "remote_backend_available": True,
        "remaining_deadline_ms": None,
    }
    values.update(kwargs)
    return PDRequestContext(**values)  # type: ignore[arg-type]


class FrozenPDAdmissionPolicyTests(unittest.TestCase):
    def test_replicated_remote_win_is_admitted_at_lower_bound(self) -> None:
        policy = FrozenPDAdmissionPolicy([_profile()])
        decision = policy.decide(_request())
        self.assertEqual(decision.route, PDRoute.REMOTE_PREFILL)
        self.assertEqual(decision.reason, PDDecisionReason.REMOTE_BENEFIT_PROVEN)
        self.assertEqual(decision.remote_advantage_lower_bound_ms, 20.0)
        self.assertTrue(decision.fallback_allowed_before_remote_start)

    def test_screen_evidence_is_local_by_default(self) -> None:
        profile = _profile(evidence=PDEvidenceLevel.SCREEN, samples=1)
        decision = FrozenPDAdmissionPolicy([profile]).decide(_request())
        self.assertEqual(decision.route, PDRoute.DECODER_LOCAL)
        self.assertEqual(decision.reason, PDDecisionReason.LOCAL_SCREEN_ONLY)

        screen_policy = FrozenPDAdmissionPolicy(
            [profile],
            PDPolicyConfig(
                minimum_samples_per_route=1,
                require_replicated_evidence=False,
            ),
        )
        self.assertEqual(screen_policy.decide(_request()).route, PDRoute.REMOTE_PREFILL)

    def test_no_profile_and_unavailable_backend_fail_closed(self) -> None:
        missing = FrozenPDAdmissionPolicy([]).decide(_request())
        self.assertEqual(missing.reason, PDDecisionReason.LOCAL_NO_PROFILE)
        unavailable = FrozenPDAdmissionPolicy([_profile()]).decide(
            _request(remote_backend_available=False)
        )
        self.assertEqual(
            unavailable.reason, PDDecisionReason.LOCAL_REMOTE_UNAVAILABLE
        )

    def test_correctness_failure_margin_and_deadline_fail_closed(self) -> None:
        bad_output = FrozenPDAdmissionPolicy(
            [_profile(equivalent=False)]
        ).decide(_request())
        self.assertEqual(
            bad_output.reason, PDDecisionReason.LOCAL_CORRECTNESS_UNPROVEN
        )
        failed_transfer = FrozenPDAdmissionPolicy(
            [_profile(failures=1)]
        ).decide(_request())
        self.assertEqual(
            failed_transfer.reason, PDDecisionReason.LOCAL_REMOTE_FAILURE
        )
        weak = FrozenPDAdmissionPolicy(
            [_profile(local_lower=204.9, remote_upper=200.0)]
        ).decide(_request())
        self.assertEqual(weak.reason, PDDecisionReason.LOCAL_MARGIN_NOT_MET)
        deadline = FrozenPDAdmissionPolicy([_profile()]).decide(
            _request(remaining_deadline_ms=199.0)
        )
        self.assertEqual(
            deadline.reason, PDDecisionReason.LOCAL_DEADLINE_INFEASIBLE
        )

    def test_epoch_and_duplicate_profiles_are_rejected(self) -> None:
        expired = FrozenPDAdmissionPolicy([_profile()]).decide(
            _request(policy_epoch=13)
        )
        self.assertEqual(expired.reason, PDDecisionReason.LOCAL_PROFILE_EXPIRED)
        with self.assertRaisesRegex(ValueError, "duplicate workload profile"):
            FrozenPDAdmissionPolicy([_profile(), _profile()])

    def test_profile_fingerprint_is_stable_and_content_bound(self) -> None:
        left = _profile()
        right = _profile(local_lower=221.0)
        self.assertEqual(left.profile_id, _profile().profile_id)
        self.assertNotEqual(left.profile_id, right.profile_id)


class PDAdmissionLedgerTests(unittest.TestCase):
    def test_remote_lifecycle_and_no_late_fallback(self) -> None:
        ledger = PDAdmissionLedger(FrozenPDAdmissionPolicy([_profile()]))
        ledger.admit(_request())
        ledger.mark_remote_started("r0")
        with self.assertRaisesRegex(ValueError, "before remote prefill"):
            ledger.fallback_before_remote_start("r0", "late failure")
        ledger.mark_decode_started("r0")
        ledger.complete("r0")
        self.assertEqual(ledger.record("r0").phase, PDRequestPhase.COMPLETE)

    def test_prestart_failure_falls_back_to_local(self) -> None:
        ledger = PDAdmissionLedger(FrozenPDAdmissionPolicy([_profile()]))
        ledger.admit(_request())
        decision = ledger.fallback_before_remote_start("r0", "reservation failed")
        self.assertEqual(decision.route, PDRoute.DECODER_LOCAL)
        self.assertEqual(
            decision.reason, PDDecisionReason.LOCAL_REMOTE_PRESTART_FAILURE
        )
        ledger.mark_decode_started("r0")
        ledger.complete("r0")

    def test_local_request_cannot_start_remote(self) -> None:
        ledger = PDAdmissionLedger(FrozenPDAdmissionPolicy([]))
        ledger.admit(_request())
        with self.assertRaisesRegex(ValueError, "expected remote_selected"):
            ledger.mark_remote_started("r0")
        ledger.mark_decode_started("r0")
        ledger.complete("r0")


if __name__ == "__main__":
    unittest.main()
