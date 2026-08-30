from __future__ import annotations

import unittest

from tempo.c0_admission import C0Admission, C0Config


MIB = 1 << 20


class C0AdmissionTests(unittest.TestCase):
    def test_inflight_cap_is_all_or_nothing(self) -> None:
        controller = C0Admission(C0Config(max_inflight_bytes=2 * MIB))
        decision = controller.try_admit(
            now_ns=0,
            request_bytes=MIB,
            inflight_bytes=2 * MIB,
        )
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.reason, "inflight_cap")

        decision = controller.try_admit(
            now_ns=1,
            request_bytes=MIB,
            inflight_bytes=MIB,
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.granted_bytes, MIB)

    def test_rate_bucket_refills_without_float_rounding(self) -> None:
        controller = C0Admission(
            C0Config(max_inflight_bytes=100, rate_bytes_per_second=100)
        )
        self.assertTrue(
            controller.try_admit(now_ns=0, request_bytes=100, inflight_bytes=0).admitted
        )
        rejected = controller.try_admit(now_ns=0, request_bytes=1, inflight_bytes=0)
        self.assertEqual(rejected.reason, "rate_cap")
        self.assertTrue(
            controller.try_admit(
                now_ns=10_000_000,
                request_bytes=1,
                inflight_bytes=0,
            ).admitted
        )

    def test_reset_restores_burst_and_counters(self) -> None:
        controller = C0Admission(
            C0Config(max_inflight_bytes=10, rate_bytes_per_second=1)
        )
        controller.try_admit(now_ns=4, request_bytes=10, inflight_bytes=0)
        controller.reset(now_ns=8)
        snapshot = controller.snapshot()
        self.assertEqual(snapshot.available_rate_tokens, 10)
        self.assertEqual(snapshot.admitted_requests, 0)
        self.assertEqual(snapshot.last_update_ns, 8)


if __name__ == "__main__":
    unittest.main()
