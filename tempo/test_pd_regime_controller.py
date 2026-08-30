from __future__ import annotations

import unittest

from tempo.pd_regime_controller import (
    AdmissionRoute,
    ArrivalRegime,
    PairArrivalRegimeController,
)


class PairArrivalRegimeControllerTest(unittest.TestCase):
    def _routes(self, interval_ns: int, count: int = 12):
        controller = PairArrivalRegimeController()
        decisions = [
            controller.decide(f"r{index}", index * interval_ns)
            for index in range(count)
        ]
        return controller, decisions

    def test_low_rate_stays_local(self):
        controller, decisions = self._routes(125_000_000, 9)
        self.assertEqual(controller.regime, ArrivalRegime.LOW)
        self.assertTrue(all(row.route is AdmissionRoute.LOCAL for row in decisions))

    def test_mid_rate_stays_local(self):
        controller, decisions = self._routes(83_000_000)
        self.assertEqual(controller.regime, ArrivalRegime.MID)
        self.assertTrue(all(row.route is AdmissionRoute.LOCAL for row in decisions))

    def test_high_rate_spills_after_eight_live_local_requests(self):
        controller, decisions = self._routes(62_500_000)
        self.assertEqual(controller.regime, ArrivalRegime.HIGH)
        self.assertEqual([row.route for row in decisions[:8]], [AdmissionRoute.LOCAL] * 8)
        self.assertEqual([row.route for row in decisions[8:]], [AdmissionRoute.REMOTE] * 4)

    def test_release_returns_high_rate_local_credit(self):
        controller, decisions = self._routes(62_500_000, 9)
        self.assertEqual(decisions[-1].route, AdmissionRoute.REMOTE)
        controller.release("r0")
        next_decision = controller.decide("r9", 9 * 62_500_000)
        self.assertEqual(next_decision.route, AdmissionRoute.LOCAL)

    def test_workload_guard_forces_local_even_after_credit_exhaustion(self):
        controller, decisions = self._routes(62_500_000, 9)
        self.assertEqual(decisions[-1].route, AdmissionRoute.REMOTE)
        guarded = controller.decide("guarded", 9 * 62_500_000, force_local=True)
        self.assertEqual(guarded.route, AdmissionRoute.LOCAL)
        self.assertEqual(guarded.reason, "workload_guard_local")
        controller.release("guarded")

    def test_rejects_duplicate_and_nonmonotonic_time(self):
        controller = PairArrivalRegimeController()
        controller.decide("r0", 1)
        with self.assertRaises(ValueError):
            controller.decide("r0", 2)
        with self.assertRaises(ValueError):
            controller.decide("r1", 1)


if __name__ == "__main__":
    unittest.main()
