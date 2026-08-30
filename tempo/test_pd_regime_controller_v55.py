from __future__ import annotations

import unittest

from tempo.pd_regime_controller import (
    AdmissionRoute,
    ArrivalRegime,
    PairArrivalRegimeController,
)


class FrozenV55ControllerTest(unittest.TestCase):
    @staticmethod
    def _controller() -> PairArrivalRegimeController:
        return PairArrivalRegimeController(
            high_pair_interval_ns=70_000_000,
            mid_pair_interval_ns=110_000_000,
            calibration_requests=3,
            high_local_inflight_cap=8,
        )

    def test_low_pair_intervals_select_only_local(self):
        controller = self._controller()
        rows = [controller.decide(f"r{i}", i * 125_000_000) for i in range(9)]
        self.assertEqual(controller.regime, ArrivalRegime.LOW)
        self.assertTrue(all(row.route is AdmissionRoute.LOCAL for row in rows))

    def test_mid_trace_with_one_short_interval_stays_local(self):
        controller = self._controller()
        times = [0, 69_000_000, 153_000_000]
        rows = [controller.decide(f"r{i}", value) for i, value in enumerate(times)]
        for i in range(3, 12):
            rows.append(controller.decide(f"r{i}", times[-1] + (i - 2) * 83_000_000))
        self.assertEqual(controller.regime, ArrivalRegime.MID)
        self.assertTrue(all(row.route is AdmissionRoute.LOCAL for row in rows))

    def test_high_trace_selects_eight_local_then_remote_spill(self):
        controller = self._controller()
        times = [0, 49_000_000, 112_000_000]
        rows = [controller.decide(f"r{i}", value) for i, value in enumerate(times)]
        for i in range(3, 12):
            rows.append(controller.decide(f"r{i}", times[-1] + (i - 2) * 62_500_000))
        self.assertEqual(controller.regime, ArrivalRegime.HIGH)
        self.assertEqual([row.route for row in rows[:8]], [AdmissionRoute.LOCAL] * 8)
        self.assertTrue(all(row.route is AdmissionRoute.REMOTE for row in rows[8:]))


if __name__ == "__main__":
    unittest.main()
