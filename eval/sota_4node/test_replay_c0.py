from __future__ import annotations

import unittest

from eval.sota_4node.replay_c0 import simulate_tandem


class C0ReplayTests(unittest.TestCase):
    def test_tandem_replay_respects_arrival_and_service_rates(self) -> None:
        result = simulate_tandem(
            demand_batches=[(0, 8)],
            state_bytes=8,
            file_bytes=8,
            d2h_rate_bps=8,
            pfs_rate_bps=8,
            d2h_request_bytes=4,
            pfs_request_bytes=4,
            finalization_reserve_ns=100,
        )
        self.assertEqual(result["d2h_finish_ns"], 1_000_000_000)
        self.assertEqual(result["pfs_finish_ns"], 1_500_000_000)
        self.assertEqual(result["predicted_completion_ns"], 1_500_000_100)
        self.assertEqual(result["d2h_requests"], 2)
        self.assertEqual(result["pfs_requests"], 2)

    def test_replay_rejects_incomplete_demand(self) -> None:
        with self.assertRaisesRegex(ValueError, "D2H demand sums"):
            simulate_tandem(
                demand_batches=[(0, 7)],
                state_bytes=8,
                file_bytes=8,
                d2h_rate_bps=8,
                pfs_rate_bps=8,
                d2h_request_bytes=4,
                pfs_request_bytes=4,
                finalization_reserve_ns=0,
            )


if __name__ == "__main__":
    unittest.main()
