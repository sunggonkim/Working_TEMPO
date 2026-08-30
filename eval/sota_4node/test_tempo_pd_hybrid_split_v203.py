from __future__ import annotations

import unittest
from pathlib import Path

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity_split import calibrated_route


class SplitV203Test(unittest.TestCase):
    def test_2048_output64_is_stably_split(self) -> None:
        self.assertIs(calibrated_route("cache-item-20", 2048, 64),
                      PDRoute.DECODER_LOCAL)
        self.assertIs(calibrated_route("cache-item-21", 2048, 64),
                      PDRoute.REMOTE_PREFILL)
        self.assertIs(calibrated_route("cache-item-02", 512, 32),
                      PDRoute.REMOTE_PREFILL)

    def test_launcher_is_one_bounded_step(self) -> None:
        root = Path(__file__).resolve().parent
        source = (root / "run_tempo_pd_same_server_hybrid_split_v203_in_allocation.sh").read_text()
        self.assertEqual(source.count(" srun "), 1)
        self.assertIn(" 64 32 128 ", source)


if __name__ == "__main__":
    unittest.main()
