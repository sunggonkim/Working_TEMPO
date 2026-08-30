from __future__ import annotations

import unittest
from pathlib import Path

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity_tailaware import calibrated_route


class TailAwareV198Test(unittest.TestCase):
    def test_only_2048_output64_moves_local(self) -> None:
        self.assertIs(calibrated_route(2048, 64), PDRoute.DECODER_LOCAL)
        for bucket in ((512, 32), (512, 64), (512, 128)):
            self.assertIs(calibrated_route(*bucket), PDRoute.REMOTE_PREFILL)

    def test_launcher_is_one_bounded_step(self) -> None:
        root = Path(__file__).resolve().parent
        source = (root / "run_tempo_pd_same_server_hybrid_tailaware_v198_in_allocation.sh").read_text()
        self.assertEqual(source.count(" srun "), 1)
        self.assertIn(" 64 32 128 ", source)


if __name__ == "__main__":
    unittest.main()
