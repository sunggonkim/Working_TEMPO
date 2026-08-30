from __future__ import annotations

import unittest
from pathlib import Path

from eval.sota_4node.tempo_pd_same_server_hybrid_phase_router_v181 import FullPhaseHybridCore


class HybridPhaseV184Test(unittest.TestCase):
    def test_cold_arm_and_existing_arms(self) -> None:
        self.assertEqual(FullPhaseHybridCore._arm("ssb-tempo-r0-cold-x"),
                         ("tempo", "cold"))
        self.assertEqual(FullPhaseHybridCore._arm("ssb-tempo-r0-warm-cache-item-00"),
                         ("tempo", "warm"))
        self.assertEqual(FullPhaseHybridCore._arm("ssb-tempo-r0-measured-cache-item-00"),
                         ("tempo", "measured"))

    def test_launcher_is_one_bounded_step(self) -> None:
        root = Path(__file__).resolve().parent
        source = (root / "run_tempo_pd_same_server_hybrid_phase_v184_in_allocation.sh").read_text()
        self.assertEqual(source.count(" srun "), 1)
        self.assertIn("hybrid_cold_transition.raw.json", source)


if __name__ == "__main__":
    unittest.main()
