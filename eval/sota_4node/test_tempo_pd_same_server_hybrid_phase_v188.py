from __future__ import annotations

import unittest
from pathlib import Path


class HybridPhaseV188Test(unittest.TestCase):
    def test_rate40_is_only_runtime_factor(self) -> None:
        root = Path(__file__).resolve().parent
        source = (root / "run_tempo_pd_same_server_hybrid_phase_v188_rate40_in_allocation.sh").read_text()
        self.assertEqual(source.count(" srun "), 1)
        self.assertIn("same_server_hybrid_phase_node_entry_v183.sh", source)
        self.assertIn(" 40 32 128 ", source)


if __name__ == "__main__":
    unittest.main()
