from __future__ import annotations

from pathlib import Path
import unittest


class ShortCrossoverTests(unittest.TestCase):
    def test_exact_short_workload_and_bounded_launcher(self):
        root = Path(__file__).resolve().parent
        workload = (root / "tempo_pd_short_workload_v14.py").read_text()
        self.assertIn("REPETITIONS = (64, 64, 64)", workload)
        wrapper = (root / "vllm_lmcache_remote_crossover_short_node_v14.py").read_text()
        self.assertIn("context_safe._prepare_workloads = short.prepare", wrapper)
        launcher = (root / "run_tempo_pd_remote_crossover_short_v14_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)
        self.assertIn("16 16 32 3", launcher)


if __name__ == "__main__":
    unittest.main()
