from __future__ import annotations

from pathlib import Path
import unittest


class ParamLauncherTests(unittest.TestCase):
    def test_parameterized_bounded_launcher(self):
        root = Path(__file__).resolve().parent
        launcher = (
            root / "run_tempo_pd_remote_crossover_balanced_param_v12_in_allocation.sh"
        ).read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)
        self.assertIn('"${RATE}" "${WORKERS}" "${OUTPUT_TOKENS}" 3', launcher)
        self.assertIn("remote_crossover_balanced_node_entry_v11.sh", launcher)


if __name__ == "__main__":
    unittest.main()
