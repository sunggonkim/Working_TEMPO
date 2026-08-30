import json
from pathlib import Path
import unittest

from tempo.pd_elastic_profile_v444 import load_elastic_profile


class WeightedLocalProfileTest(unittest.TestCase):
    def test_budget_is_exactly_one_max_row_weight(self):
        root = Path(__file__).resolve().parents[2]
        profile = load_elastic_profile(
            root / "eval/sota_4node/real_tempo_pd_elastic_profile_v447.json")
        maximum = max(row.local_compute_cost_us for row in profile.rows)
        self.assertEqual(profile.controller.local_compute_budget_us, maximum)
        launcher = (root / "eval/sota_4node/run_tempo_pd_elastic_v447_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("sbatch", launcher)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()
