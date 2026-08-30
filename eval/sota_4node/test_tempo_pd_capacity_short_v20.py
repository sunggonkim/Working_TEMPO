from pathlib import Path
import unittest


class CapacityShortV20Tests(unittest.TestCase):
    def test_launcher_reuses_exact_short_reference_and_one_step(self):
        path = Path(__file__).with_name(
            "run_tempo_pd_capacity_short_v20_in_allocation.sh"
        )
        text = path.read_text()
        self.assertEqual(text.count("srun "), 1)
        self.assertIn('16 16 32 3 3000 250 12000', text)
        self.assertIn('crossover_remote/raw.json', text)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
