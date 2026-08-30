from pathlib import Path
import unittest


class UniqueHeadR32V27Tests(unittest.TestCase):
    def test_exact_load_and_one_step(self):
        text = Path(__file__).with_name(
            "run_tempo_pd_unique_head_r32_v27_in_allocation.sh"
        ).read_text()
        self.assertIn("32 32 32 3 3000 250 12000", text)
        self.assertEqual(text.count("srun "), 1)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
