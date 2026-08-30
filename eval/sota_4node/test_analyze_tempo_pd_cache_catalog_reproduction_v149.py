from pathlib import Path
import unittest

from tempo.pd_cache_affinity import calibrated_partition


class ReproductionContractTest(unittest.TestCase):
    def test_production_partition_is_frozen(self):
        value = calibrated_partition()
        self.assertEqual(value["remote_request_count"], 8)
        self.assertEqual(value["remote_prompt_token_work"], 7168)

    def test_analyzer_has_two_lifecycle_boundary(self):
        path = Path(__file__).with_name(
            "analyze_tempo_pd_cache_catalog_reproduction_v149.py")
        text = path.read_text(encoding="utf-8")
        self.assertIn("exactly two distinct lifecycle reports required", text)
        self.assertIn("one four-node A100 allocation", text)


if __name__ == "__main__": unittest.main()
