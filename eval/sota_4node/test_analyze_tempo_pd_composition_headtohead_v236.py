import unittest

from eval.sota_4node import analyze_tempo_pd_composition_headtohead_v236 as analyzer


class HeadToHeadAnalyzerTest(unittest.TestCase):
    def test_partition_rejects_missing_decisions(self):
        self.assertFalse(analyzer._partition({"requests": []}, 0, "x"))


if __name__ == "__main__":
    unittest.main()
