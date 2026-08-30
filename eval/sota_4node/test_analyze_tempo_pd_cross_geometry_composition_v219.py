from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_cross_geometry_composition_v219 as analyzer


class CompositionEpochAnalyzerTest(unittest.TestCase):
    def test_missing_request_contract_fails(self):
        self.assertFalse(analyzer._partition({"requests": []}, 24, "reason"))


if __name__ == "__main__":
    unittest.main()
