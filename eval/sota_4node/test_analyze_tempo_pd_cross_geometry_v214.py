from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_cross_geometry_v214 as analyzer


class CrossGeometryAnalyzerTest(unittest.TestCase):
    def test_expected_failure_set_and_policy_contract(self):
        self.assertTrue(analyzer._only_failed(
            {"gates": {"pass": True, "tradeoff": False}}, {"tradeoff"}))
        self.assertFalse(analyzer._only_failed(
            {"gates": {"pass": True, "other": False}}, {"tradeoff"}))
        self.assertTrue(analyzer._policy_contract())


if __name__ == "__main__":
    unittest.main()
