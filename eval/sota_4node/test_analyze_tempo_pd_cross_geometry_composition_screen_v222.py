from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_cross_geometry_composition_screen_v222 as analyzer


class CompositionScreenAnalyzerTest(unittest.TestCase):
    def test_wrong_schema_fails(self):
        with self.assertRaises(ValueError):
            analyzer.analyze({"schema": "wrong"})


if __name__ == "__main__":
    unittest.main()
