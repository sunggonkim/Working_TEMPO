from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_prompt4096_phase_v212 as analyzer


class Prompt4096PhaseAnalyzerTest(unittest.TestCase):
    def test_output_geometry_is_exact(self):
        value = {"requests": [
            {"requested_max_tokens": token} for token in ([16] * 12 + [128] * 12)]}
        self.assertTrue(analyzer._outputs(value))
        value["requests"][-1]["requested_max_tokens"] = 64
        self.assertFalse(analyzer._outputs(value))


if __name__ == "__main__":
    unittest.main()
