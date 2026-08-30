from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_output256_balanced_v208 as analyzer


class Output256BalancedAnalyzerTest(unittest.TestCase):
    def test_partition_requires_long_prompts_remote(self):
        reason = "same_server_tempo_measured:cache_affinity_warm_hit"
        rows = []
        for index in range(24):
            long_prompt = index >= 16
            rows.append({"reason": reason,
                         "prompt_tokens": 2048 if long_prompt else 512,
                         "route": (analyzer.base.REMOTE if long_prompt
                                   else analyzer.base.LOCAL)})
        value = {"router_decisions": rows}
        self.assertTrue(analyzer._partition(value, 24, reason))
        rows[-1]["route"] = analyzer.base.LOCAL
        self.assertFalse(analyzer._partition(value, 24, reason))


if __name__ == "__main__":
    unittest.main()
