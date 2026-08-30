from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_output256_halfsplit_v210 as analyzer


class Output256HalfSplitAnalyzerTest(unittest.TestCase):
    def test_partition_is_stable_by_cache_item(self):
        reason = "same_server_tempo_measured:cache_affinity_warm_hit"
        rows = []
        for index in range(24):
            long_prompt = index >= 16
            remote = long_prompt and index % 2 == 0
            rows.append({"request_id": f"r0-cache-item-{index:02d}",
                         "reason": reason,
                         "prompt_tokens": 2048 if long_prompt else 512,
                         "route": (analyzer.base.REMOTE if remote
                                   else analyzer.base.LOCAL)})
        value = {"router_decisions": rows}
        self.assertTrue(analyzer._partition(value, 24, reason))
        rows[-2]["route"] = analyzer.base.LOCAL
        self.assertFalse(analyzer._partition(value, 24, reason))


if __name__ == "__main__":
    unittest.main()
