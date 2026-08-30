from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_cross_geometry_epoch_v218 as analyzer


class CrossGeometryEpochAnalyzerTest(unittest.TestCase):
    def test_partition_checks_route_geometry(self):
        reason = "same_server_tempo_measured:cache_affinity_warm_hit"
        geometries = ([(512, 16)] * 17 + [(512, 32)] * 7)
        rows = [{"request_id": f"r0-cache-item-{index:02d}",
                 "reason": reason, "prompt_tokens": prompt,
                 "output_tokens": output,
                 "route": (analyzer.base.REMOTE if output == 32
                           else analyzer.base.LOCAL)}
                for index, (prompt, output) in enumerate(geometries)]
        self.assertTrue(analyzer._partition({"router_decisions": rows}, 24, reason))
        rows[-1]["route"] = analyzer.base.LOCAL
        self.assertFalse(analyzer._partition({"router_decisions": rows}, 24, reason))


if __name__ == "__main__":
    unittest.main()
