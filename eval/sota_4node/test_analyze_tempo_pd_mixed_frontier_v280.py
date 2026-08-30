import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import analyze_tempo_pd_mixed_frontier_v280 as analyzer


class FrontierAnalyzerTests(unittest.TestCase):
    def test_four_load_frontier_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {}
            for rate in analyzer.VALID_RATES:
                path = root / f"rate{rate}.json"
                path.write_text(json.dumps({
                    "schema": "tempo-pd-mixed-request-crossover-analysis-263",
                    "allocation_id": 7,
                    "passes": True,
                    "route_counts": analyzer.EXPECTED_ROUTES,
                    "pairs": [
                        {"item": item, "e2e_delta_ms": -10 + (item == 23) * 20,
                         "tpot_delta_ms": -1 + (item == 23) * 2}
                        for item in range(24)
                    ],
                }))
                runs[rate] = path
            failure = root / "failure.json"
            failure.write_text(json.dumps({
                "schema": "tempo-pd-mixed-lmcache-failure-analysis-272",
                "allocation_id": 7,
                "request_rate_per_s": 56.0,
                "invalid_streams": 16,
                "verdict": "official_lmcache_concurrent_retrieval_fatal",
            }))
            report = analyzer.analyze(runs, failure, 7)

        self.assertTrue(report["passes"])
        self.assertEqual(report["pooled"]["paired_requests"], 96)
        self.assertEqual(report["pooled"]["e2e_win_count"], 92)
        self.assertEqual(report["frontier"]["highest_valid_offered_rate_per_s"], 52)
        self.assertEqual(report["frontier"]["official_lmcache_fatal_offered_rate_per_s"], 56)

    def test_wrong_route_partition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {}
            for rate in analyzer.VALID_RATES:
                path = root / f"rate{rate}.json"
                path.write_text(json.dumps({
                    "schema": "tempo-pd-mixed-request-crossover-analysis-263",
                    "allocation_id": 7,
                    "passes": True,
                    "route_counts": {"lmcache_remote": 24, "tempo_local": 20,
                                     "tempo_remote": 4},
                    "pairs": [{"item": item, "e2e_delta_ms": -1,
                               "tpot_delta_ms": -1} for item in range(24)],
                }))
                runs[rate] = path
            failure = root / "failure.json"
            failure.write_text("{}")
            with self.assertRaisesRegex(ValueError, "routes"):
                analyzer.analyze(runs, failure, 7)


if __name__ == "__main__":
    unittest.main()
