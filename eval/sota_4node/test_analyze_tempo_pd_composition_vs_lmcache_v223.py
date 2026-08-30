import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.analyze_tempo_pd_composition_vs_lmcache_v223 import analyze


def _arm(throughput: float, e2e: float, tpot: float, delta: float = 0.0):
    return {
        "model_config_sha256": "model",
        "performance": {
            "request_throughput_per_s": throughput,
            "e2e_ms": {"p99": e2e},
            "tpot_ms": {"p99": tpot},
        },
        "request_metrics": [
            {"request_id": f"r{i}", "workload_fingerprint": f"w{i}", "e2e_ms": 100.0 + i + delta}
            for i in range(48)
        ],
    }


class AnalyzerTest(unittest.TestCase):
    def test_anchored_advantage(self):
        with tempfile.TemporaryDirectory(prefix="job_7_") as root:
            root_path = Path(root)
            old_path = root_path / "old_job_7.json"
            new_path = root_path / "new_job_7.json"
            old_path.write_text(__import__("json").dumps({
                "schema": "tempo-pd-production-hybrid-controller-analysis-151",
                "lmcache_remote": _arm(10.0, 110.0, 20.0, 10.0),
                "fixed_local": _arm(10.0, 100.0, 10.0),
            }))
            new_path.write_text(__import__("json").dumps({
                "schema": "tempo-pd-hybrid-saturation-analysis-192",
                "tempo": _arm(11.0, 90.0, 9.0),
                "fixed_local_primary": _arm(10.0, 100.0, 10.0),
            }))
            report = analyze(old_path, new_path, allocation_id="7")
        self.assertTrue(report["passes"])
        self.assertEqual(report["direct"]["paired_e2e_win_count"], 48)
        self.assertIn("separate server lifecycles", report["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
