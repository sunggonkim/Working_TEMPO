from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node import analyze_live_pd_paper_evidence_v1 as analyzer


def _fixture(path: Path, delta: float = -10.0) -> Path:
    rows = []
    pairs = []
    for bucket in range(3):
        potential = {"logical_bytes": 1024 * (bucket + 1), "tp4_physical_bytes": 1024 * (bucket + 1)}
        rows.append({
            "bucket": bucket, "prompt_tokens": 10 + bucket, "completion_tokens": 2,
            "prompt_sha256": f"p{bucket}", "output_sha256": f"o{bucket}",
            "potential_kv": potential,
        })
        pairs.append({
            "bucket": bucket, "potential_kv": potential,
            "tempo_route": analyzer.LOCAL_ROUTE, "e2e_delta_ms": delta,
            "ttft_delta_ms": -1.0, "tpot_p99_delta_ms": -2.0,
        })
    report = {
        "schema": analyzer.EXPECTED_SCHEMA,
        "evidence": "actual_vllm_disaggregated_prefill_live_kv",
        "gates": {"valid": True},
        "screen_outcome": "live_pd_candidate_pass",
        "promotion_valid": False,
        "baseline": {"validation": rows},
        "tempo": {"validation": rows},
        "paired": pairs,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


class EvidenceAnalyzerTest(unittest.TestCase):
    def test_summarizes_explicit_runs_and_preserves_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = analyzer.analyze([
                ("a", _fixture(root / "a.json", -10.0)),
                ("b", _fixture(root / "b.json", -20.0)),
            ])
        self.assertEqual(result["paired_validation_count"], 6)
        self.assertEqual(result["aggregate"]["e2e_win_count"], 6)
        self.assertEqual(result["aggregate"]["e2e_delta_median_ms"], -15.0)
        self.assertTrue(result["aggregate"]["all_observed_tempo_routes_reject_remote_pd"])
        self.assertFalse(result["aggregate"]["remote_route_observed"])
        self.assertTrue(any("Mooncake" in item for item in result["claim_boundaries"]))

    def test_rejects_failed_declared_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _fixture(Path(temporary) / "bad.json")
            report = json.loads(path.read_text(encoding="utf-8"))
            report["gates"]["valid"] = False
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declared gates failed"):
                analyzer.analyze([("bad", path)])


if __name__ == "__main__":
    unittest.main()
