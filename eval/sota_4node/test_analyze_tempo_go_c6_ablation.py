from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_go_c6_ablation as analyzer


class C6AblationAnalyzerTests(unittest.TestCase):
    @staticmethod
    def _summary(*, normal: float, overload_good: int, p99: float) -> dict:
        return {
            "normal_e2e_p50_ms": normal,
            "overload": {
                "slo_good_victims": overload_good,
                "global_rejects": 0,
                "failures": 0,
                "worst_e2e_p99_ms": p99,
            },
            "all_phases": {
                "slo_good_victims": overload_good + 120,
            },
        }

    def test_effect_keeps_tail_and_normal_tradeoff_separate(self) -> None:
        full = self._summary(normal=110.0, overload_good=238, p99=3400.0)
        baseline = self._summary(
            normal=100.0, overload_good=240, p99=5200.0)
        effect = analyzer._effect(full, baseline)
        self.assertAlmostEqual(
            effect["worst_overload_p99_reduction_fraction"], 1 - 3400 / 5200)
        self.assertAlmostEqual(effect["overload_slo_goodput_ratio"], 238 / 240)
        self.assertAlmostEqual(effect["normal_e2e_p50_change_fraction"], 0.10)

    def test_remote_count_requires_official_lmcache_route(self) -> None:
        summary = {
            "route_counts": {
                "normal": {"decoder_local_chunked_prefill": 120},
                "hot_d0": {"official_lmcache_remote_prefill": 119},
                "hot_d1": {"official_lmcache_remote_prefill": 1},
            },
        }
        self.assertEqual(analyzer._remote_requests(summary), 120)


if __name__ == "__main__":
    unittest.main()
