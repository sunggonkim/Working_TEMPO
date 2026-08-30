from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_load_envelope_v204 as analyzer


class LoadEnvelopeV204Test(unittest.TestCase):
    def test_standard_gates(self) -> None:
        perf = lambda throughput, e2e, tpot: {
            "request_throughput_per_s": throughput,
            "e2e_ms": {"p99": e2e}, "tpot_ms": {"p99": tpot},
        }
        value = {
            "schema": "tempo-pd-production-hybrid-controller-analysis-151",
            "tempo": {"performance": perf(10.1, 99.9, 19.0)},
            "fixed_local": {"performance": perf(10.0, 100.0, 18.0)},
            "lmcache_remote": {"performance": perf(9.0, 120.0, 30.0)},
            "paired_tempo_minus_lmcache": {
                "e2e_win_count": 30, "e2e_delta_median_ms": -10.0},
        }
        _, gates = analyzer._standard(value, 40)
        self.assertTrue(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
