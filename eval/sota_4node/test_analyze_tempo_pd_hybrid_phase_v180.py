from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node import analyze_tempo_pd_hybrid_phase_v180 as analyzer


class HybridPhaseV180Test(unittest.TestCase):
    def test_cold_gates_accept_local_oracle_noninferiority(self) -> None:
        perf = lambda throughput, e2e, tpot: {
            "request_throughput_per_s": throughput,
            "e2e_ms": {"p99": e2e}, "tpot_ms": {"p99": tpot},
        }
        value = {
            "schema": "tempo-pd-same-server-balanced-analysis-71",
            "tempo": {"routes": {analyzer.LOCAL: 48}, "reasons": {"miss": 48},
                      "performance": perf(9.9, 101.0, 20.2)},
            "fixed_local": {"routes": {analyzer.LOCAL: 48},
                            "performance": perf(10.0, 100.0, 20.0)},
            "lmcache_remote": {"routes": {analyzer.REMOTE: 48},
                               "performance": perf(9.0, 120.0, 30.0)},
            "paired_tempo_minus_lmcache": {
                "e2e_win_count": 30, "e2e_delta_median_ms": -10.0},
        }
        _, gates = analyzer._cold_summary(value)
        self.assertTrue(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
