from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_output256_phase_v206 as analyzer


class Output256PhaseAnalyzerTest(unittest.TestCase):
    def test_request_and_route_contracts_are_fail_closed(self):
        reason = "same_server_tempo_warm:cache_affinity_warm_seed"
        value = {
            "requests": [{"error": None, "contract_violations": []}],
            "router_decisions": [{"route": analyzer.LOCAL, "reason": reason}],
        }
        self.assertTrue(analyzer._valid_requests(value, 1))
        self.assertTrue(analyzer._route_contract(value, 1, reason))
        value["router_decisions"][0]["route"] = analyzer.REMOTE
        self.assertFalse(analyzer._route_contract(value, 1, reason))
        value["requests"][0]["error"] = "boom"
        self.assertFalse(analyzer._valid_requests(value, 1))


if __name__ == "__main__":
    unittest.main()
