from __future__ import annotations

import unittest

from eval.sota_4node import analyze_tempo_pd_same_epoch_phase_v186 as analyzer


class SameEpochPhaseV186Test(unittest.TestCase):
    def test_decision_contract(self) -> None:
        rows = [{"reason": "seed", "route": analyzer.LOCAL}] * 16
        rows += [{"reason": "seed", "route": analyzer.REMOTE}] * 8
        self.assertTrue(analyzer._decision_contract(
            {"router_decisions": rows}, count=24, reason="seed", local=16, remote=8))

    def test_request_validation_is_fail_closed(self) -> None:
        self.assertTrue(analyzer._requests_valid(
            {"requests": [{"error": None, "contract_violations": []}]}, 1))
        self.assertFalse(analyzer._requests_valid(
            {"requests": [{"error": "x", "contract_violations": []}]}, 1))


if __name__ == "__main__":
    unittest.main()
