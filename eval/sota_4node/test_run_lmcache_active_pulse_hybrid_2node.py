from __future__ import annotations
import unittest
from eval.sota_4node import compile_lmcache_active_pulse_hybrid_plan as compiled
from eval.sota_4node import run_lmcache_active_pulse_hybrid_2node as runner

class HybridRunnerTests(unittest.TestCase):
    def test_exact_canonical_expansion(self) -> None:
        profile = compiled.frozen_profile(); logical = compiled.replay_hybrid(profile)
        _, runtime = runner._adapt(profile, logical)
        self.assertEqual(runtime.width_by_token, compiled.RUNTIME_WIDTH_BY_TOKEN)
        self.assertEqual(tuple(i for token in runtime.quantum_indices_by_token for i in token), tuple(range(64)))
        self.assertEqual(runtime.quantum_indices_by_token[7], tuple(range(8,16)))
        self.assertEqual(runtime.completion_token_exclusive, 29)

if __name__ == "__main__": unittest.main()
