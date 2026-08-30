from __future__ import annotations
import unittest
from eval.sota_4node import compile_lmcache_active_pulse_hybrid_plan as hybrid

class HybridCompilerTests(unittest.TestCase):
    def test_replay_and_provenance(self) -> None:
        artifact = hybrid.make_artifact()
        profile, plan = hybrid.load_artifact(artifact)
        self.assertEqual(len(profile.quanta), 52)
        self.assertEqual(plan.predicted_completion_ns, 89_977_642)
        self.assertEqual(plan.predicted_max_start_lag_ns, 2_165_292)
        self.assertEqual(sum(hybrid.RUNTIME_WIDTH_BY_TOKEN), 64)
        self.assertEqual(tuple(i for i,w in enumerate(hybrid.RUNTIME_WIDTH_BY_TOKEN) if w == 8), (7,12,20))
        provenance = artifact["provenance"]
        self.assertFalse(provenance["selection"]["independent_validation"])
        self.assertEqual(provenance["service_measurements"]["one_mib_ms"]["p99"], 5.253812)
        self.assertEqual(provenance["service_measurements"]["two_mib_ms"]["p99"], 8.892306)

if __name__ == "__main__": unittest.main()
