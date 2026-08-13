import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.build_g1_composite_readiness import build_composite_readiness


RAW = Path(__file__).resolve().parents[2] / "results" / "sota_4node" / "g1_tier_job_56824614"


class CompositeReadinessTests(unittest.TestCase):
    def test_current_raw_is_composite_observed_but_not_fine_promotion(self):
        if not RAW.is_dir():
            self.skipTest("raw G1 artifact is not present")
        result = build_composite_readiness(RAW)
        self.assertEqual(result["status"], "observed_composite")
        self.assertTrue(result["fabric_followup_eligible"])
        self.assertFalse(result["promotion_ready"])
        self.assertFalse(result["fine_domain_promotion"])
        self.assertFalse(result["hardware_counter_claim"])
        self.assertGreater(result["signals"]["d2h_composite"]["stage_bytes"], 0)
        self.assertGreater(result["signals"]["persistent_composite"]["stage_bytes"], 0)

    def test_d2h_persist_isolation_is_exact(self):
        if not RAW.is_dir():
            self.skipTest("raw G1 artifact is not present")
        result = build_composite_readiness(RAW)
        self.assertEqual(result["logical_stage_by_mode"]["d2h_only"]["pfs"]["group_max_bytes"], 0)
        self.assertEqual(result["logical_stage_by_mode"]["persist_only"]["d2h"]["group_max_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
