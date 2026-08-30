import json
from pathlib import Path
import unittest
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_g14_localrescue950_paced25us_entry as g14

class G14Test(unittest.TestCase):
    def test_exact_contract(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "eval/sota_4node/real_tp16_deadline_g14_localrescue950_paced25us.json"
        self.assertEqual(json.loads(path.read_text()), g14._expected_contract())
    def test_single_factor(self):
        self.assertEqual(g14.f13.PACED_SLEEP_S, 0.00005)
        self.assertEqual(g14._expected_contract()["algorithm"]["post_arm_sleep_us"], 25.0)
        self.assertEqual(g14._expected_contract()["algorithm"]["trigger_ms"], 950.0)

if __name__ == "__main__": unittest.main()
