import json
from pathlib import Path
import unittest
from eval.sota_4node import run_vllm_lmcache_tp16_deadline_g14_localrescue950_paced25us_entry as g14
class G14V2Test(unittest.TestCase):
    def test_exact_contract(self):
        path = Path(__file__).with_name("real_tp16_deadline_g14v2_localrescue950_paced25us.json")
        self.assertEqual(json.loads(path.read_text()), g14._expected_contract())
if __name__ == "__main__": unittest.main()
