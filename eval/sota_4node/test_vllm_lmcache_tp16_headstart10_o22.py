import json,unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_headstart10_o22_entry as o22
class T(unittest.TestCase):
 def test_contract(self):self.assertEqual(json.loads(Path(__file__).with_name("real_tp16_headstart10_o22.json").read_text()),o22._expected_contract())
if __name__=="__main__":unittest.main()
