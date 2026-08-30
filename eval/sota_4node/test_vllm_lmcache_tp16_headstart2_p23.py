import json,unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_headstart2_p23_entry as p
class T(unittest.TestCase):
 def test_contract(self):self.assertEqual(json.loads(Path(__file__).with_name("real_tp16_headstart2_p23.json").read_text()),p._expected_contract())
if __name__=="__main__":unittest.main()
