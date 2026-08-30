import json,unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_confirmed_admission_v29_entry as v
class T(unittest.TestCase):
 def test_contract(self):self.assertEqual(json.loads(Path(__file__).with_name("real_tp16_confirmed_admission_v29.json").read_text()),v._expected_contract())
if __name__=="__main__":unittest.main()
