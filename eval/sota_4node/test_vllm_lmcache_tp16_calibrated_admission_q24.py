import json,unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_calibrated_admission_q24_entry as q
class T(unittest.TestCase):
 def test_contract(self):self.assertEqual(json.loads(Path(__file__).with_name("real_tp16_calibrated_admission_q24.json").read_text()),q._expected_contract())
if __name__=="__main__":unittest.main()
