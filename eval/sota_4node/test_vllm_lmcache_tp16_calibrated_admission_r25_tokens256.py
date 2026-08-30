import json,unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_calibrated_admission_r25_tokens256_entry as r
class T(unittest.TestCase):
 def test_contract(self):self.assertEqual(json.loads(Path(__file__).with_name("real_tp16_calibrated_admission_r25_tokens256.json").read_text()),r._expected_contract())
if __name__=="__main__":unittest.main()
