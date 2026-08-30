import json,unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_hysteresis_admission_u28_entry as u
class T(unittest.TestCase):
 def test_contract(self):self.assertEqual(json.loads(Path(__file__).with_name("real_tp16_hysteresis_admission_u28.json").read_text()),u._expected_contract());self.assertEqual(u.HYSTERESIS_MS,50)
if __name__=="__main__":unittest.main()
