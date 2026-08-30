import json,unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_rolling_admission_t27_entry as t
class T(unittest.TestCase):
 def test_contract(self):self.assertEqual(json.loads(Path(__file__).with_name("real_tp16_rolling_admission_t27.json").read_text()),t._expected_contract());self.assertEqual(len(t.BLOCKS),27)
if __name__=="__main__":unittest.main()
