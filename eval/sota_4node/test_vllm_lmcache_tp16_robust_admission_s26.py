import json,unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_robust_admission_s26_entry as s
class T(unittest.TestCase):
 def test_contract(self):self.assertEqual(json.loads(Path(__file__).with_name("real_tp16_robust_admission_s26.json").read_text()),s._expected_contract());self.assertEqual(len(s.BLOCKS),21)
if __name__=="__main__":unittest.main()
