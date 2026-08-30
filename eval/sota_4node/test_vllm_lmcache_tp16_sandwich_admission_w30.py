import json,unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_sandwich_admission_w30_entry as w
class T(unittest.TestCase):
 def test_contract(self):self.assertEqual(json.loads(Path(__file__).with_name("real_tp16_sandwich_admission_w30.json").read_text()),w._expected_contract());self.assertEqual(len(w.BLOCKS),45)
if __name__=="__main__":unittest.main()
