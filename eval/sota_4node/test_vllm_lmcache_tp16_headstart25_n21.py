import json, unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_headstart25_n21_entry as n21
class TestN21(unittest.TestCase):
    def test_contract(self):
        p=Path(__file__).with_name("real_tp16_headstart25_n21.json")
        self.assertEqual(json.loads(p.read_text()),n21._expected_contract())
        self.assertEqual(n21.HEADSTART_MS,25.0)
        self.assertFalse(n21._expected_contract()["algorithm"]["completion_wait_before_decode"])
if __name__=="__main__": unittest.main()
