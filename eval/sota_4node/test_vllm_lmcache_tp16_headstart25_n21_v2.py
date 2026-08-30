import unittest
from eval.sota_4node import run_vllm_lmcache_tp16_headstart25_n21_v2_entry as n21v2
class TestN21V2(unittest.TestCase):
    def test_entry_callable(self): self.assertTrue(callable(n21v2._aggregate))
if __name__=="__main__": unittest.main()
