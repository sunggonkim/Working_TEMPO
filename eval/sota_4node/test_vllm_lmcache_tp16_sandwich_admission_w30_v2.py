import unittest
from eval.sota_4node import run_vllm_lmcache_tp16_sandwich_admission_w30_v2_entry as w
class T(unittest.TestCase):
 def test_alias(self):self.assertIsNot(w._W30_AGGREGATE,w._aggregate)
if __name__=="__main__":unittest.main()
