import unittest
from eval.sota_4node import run_vllm_lmcache_tp16_confirmed_admission_v29_v2_entry as v
class T(unittest.TestCase):
 def test_alias(self):self.assertIsNot(v._V29_AGGREGATE,v._aggregate)
if __name__=="__main__":unittest.main()
