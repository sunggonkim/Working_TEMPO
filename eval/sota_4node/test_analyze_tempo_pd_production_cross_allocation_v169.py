from pathlib import Path
import unittest
class CrossAllocationTest(unittest.TestCase):
 def test_fail_closed_contract(self):
  t=Path(__file__).with_name('analyze_tempo_pd_production_cross_allocation_v169.py').read_text();self.assertIn('exactly two distinct allocations',t);self.assertIn('both_allocations_e2e_p99_beat_lmcache',t);self.assertIn('within_0_1pct_local',t)
if __name__=='__main__':unittest.main()
