from pathlib import Path
import unittest
from eval.sota_4node.tempo_pd_same_server_hybrid_controller_router_v156 import REMOTE_BUCKETS
class DispersedTest(unittest.TestCase):
 def test_same_count_similar_work(self):
  self.assertEqual(len(REMOTE_BUCKETS),4); self.assertEqual(2*sum(p for p,o in REMOTE_BUCKETS),6968)
 def test_launcher(self):
  t=Path(__file__).with_name('run_tempo_pd_same_server_hybrid_controller_v158_dispersed_rate56_in_allocation.sh').read_text(); self.assertEqual(t.count('srun --exact'),1); self.assertIn(' 56 32 128 8 3000 250 16000',t)
if __name__=='__main__':unittest.main()
