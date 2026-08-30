from pathlib import Path
import unittest
class Rate60Test(unittest.TestCase):
 def test_only_rate_changes(self):
  t=Path(__file__).with_name('run_tempo_pd_same_server_hybrid_controller_v155_rate60_in_allocation.sh').read_text(); self.assertEqual(t.count('srun --exact'),1); self.assertIn(' 60 32 128 8 3000 250 16000',t); self.assertNotIn('salloc',t)
if __name__=='__main__':unittest.main()
