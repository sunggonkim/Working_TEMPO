from pathlib import Path
import unittest
class FinalHarnessTest(unittest.TestCase):
 def test_single_production_step(self):
  root=Path(__file__).parent; node=(root/'vllm_lmcache_same_server_hybrid_controller_node_v152.py').read_text(); launch=(root/'run_tempo_pd_same_server_hybrid_controller_v153_in_allocation.sh').read_text(); self.assertIn('hybrid_controller_router_v150',node); self.assertEqual(launch.count('srun --exact'),1); self.assertNotIn('salloc',launch)
if __name__=='__main__':unittest.main()
