from pathlib import Path
import unittest
class ProductionPrewarmTest(unittest.TestCase):
 def test_exact_wiring(self):
  root=Path(__file__).parent; node=(root/'vllm_lmcache_same_server_hybrid_controller_node_v166.py').read_text(); launch=(root/'run_tempo_pd_same_server_hybrid_controller_v167_production_prewarm_in_allocation.sh').read_text(); self.assertIn('hybrid_controller_router_v150',node); self.assertIn('cache_catalog_client_v163',node); self.assertEqual(launch.count('srun --exact'),1); self.assertIn(' 56 32 128 8 3000 250 16000',launch)
if __name__=='__main__':unittest.main()
