from pathlib import Path
import unittest
from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as balanced
from eval.sota_4node import run_tempo_pd_same_server_cache_catalog_client_v159 as client
class RemoteFirstTest(unittest.TestCase):
 def test_wrapper_restores_order(self):
  self.assertEqual(balanced._WARM_ORDER,('fixed_local','lmcache_remote','tempo')); self.assertIn('lmcache_remote", "fixed_local", "tempo',Path(client.__file__).read_text())
 def test_launcher(self):
  t=Path(__file__).with_name('run_tempo_pd_same_server_hybrid_controller_v162_remote_first_in_allocation.sh').read_text(); self.assertEqual(t.count('srun --exact'),1); self.assertIn(' 56 32 128 8 3000 250 16000',t)
if __name__=='__main__':unittest.main()
