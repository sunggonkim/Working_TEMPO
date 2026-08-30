from pathlib import Path
import unittest
from eval.sota_4node import run_tempo_pd_same_server_cache_catalog_client_v163 as client
class PrewarmTest(unittest.TestCase):
 def test_serial_remote_prewarm_is_bounded(self):
  t=Path(client.__file__).read_text(); self.assertIn('"--max-workers", "1"',t); self.assertIn('timeout=180.0',t); self.assertIn('lmcache_always_remote',t)
 def test_launcher(self):
  t=Path(__file__).with_name('run_tempo_pd_same_server_hybrid_controller_v165_transport_prewarm_in_allocation.sh').read_text(); self.assertEqual(t.count('srun --exact'),1); self.assertIn(' 56 32 128 8 3000 250 16000',t)
if __name__=='__main__':unittest.main()
