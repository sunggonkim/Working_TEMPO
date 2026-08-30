from pathlib import Path
import unittest
from eval.sota_4node import tempo_pd_same_server_cache_catalog_router_v146 as router
from tempo.pd_admission import PDRoute
class SparseTest(unittest.TestCase):
 def test_exact_quarter_prompt_work(self):
  rows=[(p,o) for p in (512,1230,2048) for o in (16,32,64,128) for _ in range(2)]
  remote=sum(p for p,o in rows if router._selected_route(p,o) is PDRoute.REMOTE_PREFILL)
  total=sum(p for p,o in rows)
  self.assertEqual(remote,7580); self.assertEqual(total,30320); self.assertEqual(4*remote,total)
 def test_launcher(self):
  t=Path(__file__).with_name('run_tempo_pd_same_server_cache_catalog_v148_in_allocation.sh').read_text(); self.assertEqual(t.count('srun --exact'),1); self.assertNotIn('salloc',t)
if __name__=='__main__': unittest.main()
