from pathlib import Path
import unittest
from eval.sota_4node import tempo_pd_same_server_cache_catalog_router_v142 as router
from tempo.pd_admission import PDRoute

class LowKVTest(unittest.TestCase):
    def test_only_512_is_remote(self):
        for output in (16,32,64,128):
            self.assertIs(router._selected_route(512, output), PDRoute.REMOTE_PREFILL)
            self.assertIs(router._selected_route(2048, output), PDRoute.DECODER_LOCAL)
    def test_single_step(self):
        text=Path(__file__).with_name('run_tempo_pd_same_server_cache_catalog_v144_in_allocation.sh').read_text()
        self.assertEqual(text.count('srun --exact'),1); self.assertNotIn('salloc',text)

if __name__ == '__main__': unittest.main()
