from pathlib import Path
import unittest
from eval.sota_4node import tempo_pd_router_v1 as base
from eval.sota_4node import tempo_pd_same_server_hybrid_cold_router_v170 as router
def config():return base.RouterConfig(mode=base.RouterMode.TEMPO_AUTO,local_url='http://l',remote_url='http://r',tokenizer_url='http://t',served_model_name='m',model_id='m',model_revision='r',topology_id='t',remote_backend='UCX',classifier_version='v',decoder_load_bucket='d',kv_bytes_per_token=1)
class ColdTest(unittest.TestCase):
 def test_direct_cold_path(self):
  c=router.HybridColdCore(config());d=c.decide(request_id='ssb-tempo-r0-warm-x',prompt_tokens=512,output_tokens=128);self.assertEqual(d.reason,'same_server_tempo_warm:hybrid_cold:output128_direct_local_fast_path');c.complete(d.request_id)
 def test_launcher(self):
  t=Path(__file__).with_name('run_tempo_pd_same_server_hybrid_cold_v173_in_allocation.sh').read_text();self.assertEqual(t.count('srun --exact'),1);self.assertIn(' 48 32 128 8 3000 250 16000',t)
if __name__=='__main__':unittest.main()
