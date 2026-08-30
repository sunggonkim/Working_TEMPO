import unittest
from eval.sota_4node import tempo_pd_queue_crossover_router_v40 as router
from eval.sota_4node import tempo_pd_router_v1 as base


class QueueCrossoverV41Tests(unittest.TestCase):
    def test_first_eight_local_then_remote(self):
        config=base.RouterConfig(mode=base.RouterMode.TEMPO_AUTO,local_url="l",remote_url="r",tokenizer_url="t",served_model_name="m",model_id="m",model_revision="x",topology_id="t",remote_backend="b",classifier_version="c",decoder_load_bucket="high",kv_bytes_per_token=1)
        core=router.QueueCrossoverCore(config)
        rows=[core.decide(request_id=str(i),prompt_tokens=1220,output_tokens=32) for i in range(12)]
        self.assertEqual(sum(r.route.value=="decoder_local_recompute_or_cache" for r in rows),8)
        self.assertEqual(sum(r.route.value=="remote_prefill_live_kv" for r in rows),4)


if __name__=="__main__": unittest.main()
