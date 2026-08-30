from __future__ import annotations

import unittest

from eval.sota_4node import tempo_pd_router_v1 as base
from eval.sota_4node import tempo_pd_same_server_hybrid_controller_router_v150 as router
from tempo.pd_admission import PDRoute


def _config():
    return base.RouterConfig(
        mode=base.RouterMode.TEMPO_AUTO, local_url="http://local",
        remote_url="http://remote", tokenizer_url="http://tokenizer",
        served_model_name="model", model_id="model", model_revision="revision",
        topology_id="topology", remote_backend="UCX",
        classifier_version="exact", decoder_load_bucket="test",
        kv_bytes_per_token=1)


class ProductionRouterTest(unittest.TestCase):
    def test_warm_then_measured_uses_same_route(self):
        core = router.ProductionHybridCore(_config())
        warm = core.decide(
            request_id="ssb-tempo-r0-warm-cache-item-02",
            prompt_tokens=512, output_tokens=32)
        self.assertIs(warm.route, PDRoute.REMOTE_PREFILL)
        core.complete(warm.request_id)
        measured = core.decide(
            request_id="ssb-tempo-r1-measured-cache-item-02",
            prompt_tokens=512, output_tokens=32)
        self.assertIs(measured.route, warm.route)
        core.complete(measured.request_id)

    def test_measured_item_on_unseeded_pair_fails(self):
        core = router.ProductionHybridCore(_config())
        with self.assertRaisesRegex(ValueError, "not seeded"):
            core.decide(request_id="ssb-tempo-r0-measured-cache-item-00",
                        prompt_tokens=512, output_tokens=16)

    def test_fixed_arms_remain_exact(self):
        core = router.ProductionHybridCore(_config())
        local = core.decide(request_id="ssb-local-r0-warm-cache-item-00",
                            prompt_tokens=512, output_tokens=16)
        remote = core.decide(request_id="ssb-remote-r0-warm-cache-item-00",
                             prompt_tokens=512, output_tokens=16)
        self.assertIs(local.route, PDRoute.DECODER_LOCAL)
        self.assertIs(remote.route, PDRoute.REMOTE_PREFILL)


if __name__ == "__main__": unittest.main()
