from __future__ import annotations

import unittest

from eval.sota_4node import tempo_pd_router_v1 as base
from eval.sota_4node import tempo_pd_same_server_production_interleaved_router_v108 as router


def _config() -> base.RouterConfig:
    return base.RouterConfig(
        mode=base.RouterMode.TEMPO_AUTO,
        local_url="http://local", remote_url="http://remote",
        tokenizer_url="http://tokenizer", served_model_name="model",
        model_id="model", model_revision="revision", topology_id="topology",
        remote_backend="lmcache", classifier_version="test",
        decoder_load_bucket="test", kv_bytes_per_token=1024,
    )


class SameServerMultiEpochTest(unittest.TestCase):
    def test_output32_and_output64_can_overlap(self) -> None:
        core = router.ProductionInterleavedCore(_config())
        first = core.decide(
            request_id="ssi-tempo-r0-measured-p32", prompt_tokens=1230,
            output_tokens=32)
        second = core.decide(
            request_id="ssi-tempo-r0-measured-p64", prompt_tokens=1230,
            output_tokens=64)
        self.assertEqual(len(core._tempo_controllers), 2)
        self.assertEqual(set(core._tempo_owned), {
            "ssi-tempo-r0-measured-p32", "ssi-tempo-r0-measured-p64"})
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        core.complete("ssi-tempo-r0-measured-p64")
        core.complete("ssi-tempo-r0-measured-p32")
        self.assertFalse(core._tempo_owned)
        self.assertTrue(all(controller.local_inflight == 0
                            for controller in core._tempo_controllers.values()))


if __name__ == "__main__":
    unittest.main()
