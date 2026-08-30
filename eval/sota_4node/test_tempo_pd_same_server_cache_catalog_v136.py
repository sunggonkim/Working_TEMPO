from __future__ import annotations

import unittest

from eval.sota_4node import run_tempo_pd_same_server_cache_catalog_client_v136 as client
from eval.sota_4node import tempo_pd_same_server_cache_catalog_router_v136 as router
from tempo.pd_admission import PDRoute


class CacheCatalogTest(unittest.TestCase):
    def test_request_identity_is_stable_across_phase_and_replicate(self) -> None:
        rows = [{"request_id": "arbitrary", "prompt": "nonce 001. payload",
                 "max_tokens": 16}]
        warm = client._derive(rows, prefix="ssb-tempo-r0-warm-", offset=0)
        measured = client._derive(rows, prefix="ssb-tempo-r1-measured-", offset=0)
        self.assertEqual(router._cache_item(warm[0]["request_id"]), "cache-item-00")
        self.assertEqual(router._cache_item(measured[0]["request_id"]), "cache-item-00")
        self.assertEqual(warm[0]["prompt"], measured[0]["prompt"])

    def test_frozen_cache_hit_crossover(self) -> None:
        remote = {(512, 32), (512, 64), (512, 128), (2048, 64)}
        for prompt in (512, 1230, 2048):
            for output in (16, 32, 64, 128):
                expected = (PDRoute.REMOTE_PREFILL if (prompt, output) in remote
                            else PDRoute.DECODER_LOCAL)
                self.assertIs(router._selected_route(prompt, output), expected)

    def test_unvalidated_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt"):
            router._selected_route(513, 32)
        with self.assertRaisesRegex(ValueError, "output"):
            router._selected_route(512, 256)


if __name__ == "__main__":
    unittest.main()
