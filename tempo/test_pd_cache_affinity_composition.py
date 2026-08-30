from __future__ import annotations

import unittest

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity import CacheAffinityCatalog


class CompositionAffinityTest(unittest.TestCase):
    def test_output256_only_keeps_long_prompt_remote(self):
        catalog = CacheAffinityCatalog()
        catalog.seed("cache-item-00", 512, 256)
        placement = catalog.seed("cache-item-01", 2048, 256)
        self.assertIs(placement.route, PDRoute.REMOTE_PREFILL)
        self.assertEqual(catalog.hit("cache-item-01", 2048, 256), placement)

    def test_mixed_remote_history_defers_long_output256_to_local(self):
        catalog = CacheAffinityCatalog()
        catalog.seed("cache-item-00", 512, 32)
        placement = catalog.seed("cache-item-01", 2048, 256)
        self.assertIs(placement.route, PDRoute.DECODER_LOCAL)
        self.assertEqual(catalog.hit("cache-item-01", 2048, 256), placement)


if __name__ == "__main__":
    unittest.main()
