from __future__ import annotations

import unittest

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity import CacheAffinityCatalog, POLICY_ID


class CompositionAffinityV9Test(unittest.TestCase):
    def test_policy_id_and_mixed_sensitive_buckets(self):
        self.assertEqual(POLICY_ID, "qwen25-7b-tp4x2-warm-affinity-8")
        catalog = CacheAffinityCatalog()
        catalog.seed("cache-item-00", 1230, 16)
        self.assertIs(
            catalog.seed("cache-item-01", 512, 128).route,
            PDRoute.REMOTE_PREFILL,
        )
        self.assertIs(
            catalog.seed("cache-item-02", 2048, 64).route,
            PDRoute.REMOTE_PREFILL,
        )

    def test_single_output_epochs_preserve_calibrated_remote(self):
        for prompt, output in ((512, 128), (2048, 64), (2048, 256)):
            catalog = CacheAffinityCatalog()
            catalog.seed("cache-item-00", 512, output)
            self.assertIs(
                catalog.seed("cache-item-01", prompt, output).route,
                PDRoute.REMOTE_PREFILL,
            )

    def test_non_sensitive_remote_bucket_stays_remote_when_mixed(self):
        catalog = CacheAffinityCatalog()
        catalog.seed("cache-item-00", 1230, 16)
        self.assertIs(
            catalog.seed("cache-item-01", 512, 32).route,
            PDRoute.REMOTE_PREFILL,
        )


if __name__ == "__main__":
    unittest.main()
