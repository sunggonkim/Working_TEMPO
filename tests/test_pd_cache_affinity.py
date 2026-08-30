from __future__ import annotations

import unittest

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity import (
    CacheAffinityCatalog, REMOTE_BUCKETS, calibrated_partition, calibrated_route)


class CacheAffinityTest(unittest.TestCase):
    def test_exact_validated_partition(self) -> None:
        partition = calibrated_partition()
        self.assertEqual(partition["request_count"], 24)
        self.assertEqual(partition["remote_request_count"], 8)
        self.assertEqual(partition["prompt_token_work"], 30320)
        self.assertEqual(partition["remote_prompt_token_work"], 7168)
        self.assertAlmostEqual(partition["remote_prompt_token_work_fraction"], 7168 / 30320)

    def test_exact_remote_buckets(self) -> None:
        for prompt in (512, 1230, 2048):
            for output in (16, 32, 64, 128):
                expected = (PDRoute.REMOTE_PREFILL
                            if (prompt, output) in REMOTE_BUCKETS
                            else PDRoute.DECODER_LOCAL)
                self.assertIs(calibrated_route(prompt, output), expected)

    def test_seed_then_hit_is_immutable(self) -> None:
        catalog = CacheAffinityCatalog()
        seeded = catalog.seed("cache-item-02", 512, 32)
        self.assertEqual(catalog.hit("cache-item-02", 512, 32), seeded)
        with self.assertRaisesRegex(ValueError, "changed"):
            catalog.hit("cache-item-02", 512, 64)

    def test_miss_and_unvalidated_geometry_fail_closed(self) -> None:
        catalog = CacheAffinityCatalog()
        with self.assertRaisesRegex(ValueError, "not seeded"):
            catalog.hit("cache-item-00", 512, 16)
        with self.assertRaisesRegex(ValueError, "unvalidated"):
            calibrated_route(4096, 16)


if __name__ == "__main__":
    unittest.main()
