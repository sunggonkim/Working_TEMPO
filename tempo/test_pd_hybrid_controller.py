from __future__ import annotations

import unittest

from tempo.pd_admission import PDRoute
from tempo.pd_hybrid_controller import CachePhase, HybridPDController


class HybridControllerTest(unittest.TestCase):
    def test_warm_affinity_is_seeded_then_reused(self):
        core = HybridPDController()
        seed = core.decide(request_id="seed", prompt_tokens=512, output_tokens=32,
                           now_ns=1, cache_phase=CachePhase.WARM_SEED,
                           cache_item="cache-item-02")
        self.assertIs(seed.route, PDRoute.REMOTE_PREFILL)
        core.complete("seed")
        hit = core.decide(request_id="hit", prompt_tokens=512, output_tokens=32,
                          now_ns=2, cache_phase=CachePhase.WARM_HIT,
                          cache_item="cache-item-02")
        self.assertIs(hit.route, seed.route)

    def test_cache_hit_without_same_pair_seed_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "not seeded"):
            HybridPDController().decide(
                request_id="hit", prompt_tokens=512, output_tokens=16,
                now_ns=1, cache_phase=CachePhase.WARM_HIT,
                cache_item="cache-item-00")

    def test_cold_direct_local_and_controller_paths(self):
        core = HybridPDController()
        direct = core.decide(
            request_id="direct", prompt_tokens=4096, output_tokens=128,
            now_ns=1, cache_phase=CachePhase.MISS)
        self.assertIs(direct.route, PDRoute.DECODER_LOCAL)
        core.complete("direct")
        controlled = core.decide(
            request_id="controlled", prompt_tokens=2048, output_tokens=32,
            now_ns=2, cache_phase=CachePhase.MISS)
        self.assertIs(controlled.route, PDRoute.DECODER_LOCAL)
        core.complete("controlled")

    def test_output256_cold_is_local_and_warm_even_item_is_remote(self):
        core = HybridPDController()
        cold = core.decide(
            request_id="cold256", prompt_tokens=2048, output_tokens=256,
            now_ns=1, cache_phase=CachePhase.MISS)
        self.assertIs(cold.route, PDRoute.DECODER_LOCAL)
        core.complete("cold256")
        seed = core.decide(
            request_id="seed256", prompt_tokens=2048, output_tokens=256,
            now_ns=2, cache_phase=CachePhase.WARM_SEED,
            cache_item="cache-item-22")
        self.assertIs(seed.route, PDRoute.REMOTE_PREFILL)
        core.complete("seed256")
        hit = core.decide(
            request_id="hit256", prompt_tokens=2048, output_tokens=256,
            now_ns=3, cache_phase=CachePhase.WARM_HIT,
            cache_item="cache-item-22")
        self.assertIs(hit.route, PDRoute.REMOTE_PREFILL)
        core.complete("hit256")

    def test_unvalidated_cold_and_warm_workloads_fail_closed(self):
        core = HybridPDController()
        with self.assertRaises(ValueError):
            core.decide(request_id="cold", prompt_tokens=4097, output_tokens=16,
                        now_ns=1, cache_phase=CachePhase.MISS)
        with self.assertRaises(ValueError):
            core.decide(request_id="warm", prompt_tokens=4096, output_tokens=32,
                        now_ns=2, cache_phase=CachePhase.WARM_SEED,
                        cache_item="cache-item-00")


if __name__ == "__main__": unittest.main()
