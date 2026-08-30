import unittest

from tempo.pd_admission import PDRoute
from tempo.pd_warm_regime_controller import (
    WarmLoadRegime,
    WarmRegimeController,
)


class WarmRegimeControllerTests(unittest.TestCase):
    def test_affinity_regime_preserves_policy8_remote_route(self) -> None:
        controller = WarmRegimeController(48)
        seed = controller.seed("cache-item-00", 512, 64)
        hit = controller.hit("cache-item-00", 512, 64)
        self.assertEqual(seed.regime, WarmLoadRegime.AFFINITY)
        self.assertEqual(seed.route, PDRoute.REMOTE_PREFILL)
        self.assertEqual(hit, seed)

    def test_rate52_bypasses_remote_affinity_route(self) -> None:
        controller = WarmRegimeController(52)
        seed = controller.seed("cache-item-00", 512, 64)
        hit = controller.hit("cache-item-00", 512, 64)
        self.assertEqual(seed.regime, WarmLoadRegime.HIGH_LOAD_LOCAL_BYPASS)
        self.assertEqual(seed.affinity_route, PDRoute.REMOTE_PREFILL)
        self.assertEqual(seed.route, PDRoute.DECODER_LOCAL)
        self.assertEqual(hit, seed)

    def test_rate52_keeps_local_affinity_local(self) -> None:
        decision = WarmRegimeController(52).seed("cache-item-00", 1230, 32)
        self.assertEqual(decision.affinity_route, PDRoute.DECODER_LOCAL)
        self.assertEqual(decision.route, PDRoute.DECODER_LOCAL)

    def test_unknown_load_fails_closed(self) -> None:
        for rate in (0, 50, 56, float("nan")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                WarmRegimeController(rate)

    def test_hit_still_requires_seed_and_exact_geometry(self) -> None:
        controller = WarmRegimeController(52)
        with self.assertRaises(ValueError):
            controller.hit("cache-item-00", 512, 64)
        controller.seed("cache-item-00", 512, 64)
        with self.assertRaises(ValueError):
            controller.hit("cache-item-00", 512, 128)


if __name__ == "__main__":
    unittest.main()
