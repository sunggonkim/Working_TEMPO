import unittest

from tempo.pd_online_regime_latched_v380 import (
    LatchedOnlineRegimeClassifier,
    OnlineRegime,
)


class LatchedOnlineRegimeClassifierTest(unittest.TestCase):
    def test_slow_then_burst_then_slow_keeps_only_opportunity_latched(self):
        classifier = LatchedOnlineRegimeClassifier()
        snapshots = [classifier.observe(value) for value in (
            0, 100_000_000, 200_000_000, 300_000_000, 400_000_000,
        )]
        self.assertEqual(snapshots[-1].raw_regime, OnlineRegime.AFFINITY)
        self.assertEqual(snapshots[-1].regime, OnlineRegime.AFFINITY)
        now = 400_000_000
        for _ in range(4):
            now += 14_000_000
            burst = classifier.observe(now)
        self.assertEqual(burst.raw_regime, OnlineRegime.HIGH_LOAD_LOCAL_BYPASS)
        self.assertEqual(burst.regime, OnlineRegime.HIGH_LOAD_LOCAL_BYPASS)
        self.assertTrue(burst.high_load_latched)
        for _ in range(4):
            now += 100_000_000
            sparse = classifier.observe(now)
        self.assertEqual(sparse.raw_regime, OnlineRegime.AFFINITY)
        self.assertEqual(sparse.regime, OnlineRegime.HIGH_LOAD_LOCAL_BYPASS)
        self.assertTrue(sparse.high_load_latched)

    def test_clock_must_strictly_increase(self):
        classifier = LatchedOnlineRegimeClassifier()
        classifier.observe(1)
        with self.assertRaises(ValueError):
            classifier.observe(1)


if __name__ == "__main__":
    unittest.main()
