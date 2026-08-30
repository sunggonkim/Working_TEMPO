import unittest

from tempo.pd_online_regime_fast_v311 import OnlineRegime, OnlineRegimeClassifier


class FastClassifierTests(unittest.TestCase):
    def _classify(self, gaps):
        classifier = OnlineRegimeClassifier()
        now = 0
        classifier.observe(now)
        for gap in gaps:
            now += int(gap)
            snapshot = classifier.observe(now)
        return snapshot

    def test_recorded_rate48_pairs_choose_affinity(self):
        for gaps in ([39947585, 39947586, 39947585, 39947586],
                     [40894974, 40894975, 40894974, 40894975]):
            self.assertIs(self._classify(gaps).regime, OnlineRegime.AFFINITY)

    def test_recorded_rate52_pairs_choose_high_load(self):
        for gaps in ([37328460] * 4, [38035407, 38035408] * 2):
            snapshot = self._classify(gaps)
            self.assertIs(snapshot.regime, OnlineRegime.HIGH_LOAD_LOCAL_BYPASS)
            self.assertEqual(snapshot.observations, 5)


if __name__ == "__main__":
    unittest.main()
