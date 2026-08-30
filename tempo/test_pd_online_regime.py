import unittest

from tempo.pd_online_regime import (
    HIGH_LOAD_THRESHOLD_NS,
    OnlineRegime,
    OnlineRegimeClassifier,
)


class OnlineRegimeClassifierTests(unittest.TestCase):
    def _feed(self, gaps_ms):
        classifier = OnlineRegimeClassifier()
        now = 1_000_000_000
        classifier.observe(now)
        for gap_ms in gaps_ms:
            now += round(gap_ms * 1_000_000)
            snapshot = classifier.observe(now)
        return classifier, snapshot

    def test_recorded_rate48_router_windows_select_affinity(self) -> None:
        for gaps in (
            [5.969933, 69.237451, 17.394865, 59.292608, 20.961857, 65.240268],
            [20.478948, 64.711013, 20.356540, 60.860293, 21.079743, 64.407945],
        ):
            with self.subTest(gaps=gaps):
                _, snapshot = self._feed(gaps)
                self.assertEqual(snapshot.regime, OnlineRegime.AFFINITY)
                self.assertGreater(snapshot.median_gap_ns, HIGH_LOAD_THRESHOLD_NS)

    def test_recorded_rate52_router_windows_select_local_bypass(self) -> None:
        for gaps in (
            [4.636036, 60.339943, 19.355911, 54.646635, 19.066903, 61.058170],
            [18.965126, 59.549807, 19.070130, 56.184840, 19.947211, 58.350951],
        ):
            with self.subTest(gaps=gaps):
                _, snapshot = self._feed(gaps)
                self.assertEqual(snapshot.regime,
                                 OnlineRegime.HIGH_LOAD_LOCAL_BYPASS)
                self.assertLessEqual(snapshot.median_gap_ns,
                                     HIGH_LOAD_THRESHOLD_NS)

    def test_classification_is_frozen(self) -> None:
        classifier, first = self._feed([37, 37, 37, 37, 37, 37])
        later = classifier.observe(9_000_000_000)
        self.assertEqual(later, first)

    def test_pending_until_seven_observations(self) -> None:
        classifier, snapshot = self._feed([20, 60, 20, 60, 20])
        self.assertEqual(snapshot.regime, OnlineRegime.PENDING)
        self.assertEqual(snapshot.observations, 6)

    def test_nonmonotonic_clock_is_rejected(self) -> None:
        classifier = OnlineRegimeClassifier()
        classifier.observe(10)
        with self.assertRaisesRegex(ValueError, "increase"):
            classifier.observe(10)


if __name__ == "__main__":
    unittest.main()
