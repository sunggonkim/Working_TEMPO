from __future__ import annotations

import unittest

from tempo.domain_counters import CounterSnapshot, counter_delta, validate_counter_series
from tempo.domain_evidence import CounterSupport
from tempo.resource_domain import ResourceDomain


class DomainCounterTests(unittest.TestCase):
    def snap(self, sample: str, timestamp: int, bytes_: int, busy: int) -> CounterSnapshot:
        return CounterSnapshot(
            ResourceDomain.PCIE_HOST, sample, "fixture", timestamp, bytes_, busy,
            CounterSupport.SUPPORTED,
        )

    def test_delta_is_monotonic_and_integer_rate(self) -> None:
        delta = counter_delta(self.snap("a", 10, 100, 2), self.snap("b", 20, 300, 6))
        self.assertEqual((delta.interval_ns, delta.bytes, delta.busy_ns), (10, 200, 4))
        self.assertEqual(delta.bytes_per_second, 50_000_000_000)

    def test_regression_and_timestamp_reuse_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "regressed"):
            counter_delta(self.snap("a", 10, 100, 2), self.snap("b", 20, 90, 2))
        with self.assertRaisesRegex(ValueError, "timestamps"):
            counter_delta(self.snap("a", 10, 100, 2), self.snap("b", 10, 110, 3))

    def test_unsupported_interval_is_not_rate_estimated(self) -> None:
        previous = self.snap("a", 10, 100, 2)
        current = CounterSnapshot(
            ResourceDomain.PCIE_HOST, "b", "fixture", 20, 0, 0, CounterSupport.NOT_COLLECTED
        )
        delta = counter_delta(previous, current)
        self.assertIsNone(delta.bytes_per_second)
        self.assertEqual(delta.support, CounterSupport.NOT_COLLECTED)

    def test_series_rejects_duplicate_samples(self) -> None:
        first = self.snap("a", 10, 100, 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_counter_series([first, first])

    def test_zero_busy_interval_has_no_infinite_rate(self) -> None:
        delta = counter_delta(self.snap("a", 10, 100, 2), self.snap("b", 20, 100, 2))
        self.assertIsNone(delta.bytes_per_second)
        self.assertEqual(delta.bytes, 0)


if __name__ == "__main__":
    unittest.main()
