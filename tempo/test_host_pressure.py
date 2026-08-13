from __future__ import annotations

import unittest

from tempo.host_pressure import (
    MIN_BUFFER_BYTES,
    HostPressureSample,
    HostPressureSpec,
    host_pressure_route_is_placebo,
    validate_host_pressure_series,
)


def _spec() -> HostPressureSpec:
    return HostPressureSpec(
        rank=0,
        world_size=4,
        numa_node=3,
        buffer_bytes=MIN_BUFFER_BYTES,
        duration_ns=10_000,
        sample_period_ns=1_000,
    )


def _samples() -> list[HostPressureSample]:
    return [
        HostPressureSample("start", 100, 0, 0, 0),
        HostPressureSample("end", 10_100, MIN_BUFFER_BYTES, 9_000, MIN_BUFFER_BYTES),
    ]


class HostPressureTests(unittest.TestCase):
    def test_valid_series_is_monotonic_and_buffer_backed(self) -> None:
        result = validate_host_pressure_series(_spec(), _samples())
        self.assertEqual(result[-1].cumulative_touched_bytes, MIN_BUFFER_BYTES)

    def test_placebo_has_no_auxiliary_route(self) -> None:
        self.assertTrue(host_pressure_route_is_placebo(()))
        self.assertFalse(host_pressure_route_is_placebo(("host_numa",)))

    def test_world_and_buffer_contract_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "four-rank"):
            HostPressureSpec(0, 1, 0, MIN_BUFFER_BYTES, 10, 1)
        with self.assertRaisesRegex(ValueError, "minimum"):
            HostPressureSpec(0, 4, 0, MIN_BUFFER_BYTES - 1, 10, 1)

    def test_counter_regression_is_rejected(self) -> None:
        samples = _samples()
        samples[1] = HostPressureSample("end", 10_100, MIN_BUFFER_BYTES - 1, 9_000, MIN_BUFFER_BYTES)
        with self.assertRaisesRegex(ValueError, "did not touch"):
            validate_host_pressure_series(_spec(), samples)

    def test_timestamp_and_duplicate_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timestamps"):
            validate_host_pressure_series(
                _spec(),
                [HostPressureSample("a", 2, 0, 0, 0), HostPressureSample("b", 2, MIN_BUFFER_BYTES, 1, MIN_BUFFER_BYTES)],
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_host_pressure_series(_spec(), [_samples()[0], _samples()[0], _samples()[1]])


if __name__ == "__main__":
    unittest.main()
