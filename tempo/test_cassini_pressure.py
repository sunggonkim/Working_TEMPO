from pathlib import Path
import tempfile
import unittest

from tempo import cassini_pressure as pressure


class CassiniPressureSamplerTest(unittest.TestCase):
    def _write(
        self, root: Path, *, timestamp_ns: int,
        rx_pause_1: int, blocked: int, packets: int,
    ) -> None:
        seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
        stamp = f"{seconds}.{nanoseconds:09d}"
        for nic in range(4):
            telemetry = root / f"cxi{nic}" / "device" / "telemetry"
            telemetry.mkdir(parents=True, exist_ok=True)
            values = {
                "hni_rx_paused_0": 1000,
                "hni_rx_paused_1": 1000 + (rx_pause_1 if nic == 0 else 0),
                "hni_tx_paused_0": 1000,
                "hni_tx_paused_1": 1000,
                "parbs_tarb_pi_posted_blocked_cnt": 1000 + blocked,
                "parbs_tarb_pi_posted_pkts": 1000 + packets,
            }
            for name, value in values.items():
                (telemetry / name).write_text(
                    f"{value}@{stamp}\n", encoding="ascii")

    def test_counter_value_preserves_nanosecond_timestamp(self):
        self.assertEqual(
            pressure._counter_value("123@45.000000067\n"),
            (123, 45_000_000_067),
        )

    def test_pause_fraction_and_host_backpressure_ratio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(
                root, timestamp_ns=100_000_000_000,
                rx_pause_1=0, blocked=0, packets=0)
            sampler = pressure.CassiniPressureSampler(
                root, min_interval_ms=0, max_window_ms=1000)
            baseline = sampler.sample(force=True)
            self.assertFalse(baseline["valid"])
            self._write(
                root, timestamp_ns=100_100_000_000,
                rx_pause_1=20_000_000, blocked=1000, packets=100)
            result = sampler.sample(force=True)
            self.assertTrue(result["valid"])
            self.assertAlmostEqual(
                result["rx_pause_fraction_max"], 0.2, places=9)
            self.assertAlmostEqual(
                result["host_blocked_cycles_per_packet_max"], 10.0)
            self.assertEqual(result["nic_count"], 4)

    def test_stale_counter_window_resets_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(
                root, timestamp_ns=100_000_000_000,
                rx_pause_1=0, blocked=0, packets=0)
            sampler = pressure.CassiniPressureSampler(
                root, min_interval_ms=0, max_window_ms=1000)
            sampler.sample(force=True)
            self._write(
                root, timestamp_ns=103_000_000_000,
                rx_pause_1=1, blocked=1, packets=1)
            result = sampler.sample(force=True)
            self.assertFalse(result["valid"])
            self.assertEqual(result["invalid_reason"], "counter_window_stale")

    def test_missing_explicit_nic_counter_fails_at_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                pressure.CassiniPressureSampler(temporary)


if __name__ == "__main__":
    unittest.main()
