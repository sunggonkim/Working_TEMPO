from pathlib import Path
import tempfile
import unittest

from tempo import cassini_endpoint as endpoint
from tempo.domain_evidence import CounterSupport
from tempo.pd_endpoint_evidence import PDEndpointIdentity, PDEndpointRole


CORE_COUNTERS = set(endpoint._CORE_COUNTERS)
OPTIONAL_COUNTERS = {
    counter
    for counters in endpoint._OPTIONAL_COUNTER_GROUPS.values()
    for counter in counters
}


class CassiniEndpointSamplerTest(unittest.TestCase):
    identity = PDEndpointIdentity(
        endpoint_id="decoder-0",
        role=PDEndpointRole.DECODER,
        pair_index=0,
    )

    @staticmethod
    def _delta(nic: int, name: str) -> int:
        if name == "hni_rx_paused_0":
            return 20_000_000 if nic == 0 else 5_000_000
        if name == "hni_tx_paused_0":
            return 10_000_000 if nic == 0 else 2_000_000
        if name.startswith(("hni_rx_paused_", "hni_tx_paused_")):
            return 0
        if name == "parbs_tarb_pi_posted_blocked_cnt":
            return 1_000
        if name == "parbs_tarb_pi_posted_pkts":
            return 100
        if name == "parbs_tarb_pi_non_posted_blocked_cnt":
            return 400
        if name == "parbs_tarb_pi_non_posted_pkts":
            return 100
        if name.startswith("hni_pkts_"):
            return 10
        if name == "oxe_channel_idle":
            return 20_000_000 if nic == 0 else 40_000_000
        if name == "lpe_net_match_priority_0":
            return 90
        if name == "lpe_net_match_overflow_0":
            return 10
        if "_no_ecn_pkts_" in name:
            return 20
        if "_ecn_pkts_" in name:
            return 5
        if name in {"pct_no_tct_nacks", "pct_no_trs_nacks", "pct_no_mst_nacks"}:
            return 1
        if name == "pct_retry_srb_requests":
            return 2
        if name in {"pct_sct_timeouts", "pct_spt_timeouts"}:
            return 3
        raise AssertionError(f"test fixture lacks counter {name}")

    def _write(
        self,
        root: Path,
        *,
        timestamp_ns: int,
        active: bool,
        optional: set[str],
    ) -> None:
        seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
        stamp = f"{seconds}.{nanoseconds:09d}"
        for nic in range(2):
            telemetry = root / f"cxi{nic}" / "device" / "telemetry"
            telemetry.mkdir(parents=True, exist_ok=True)
            for name in CORE_COUNTERS | optional:
                value = 10_000 + (self._delta(nic, name) if active else 0)
                (telemetry / name).write_text(
                    f"{value}@{stamp}\n", encoding="ascii")

    def _sample_pair(
        self, root: Path, optional: set[str], *, max_window_ms: float = 1000,
    ) -> tuple[endpoint.CassiniEndpointSampler, dict[str, object]]:
        self._write(
            root,
            timestamp_ns=100_000_000_000,
            active=False,
            optional=optional,
        )
        sampler = endpoint.CassiniEndpointSampler(
            self.identity,
            root,
            nic_count=2,
            min_interval_ms=0,
            max_window_ms=max_window_ms,
        )
        baseline = sampler.sample(force=True)
        self.assertFalse(baseline["valid"])
        self.assertEqual(
            baseline["invalid_reason"], "counter_baseline_initialized")
        self._write(
            root,
            timestamp_ns=100_100_000_000,
            active=True,
            optional=optional,
        )
        return sampler, sampler.sample(force=True)

    def test_full_inventory_keeps_endpoint_signals_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            _sampler, result = self._sample_pair(
                Path(temporary), OPTIONAL_COUNTERS)

        self.assertTrue(result["valid"])
        self.assertEqual(result["endpoint_id"], "decoder-0")
        self.assertEqual(result["role"], "decoder")
        self.assertEqual(result["pair_index"], 0)
        self.assertEqual(
            set(result["support"].values()),
            {CounterSupport.SUPPORTED.value},
        )
        signals = result["signals"]
        self.assertAlmostEqual(signals["rx_pause_fraction_max"], 0.2)
        self.assertAlmostEqual(signals["tx_pause_fraction_max"], 0.1)
        self.assertAlmostEqual(
            signals["host_posted_cycles_per_packet_max"], 10.0)
        self.assertAlmostEqual(
            signals["host_nonposted_cycles_per_packet_max"], 4.0)
        self.assertEqual(signals["tx_packets"], 160)
        self.assertEqual(signals["rx_packets"], 160)
        self.assertAlmostEqual(signals["tx_packets_per_s"], 1600.0)
        self.assertAlmostEqual(signals["rx_packets_per_s"], 1600.0)
        self.assertAlmostEqual(signals["oxe_channel_idle_fraction_min"], 0.2)
        self.assertAlmostEqual(signals["oxe_channel_idle_fraction_mean"], 0.3)
        self.assertAlmostEqual(signals["oxe_channel_idle_fraction_max"], 0.4)
        self.assertAlmostEqual(signals["oxe_channel_active_fraction_max"], 0.8)
        self.assertAlmostEqual(signals["oxe_channel_active_fraction_mean"], 0.7)
        self.assertEqual(result["by_nic"][0]["tx_packets"], 80)
        self.assertAlmostEqual(
            result["by_nic"][0]["oxe_channel_active_fraction"], 0.8)
        self.assertAlmostEqual(
            signals["receive_overflow_fraction_max"], 0.1)
        self.assertAlmostEqual(signals["ecn_fraction_max"], 0.2)
        self.assertEqual(signals["resource_nacks"], 6)
        self.assertEqual(signals["retries"], 4)
        self.assertEqual(signals["timeouts"], 12)
        endpoint.validate_cassini_endpoint_sample(result)

    def test_absent_optional_counters_are_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            _sampler, result = self._sample_pair(Path(temporary), set())

        self.assertTrue(result["valid"])
        for group in endpoint._OPTIONAL_COUNTER_GROUPS:
            self.assertEqual(
                result["support"][group],
                CounterSupport.NOT_SUPPORTED.value,
            )
        optional_signals = set(result["signals"]) - {
            "rx_pause_fraction_max",
            "rx_pause_fraction_mean",
            "tx_pause_fraction_max",
            "tx_pause_fraction_mean",
            "host_posted_cycles_per_packet_max",
        }
        self.assertTrue(optional_signals)
        self.assertTrue(
            all(result["signals"][name] is None for name in optional_signals))

    def test_partial_optional_group_is_ambiguous_and_fail_closed(self):
        partial = {"hni_pkts_sent_by_tc_0"}
        with tempfile.TemporaryDirectory() as temporary:
            _sampler, result = self._sample_pair(Path(temporary), partial)

        self.assertTrue(result["valid"])
        self.assertEqual(
            result["support"]["packet_counts"],
            CounterSupport.AMBIGUOUS.value,
        )
        self.assertIsNone(result["signals"]["tx_packets"])
        self.assertIsNone(result["signals"]["rx_packets"])

    def test_stale_optional_group_does_not_discard_fresh_core_pressure(self):
        """One stale optional counter must not erase valid pause evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(
                root,
                timestamp_ns=100_000_000_000,
                active=False,
                optional=OPTIONAL_COUNTERS,
            )
            sampler = endpoint.CassiniEndpointSampler(
                self.identity,
                root,
                nic_count=2,
                min_interval_ms=0,
                max_window_ms=1000,
            )
            sampler.sample(force=True)
            self._write(
                root,
                timestamp_ns=100_100_000_000,
                active=True,
                optional=OPTIONAL_COUNTERS,
            )
            stale_ns = 101_500_000_000
            seconds, nanoseconds = divmod(stale_ns, 1_000_000_000)
            stamp = f"{seconds}.{nanoseconds:09d}"
            for nic in range(2):
                counter = (
                    root / f"cxi{nic}" / "device" / "telemetry"
                    / "oxe_channel_idle"
                )
                counter.write_text(
                    f"{10_000 + self._delta(nic, 'oxe_channel_idle')}@{stamp}\n",
                    encoding="ascii",
                )
            result = sampler.sample(force=True)

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["signals"]["rx_pause_fraction_max"], 0.2)
        self.assertEqual(
            result["support"]["oxe_channel_idle_fraction"],
            CounterSupport.AMBIGUOUS.value,
        )
        self.assertIsNone(
            result["signals"]["oxe_channel_active_fraction_max"])
        self.assertIsNone(
            result["by_nic"][0]["oxe_channel_active_fraction"])

    def test_counter_regression_and_stale_window_are_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sampler, result = self._sample_pair(root, set())
            self.assertTrue(result["valid"])
            self._write(
                root,
                timestamp_ns=100_200_000_000,
                active=False,
                optional=set(),
            )
            regressed = sampler.sample(force=True)
            self.assertFalse(regressed["valid"])
            self.assertEqual(
                regressed["invalid_reason"],
                "counter_regressed_or_timestamp_not_monotonic",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(
                root,
                timestamp_ns=200_000_000_000,
                active=False,
                optional=set(),
            )
            sampler = endpoint.CassiniEndpointSampler(
                self.identity,
                root,
                nic_count=2,
                min_interval_ms=0,
                max_window_ms=100,
            )
            sampler.sample(force=True)
            self._write(
                root,
                timestamp_ns=200_200_000_000,
                active=True,
                optional=set(),
            )
            stale = sampler.sample(force=True)
            self.assertFalse(stale["valid"])
            self.assertEqual(stale["invalid_reason"], "counter_window_stale")

    def test_validator_rejects_forged_identity_and_optional_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            _sampler, result = self._sample_pair(Path(temporary), set())

        forged_role = dict(result)
        forged_role["role"] = "router"
        with self.assertRaisesRegex(ValueError, "role"):
            endpoint.validate_cassini_endpoint_sample(forged_role)

        forged_signal = dict(result)
        forged_signal["signals"] = dict(result["signals"])
        forged_signal["signals"]["ecn_fraction_max"] = 0.0
        with self.assertRaisesRegex(ValueError, "unavailable Cassini group"):
            endpoint.validate_cassini_endpoint_sample(forged_signal)


if __name__ == "__main__":
    unittest.main()
