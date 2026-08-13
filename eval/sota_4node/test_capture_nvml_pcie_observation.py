from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.sota_4node.capture_nvml_pcie_observation import (
    NVML_WINDOW_NS,
    integrate_rate_samples,
    snapshot,
    write_snapshot,
    write_stream,
)


class _PCI:
    busId = "00000000:03:00.0"


class _NVML:
    NVML_PCIE_UTIL_TX_BYTES = 0
    NVML_PCIE_UTIL_RX_BYTES = 1

    def nvmlInit(self): pass
    def nvmlShutdown(self): pass
    def nvmlDeviceGetCount(self): return 4
    def nvmlDeviceGetHandleByIndex(self, index): return index
    def nvmlDeviceGetPcieThroughput(self, handle, kind): return 100 + handle + kind
    def nvmlDeviceGetPciInfo(self, handle): return _PCI()


class NVMLPCIeObservationTests(unittest.TestCase):
    def test_rate_schema_is_not_cumulative_counter(self):
        with mock.patch("eval.sota_4node.capture_nvml_pcie_observation.pynvml", _NVML()), \
             mock.patch.dict(os.environ, {"SLURM_PROCID": "2"}, clear=False):
            value = snapshot(mode="d2h_only", phase="start")
        self.assertEqual(value["scope_id"], "rank 2")
        self.assertEqual(value["unit"], "KB/s")
        self.assertFalse(value["causal_ready"])
        self.assertEqual(value["counter_semantics"], "instantaneous_rate_not_cumulative_bytes")

    def test_write_is_rank_bound_and_deterministic_keys(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("eval.sota_4node.capture_nvml_pcie_observation.pynvml", _NVML()), \
             mock.patch.dict(os.environ, {"SLURM_PROCID": "0"}, clear=False):
            path = write_snapshot(Path(tmp), mode="fg_only", phase="end")
            value = json.loads(path.read_text())
        self.assertEqual(value["scope"], "rank")
        self.assertEqual(value["pci_bus_id"], "00000000:03:00.0")

    def test_integrated_trace_is_explicitly_noncausal_and_bounded(self):
        start = 100_000_000
        samples = [
            {"timestamp_ns": start + 1_000_000, "tx_kb_per_s": 1000, "rx_kb_per_s": 0},
            {"timestamp_ns": start + 6_000_000, "tx_kb_per_s": 2000, "rx_kb_per_s": 100},
            {"timestamp_ns": start + 11_000_000, "tx_kb_per_s": 1000, "rx_kb_per_s": 0},
        ]
        result = integrate_rate_samples(
            samples,
            interval_start_ns=start,
            interval_end_ns=start + 12_000_000,
        )
        self.assertEqual(result["tx_estimated_bytes"], 15_000)
        self.assertEqual(result["rx_estimated_bytes"], 500)
        self.assertGreater(result["uncertainty_bytes"], 0)
        self.assertFalse(result["causal_ready"])

    def test_integrated_trace_rejects_uncovered_gap(self):
        with self.assertRaisesRegex(ValueError, "gap"):
            integrate_rate_samples(
                [
                    {"timestamp_ns": 100_000_000, "tx_kb_per_s": 1, "rx_kb_per_s": 1},
                    {"timestamp_ns": 100_000_000 + NVML_WINDOW_NS + 1, "tx_kb_per_s": 1, "rx_kb_per_s": 1},
                ],
                interval_start_ns=100_000_000,
                interval_end_ns=100_000_000 + NVML_WINDOW_NS + 1,
            )

    def test_stream_writer_emits_multiple_rate_samples(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("eval.sota_4node.capture_nvml_pcie_observation.pynvml", _NVML()), \
             mock.patch.dict(os.environ, {"SLURM_PROCID": "0"}, clear=False):
            path = write_stream(Path(tmp), mode="d2h_only", duration_ms=2, sample_interval_ms=1)
            lines = path.read_text().splitlines()
        self.assertGreaterEqual(len(lines), 1)
        records = [json.loads(line) for line in lines]
        self.assertTrue(all(record["phase"] == "stream" for record in records))
        self.assertTrue(all(record["counter_semantics"] == "instantaneous_rate_not_cumulative_bytes" for record in records))


if __name__ == "__main__":
    unittest.main()
