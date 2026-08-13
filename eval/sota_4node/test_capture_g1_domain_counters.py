from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.sota_4node.capture_g1_domain_counters import (
    CAPABILITY_SCHEMA,
    SCHEMA,
    capture_mode,
    parse_nsys_gpu_mem_report,
    write_capabilities,
)


class CaptureG1DomainCounterTests(unittest.TestCase):
    def test_mode_capture_keeps_only_monotonic_nic_samples(self) -> None:
        values = iter(
            (
                (100, {"hsn0.rx_bytes": 10, "hsn0.tx_bytes": 20}, "fake-hsn"),
                (200, {"hsn0.rx_bytes": 110, "hsn0.tx_bytes": 220}, "fake-hsn"),
            )
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "eval.sota_4node.capture_g1_domain_counters._hsn_snapshot",
            side_effect=lambda: next(values),
        ):
            path = Path(tmp) / "nic.start.json"
            capture_mode("open_combined", path, "start")
            capture_mode("open_combined", path, "end")
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["schema"], SCHEMA)
        self.assertEqual(record["domain"], "nic_fabric")
        self.assertEqual(record["scope"], "host")
        self.assertTrue(record["scope_id"])
        self.assertEqual(record["intervention_id"], "open_combined")
        self.assertTrue(record["hardware_counter"])
        self.assertEqual([item["cumulative_bytes"] for item in record["samples"]], [30, 330])
        self.assertEqual(set(record["samples"][0]), {
            "sample_id", "source", "timestamp_ns", "cumulative_bytes",
            "cumulative_busy_ns", "support",
        })

    def test_capability_record_marks_unavailable_domains_without_invention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "eval.sota_4node.capture_g1_domain_counters._hsn_snapshot",
            return_value=(123, {"hsn0.rx_bytes": 1, "hsn0.tx_bytes": 2}, "fake-hsn"),
        ), patch(
            "eval.sota_4node.capture_g1_domain_counters._command",
            return_value={"argv": [], "returncode": 0, "stdout": "", "stderr": ""},
        ):
            path = Path(tmp) / "capabilities.json"
            write_capabilities(path)
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["schema"], CAPABILITY_SCHEMA)
        self.assertEqual(record["domains"]["nic_fabric"]["counter_support"], "supported")
        self.assertEqual(record["domains"]["nic_fabric"]["causal_scope_support"], "not_supported")
        self.assertTrue(record["domains"]["nic_fabric"]["diagnostic_only"])
        self.assertEqual(record["domains"]["nic_fabric"]["scope"], "host")
        self.assertEqual(record["domains"]["slingshot_fabric"]["counter_support"], "not_supported")
        self.assertEqual(record["domains"]["gpu_local"]["counter_support"], "not_collected")
        self.assertIn("counter_probe", record["context"])
        self.assertEqual(
            record["context"]["counter_probe"]["interpretation"],
            "capability_probe_only; no candidate is causal evidence",
        )

    def test_nsys_parser_requires_exact_dtoh_and_emits_gpu_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nsys.json"
            path.write_text(json.dumps({"rows": [
                {"Operation": "Memcpy DtoH", "Total": "1.5 MiB"},
                {"Operation": "Device-to-Host", "Bytes": 512},
                {"Operation": "Memcpy HtoD", "Total": "99 MiB"},
            ]}), encoding="utf-8")
            record = parse_nsys_gpu_mem_report(path, mode="d2h_only", timestamp_ns=100)
        self.assertEqual(record["domain"], "gpu_local")
        self.assertEqual(record["path_evidence"], "gpu_hbm_copy_engine")
        self.assertEqual(record["counter_family"], "gpu_copy_engine_bytes")
        self.assertTrue(record["diagnostic_only"])
        self.assertEqual([s["cumulative_bytes"] for s in record["samples"]], [0, 1_573_376])
        self.assertEqual(record["samples"][1]["timestamp_ns"], 100)

    def test_nsys_25_mb_column_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nsys_25.json"
            path.write_text(json.dumps([{
                "Operation": "[CUDA memcpy Device-to-Host]",
                "Total (MB)": 268.435,
                "Count": 2,
            }]), encoding="utf-8")
            record = parse_nsys_gpu_mem_report(path, mode="gpu_diag", timestamp_ns=100)
        self.assertEqual(record["samples"][1]["cumulative_bytes"], 268_435_000)
        self.assertTrue(record["diagnostic_only"])

    def test_nsys_rank_scope_is_bound_from_diagnostic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nsys_rank.json"
            path.write_text(json.dumps([{
                "Operation": "[CUDA memcpy Device-to-Host]",
                "Total (MB)": 1.0,
            }]), encoding="utf-8")
            record = parse_nsys_gpu_mem_report(path, mode="gpu_diag_rank_3", timestamp_ns=100)
        self.assertEqual(record["scope"], "rank")
        self.assertEqual(record["scope_id"], "rank 3")

    def test_nsys_time_report_binds_positive_busy_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            size = Path(tmp) / "size.json"
            duration = Path(tmp) / "time.json"
            size.write_text(json.dumps([{
                "Operation": "[CUDA memcpy Device-to-Host]",
                "Total (MB)": 2.0,
            }]), encoding="utf-8")
            duration.write_text(json.dumps([{
                "Operation": "[CUDA memcpy Device-to-Host]",
                "Total Time (ns)": 12345,
            }]), encoding="utf-8")
            record = parse_nsys_gpu_mem_report(
                size, mode="gpu_diag_rank_0", timestamp_ns=100,
                busy_report=duration,
            )
        self.assertEqual(record["samples"][1]["cumulative_busy_ns"], 12345)
        self.assertNotIn("diagnostic_only", record)

    def test_nsys_parser_rejects_missing_or_ambiguous_dtoh_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            missing.write_text(json.dumps({"rows": [{"Operation": "Memcpy DtoH"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_nsys_gpu_mem_report(missing, mode="d2h_only", timestamp_ns=1)

            ambiguous = Path(tmp) / "ambiguous.json"
            ambiguous.write_text(json.dumps({"rows": [
                {"Operation": "Memcpy DtoH", "Total": 10, "Bytes": 11},
            ]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_nsys_gpu_mem_report(ambiguous, mode="d2h_only", timestamp_ns=1)

            no_dtoh = Path(tmp) / "no_dtoh.json"
            no_dtoh.write_text(json.dumps({"rows": [{"Operation": "Memcpy HtoD", "Total": 10}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_nsys_gpu_mem_report(no_dtoh, mode="d2h_only", timestamp_ns=1)


if __name__ == "__main__":
    unittest.main()
