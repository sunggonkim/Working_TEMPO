from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.build_g1_causal_readiness import (
    COUNTER_SCHEMA,
    _validate_counter_file,
    build_g1_causal_readiness,
)
from tempo.resource_domain import ResourceDomain, domain_contract


RAW_G1 = Path(__file__).resolve().parents[2] / "results" / "sota_4node" / "g1_tier_job_56803347"


class G1CausalReadinessTests(unittest.TestCase):
    def test_slingshot_is_explicitly_deferred_to_g2(self) -> None:
        from eval.sota_4node.build_g1_causal_readiness import _required_pairs
        self.assertNotIn(
            ResourceDomain.SLINGSHOT_FABRIC.value,
            {domain.value for _, domain in _required_pairs()},
        )

    def test_collection_plan_keeps_gpu_diagnostic_out_of_timed_metrics(self) -> None:
        from eval.sota_4node.build_g1_causal_readiness import G1_COUNTER_COLLECTION_PLAN
        gpu = G1_COUNTER_COLLECTION_PLAN[ResourceDomain.GPU_LOCAL.value]
        self.assertEqual(gpu["stage"], "g1")
        self.assertFalse(gpu["timed_metrics_eligible"])
        self.assertEqual(
            G1_COUNTER_COLLECTION_PLAN[ResourceDomain.SLINGSHOT_FABRIC.value]["stage"], "g2"
        )
    def test_historical_raw_artifact_is_not_ready_without_domain_counters(self) -> None:
        if not RAW_G1.is_dir():
            self.skipTest("historical raw G1 artifact is not present")
        result = build_g1_causal_readiness(RAW_G1)
        self.assertEqual(result["status"], "not_ready")
        self.assertFalse(result["promotion_ready"])
        self.assertEqual(result["logical_stage"]["present"], 0)
        self.assertGreater(result["logical_stage"]["expected"], 0)
        self.assertTrue(result["missing_mode_domain_pairs"])
        self.assertIn("gpu_local", result["missing_domains"])
        self.assertEqual(result["foreground_path"]["status"], "missing")
        self.assertIn("foreground_path.json is missing", result["reasons"])
        self.assertGreater(result["raw_mode_summary"]["open_combined"]["rank_max_step_p99_ms"], 0.0)
        self.assertIsNotNone(result["raw_mode_summary"]["open_combined"]["rank_max_durable_ms"])

    def test_foreground_path_sidecar_is_required_and_byte_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "foreground_path.json"
            path.write_text("{}\n", encoding="utf-8")
            # The raw artifact is intentionally incomplete; this assertion
            # only exercises the sidecar branch before schema parsing.
            result = __import__(
                "eval.sota_4node.build_g1_causal_readiness",
                fromlist=["_foreground_path_status"],
            )._foreground_path_status(root)
            self.assertEqual(result["status"], "invalid")
            self.assertIn("sha256 is missing", result["reasons"][0])
            path.with_suffix(".json.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
            result = __import__(
                "eval.sota_4node.build_g1_causal_readiness",
                fromlist=["_foreground_path_status"],
            )._foreground_path_status(root)
            self.assertIn("does not match", result["reasons"][0])
    def test_counter_record_requires_observed_hardware_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.json"
            contract = domain_contract(ResourceDomain.GPU_LOCAL)
            record = {
                "schema": COUNTER_SCHEMA,
                "mode": "d2h_only",
                "domain": ResourceDomain.GPU_LOCAL.value,
                "scope": "rank",
                "scope_id": "0",
                "intervention_id": "d2h_only",
                "path_evidence": contract.path_evidence,
                "counter_family": contract.counter_family,
                "path_status": "observed",
                "counter_support": "supported",
                "source": "nvml-copy-engine",
                "hardware_counter": True,
                "samples": [
                    {"sample_id": "start", "source": "nvml-copy-engine", "timestamp_ns": 1,
                     "cumulative_bytes": 0, "cumulative_busy_ns": 0, "support": "supported"},
                    {"sample_id": "end", "source": "nvml-copy-engine", "timestamp_ns": 2,
                     "cumulative_bytes": 4096, "cumulative_busy_ns": 1, "support": "supported"},
                ],
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            parsed = _validate_counter_file(path, "d2h_only", ResourceDomain.GPU_LOCAL)
            self.assertEqual(parsed["samples"], 2)
            record["path_status"] = "declared"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_counter_file(path, "d2h_only", ResourceDomain.GPU_LOCAL)

    def test_counter_record_rejects_regressing_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pfs.json"
            contract = domain_contract(ResourceDomain.PERSISTENT_ENDPOINT)
            record = {
                "schema": COUNTER_SCHEMA,
                "mode": "persist_only",
                "domain": ResourceDomain.PERSISTENT_ENDPOINT.value,
                "scope": "endpoint",
                "scope_id": "ost0",
                "intervention_id": "persist_only",
                "path_evidence": contract.path_evidence,
                "counter_family": contract.counter_family,
                "path_status": "observed",
                "counter_support": "supported",
                "source": "lustre-ost",
                "hardware_counter": True,
                "samples": [
                    {"sample_id": "start", "source": "lustre-ost", "timestamp_ns": 1,
                     "cumulative_bytes": 4096, "cumulative_busy_ns": 2, "support": "supported"},
                    {"sample_id": "end", "source": "lustre-ost", "timestamp_ns": 2,
                     "cumulative_bytes": 2048, "cumulative_busy_ns": 3, "support": "supported"},
                ],
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_counter_file(path, "persist_only", ResourceDomain.PERSISTENT_ENDPOINT)

    def test_counter_record_identifiers_are_strict_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pfs.json"
            contract = domain_contract(ResourceDomain.PERSISTENT_ENDPOINT)
            record = {
                "schema": COUNTER_SCHEMA,
                "mode": "persist_only",
                "domain": ResourceDomain.PERSISTENT_ENDPOINT.value,
                "scope": "endpoint",
                "scope_id": "ost0",
                "intervention_id": "persist_only",
                "path_evidence": contract.path_evidence,
                "counter_family": contract.counter_family,
                "path_status": "observed",
                "counter_support": "supported",
                "source": "lustre-ost",
                "hardware_counter": True,
                "samples": [
                    {"sample_id": "start", "source": "lustre-ost", "timestamp_ns": 1,
                     "cumulative_bytes": 0, "cumulative_busy_ns": 0, "support": "supported"},
                    {"sample_id": "end", "source": "lustre-ost", "timestamp_ns": 2,
                     "cumulative_bytes": 4096, "cumulative_busy_ns": 1, "support": "supported"},
                ],
            }
            for field in ("sample_id", "source"):
                candidate = json.loads(json.dumps(record))
                candidate["samples"][0][field] = 1
                path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "sample_id/source"):
                        _validate_counter_file(path, "persist_only", ResourceDomain.PERSISTENT_ENDPOINT)

    def test_host_wide_counter_cannot_promote_as_rank_bound_nic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nic.json"
            contract = domain_contract(ResourceDomain.NIC_FABRIC)
            record = {
                "schema": COUNTER_SCHEMA,
                "mode": "open_combined",
                "domain": ResourceDomain.NIC_FABRIC.value,
                "scope": "host",
                "scope_id": "nid00001",
                "intervention_id": "open_combined",
                "path_evidence": contract.path_evidence,
                "counter_family": contract.counter_family,
                "path_status": "observed",
                "counter_support": "supported",
                "source": "sysfs-hsn-host-total",
                "hardware_counter": True,
                "samples": [
                    {"sample_id": "start", "source": "sysfs-hsn-host-total", "timestamp_ns": 1,
                     "cumulative_bytes": 0, "cumulative_busy_ns": 0, "support": "supported"},
                    {"sample_id": "end", "source": "sysfs-hsn-host-total", "timestamp_ns": 2,
                     "cumulative_bytes": 4096, "cumulative_busy_ns": 0, "support": "supported"},
                ],
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scope"):
                _validate_counter_file(path, "open_combined", ResourceDomain.NIC_FABRIC)


if __name__ == "__main__":
    unittest.main()
