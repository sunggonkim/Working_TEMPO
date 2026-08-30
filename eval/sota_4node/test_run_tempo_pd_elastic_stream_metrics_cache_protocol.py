from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import (
    run_tempo_pd_elastic_stream_metrics_cache_protocol as protocol,
)


class CacheProtocolStreamClientTest(unittest.TestCase):
    def test_measurement_start_marker_is_atomic_exact_and_one_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "start.json"
            observer = protocol._measurement_start_observer(marker)
            observer(123456789)
            value = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(value["schema"], protocol.START_MARKER_SCHEMA)
            self.assertEqual(value["clock"], "client time.perf_counter_ns")
            self.assertEqual(value["run_start_ns"], 123456789)
            self.assertGreater(value["publisher_pid"], 0)
            self.assertEqual(
                list(Path(directory).glob(".*.tmp-*")), [])
            with self.assertRaisesRegex(ValueError, "published twice"):
                observer(123456790)
            with self.assertRaisesRegex(ValueError, "already exists"):
                protocol._measurement_start_observer(marker)

    def test_decoder_prepare_is_exact_serial_seed_probe_plan(self):
        rows = [
            {
                "request_id": (
                    "epd-tempo-c4prep-a-warm-seed-o256-cache-d-seed-"
                    "item-000000"),
                "prompt": "same",
                "max_tokens": 2,
            },
            {
                "request_id": (
                    "epd-tempo-c4prep-a-warm-cache-d-probe-item-000000"),
                "prompt": "same",
                "max_tokens": 2,
            },
        ]
        plan = {
            "decoder_prepare_request_ids": [
                row["request_id"] for row in rows],
        }
        protocol.validate_invocation(
            phase="decoder_prepare", rows=rows, plan=plan, evidence=None)
        with self.assertRaisesRegex(ValueError, "differs from frozen plan"):
            protocol.validate_invocation(
                phase="decoder_prepare", rows=list(reversed(rows)),
                plan=plan, evidence=None)

    def test_measured_requires_ready_bound_evidence_and_exact_markers(self):
        request_id = (
            "epd-tempo-cache-p-only-measured-r0-item-000000")
        rows = [{"request_id": request_id, "prompt": "x", "max_tokens": 2}]
        plan = {
            "fingerprint_sha256": "a" * 64,
            "items": [{"request_id": request_id}],
        }
        evidence = {
            "schema": protocol.RUNTIME_EVIDENCE_SCHEMA,
            "ready_for_measurement": True,
            "plan_fingerprint_sha256": "a" * 64,
        }
        protocol.validate_invocation(
            phase="measured", rows=rows, plan=plan, evidence=evidence)
        evidence["ready_for_measurement"] = False
        with self.assertRaisesRegex(ValueError, "not measurement-ready"):
            protocol.validate_invocation(
                phase="measured", rows=rows, plan=plan, evidence=evidence)


if __name__ == "__main__":
    unittest.main()
