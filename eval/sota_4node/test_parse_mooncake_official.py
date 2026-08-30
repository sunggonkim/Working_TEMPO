import json
import tempfile
import unittest
from pathlib import Path

import parse_mooncake_official as parser


COMMIT = "47834aea34a61823b02b6d86c2f385100ad51eaa"


class ParseMooncakeOfficialTest(unittest.TestCase):
    def test_parses_official_glog_completion_line(self):
        text = (
            "I0813 21:00:00.000000 1 transfer_engine_bench.cpp:564] "
            "Test completed: duration 5.25, batch count 42, "
            "throughput 12.500000 GB/s\n"
        )
        measurement = parser.parse_benchmark_output(text)
        self.assertEqual(measurement["duration_seconds"], 5.25)
        self.assertEqual(measurement["batch_count"], 42)
        self.assertEqual(measurement["throughput"]["unit"], "GB/s")
        self.assertEqual(
            measurement["throughput"]["bytes_per_second"], 12_500_000_000.0
        )

    def test_rejects_log_without_completed_transfer(self):
        with self.assertRaises(parser.MooncakeParseError):
            parser.parse_benchmark_output("target waiting for initiator\n")

    def test_result_is_explicitly_tcp_and_throughput_only(self):
        manifest = parser.make_manifest(
            wheel_version="0.3.12.post1",
            git_commit=COMMIT,
            git_repository=parser.OFFICIAL_REPOSITORY,
            binary=".sota_venv/bin/transfer_engine_bench",
            job_id="12345",
            node_list="nid[001-002]",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initiator = root / "rank-1.stderr.log"
            target = root / "rank-0.stderr.log"
            initiator.write_text(
                "Test completed: duration 5.01, batch count 9, "
                "throughput 1.250000 GiB/s\n"
            )
            target.write_text("target stopped\n")
            result = parser.build_result(manifest, [initiator], [target])

        self.assertEqual(result["transport"], "tcp")
        self.assertEqual(result["scope"], "official-transfer-engine-only")
        self.assertEqual(result["measurement_scope"], "aggregate-throughput-only")
        self.assertNotIn("latency", json.dumps(result).lower())
        self.assertEqual(
            result["manifest"]["artifact"]["source_git_commit"], COMMIT
        )

    def test_manifest_rejects_overstated_transport(self):
        manifest = parser.make_manifest(
            wheel_version="0.3.12.post1",
            git_commit=COMMIT,
            git_repository=parser.OFFICIAL_REPOSITORY,
            binary=".sota_venv/bin/transfer_engine_bench",
            job_id="12345",
            node_list="nid[001-002]",
        )
        manifest["transport"] = "cxi"
        with self.assertRaises(parser.MooncakeParseError):
            parser.validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
