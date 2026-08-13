from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.capture_lustre_rpc_observation import parse_rpc_stats, snapshot, write_snapshot


RPC_STATS = """snapshot_time: 1.000000000 secs.nsecs

\t\t\tread\t\t\twrite
pages per rpc         rpcs   % cum % |       rpcs   % cum %
1:                     2  50  50   |         3  60  60
16:                    1  25  75   |         2  40 100

rpcs in flight        rpcs   % cum % |       rpcs   % cum %
"""


class LustreRPCObservationTests(unittest.TestCase):
    def test_parse_preserves_page_histogram_without_calling_it_bytes(self):
        value = parse_rpc_stats(RPC_STATS)
        self.assertEqual(value["read_pages"], 18)
        self.assertEqual(value["write_pages"], 35)
        self.assertEqual(len(value["rows"]), 2)

    def test_snapshot_is_endpoint_scoped_and_not_causal(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "osc"
            endpoint = proc / "scratch-OST0001-osc-test"
            endpoint.mkdir(parents=True)
            (endpoint / "rpc_stats").write_text(RPC_STATS)
            value = snapshot(mode="persist_only", phase="start", proc_root=proc)
        self.assertEqual(value["scope"], "endpoint")
        self.assertFalse(value["hardware_counter"])
        self.assertFalse(value["causal_ready"])
        self.assertEqual(value["records"][0]["write_pages"], 35)

    def test_write_snapshot_has_exact_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "osc"
            endpoint = proc / "scratch-OST0001-osc-test"
            endpoint.mkdir(parents=True)
            (endpoint / "rpc_stats").write_text(RPC_STATS)
            path = write_snapshot(Path(tmp) / "out", mode="combined", phase="end", proc_root=proc)
            value = json.loads(path.read_text())
        self.assertEqual(value["schema"], "tempo-rd-lustre-rpc-page-observation-1")
        self.assertEqual(value["phase"], "end")

    def test_missing_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pages-per-rpc"):
            parse_rpc_stats("snapshot_time: 1\n")


if __name__ == "__main__":
    unittest.main()
