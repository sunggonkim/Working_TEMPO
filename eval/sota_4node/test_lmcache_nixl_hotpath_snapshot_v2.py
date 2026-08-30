from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.sota_4node.test_tempo_lmcache_nixl_hotpath_v1 import _Channel
from lmcache.v1.transfer_channel import tempo_nixl_hotpath as v1
from lmcache.v1.transfer_channel import tempo_nixl_hotpath_v2 as v2


class SnapshotTests(unittest.TestCase):
    def test_snapshot_survives_without_close(self):
        channel = _Channel()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "TEMPO_LMCACHE_NIXL_STATS_DIR": directory,
            "TEMPO_NIXL_YIELD_POLLS": "0",
        }):
            asyncio.run(v1.async_batched_write(
                channel, [type("Obj", (), {"meta": type("Meta", (), {"address": 9})()})()],
                {"receiver_id": "peer", "remote_indexes": [3]},
            ))
            v2._snapshot(channel._tempo_nixl_hotpath_state)
            paths = list(Path(directory).glob("nixl-hotpath-*.json"))
            self.assertEqual(len(paths), 1)
            payload = json.loads(paths[0].read_text())
            self.assertEqual(payload["stats"]["transfer_count"], 1)
            self.assertEqual(payload["inflight_handles_at_snapshot"], 0)
            self.assertEqual(payload["idle_handles_at_snapshot"], 1)

    def test_launcher_is_one_bounded_step(self):
        root = Path(__file__).resolve().parent
        source = (root / "run_lmcache_nixl_hotpath_snapshot_v2_in_allocation.sh").read_text()
        self.assertEqual(source.count("srun "), 1)
        self.assertNotIn("salloc", source)
        self.assertNotIn("sbatch", source)


if __name__ == "__main__":
    unittest.main()
