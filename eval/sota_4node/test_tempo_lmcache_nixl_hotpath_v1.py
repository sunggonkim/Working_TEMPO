from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lmcache.v1.transfer_channel import tempo_nixl_hotpath as hotpath


class _Meta:
    def __init__(self, address: int):
        self.address = address


class _Object:
    def __init__(self, address: int):
        self.meta = _Meta(address)


class _Agent:
    def __init__(self):
        self.made = []
        self.transferred = []
        self.released = []
        self.statuses: list[list[str]] = []

    def make_prepped_xfer(self, *args):
        handle = object()
        self.made.append((handle, args))
        return handle

    def transfer(self, handle):
        self.transferred.append(handle)

    def check_xfer_state(self, handle):
        if not self.statuses:
            return "DONE"
        values = self.statuses[0]
        value = values.pop(0)
        if not values:
            self.statuses.pop(0)
        return value

    def release_xfer_handle(self, handle):
        self.released.append(handle)


class _Wrapper:
    xfer_handler = object()


class _Channel:
    def __init__(self):
        self.nixl_agent = _Agent()
        self.nixl_wrapper = _Wrapper()
        self.remote_xfer_handlers_dict = {"peer": object()}

    def get_local_mem_indices(self, objects):
        return [item.meta.address for item in objects]


class HotPathTests(unittest.TestCase):
    def _run(self, channel, addresses=(10, 20), remote=(1, 2)):
        return asyncio.run(hotpath.async_batched_write(
            channel,
            [_Object(value) for value in addresses],
            {"receiver_id": "peer", "remote_indexes": list(remote)},
        ))

    def test_exact_signature_reuses_done_handle(self):
        channel = _Channel()
        with patch.dict(os.environ, {
            "TEMPO_NIXL_CACHE_CAPACITY": "4",
            "TEMPO_NIXL_YIELD_POLLS": "0",
            "TEMPO_NIXL_SLEEP_US": "1",
        }):
            self.assertEqual(self._run(channel), 2)
            self.assertEqual(self._run(channel), 2)
        self.assertEqual(len(channel.nixl_agent.made), 1)
        self.assertIs(channel.nixl_agent.transferred[0], channel.nixl_agent.transferred[1])
        state = channel._tempo_nixl_hotpath_state
        self.assertEqual(state.stats.reuse_count, 1)
        self.assertEqual(state.inflight, {})

    def test_different_index_or_address_misses(self):
        channel = _Channel()
        with patch.dict(os.environ, {"TEMPO_NIXL_CACHE_CAPACITY": "8"}):
            self._run(channel)
            self._run(channel, remote=(1, 3))
            self._run(channel, addresses=(10, 30))
        self.assertEqual(len(channel.nixl_agent.made), 3)

    def test_error_releases_and_does_not_cache(self):
        channel = _Channel()
        channel.nixl_agent.statuses = [["ERR"]]
        with self.assertRaisesRegex(RuntimeError, "Failed to send"):
            self._run(channel)
        state = channel._tempo_nixl_hotpath_state
        self.assertEqual(state.idle_count, 0)
        self.assertEqual(len(channel.nixl_agent.released), 1)

    def test_adaptive_poll_counts_are_recorded(self):
        channel = _Channel()
        channel.nixl_agent.statuses = [["PROC", "PROC", "PROC", "DONE"]]
        with patch.dict(os.environ, {
            "TEMPO_NIXL_YIELD_POLLS": "2",
            "TEMPO_NIXL_SLEEP_US": "1",
        }):
            self._run(channel)
        stats = channel._tempo_nixl_hotpath_state.stats
        self.assertEqual(stats.yield_poll_count, 2)
        self.assertEqual(stats.sleep_poll_count, 1)

    def test_stats_write_is_exclusive_and_complete(self):
        channel = _Channel()
        self._run(channel)
        state = channel._tempo_nixl_hotpath_state
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"TEMPO_LMCACHE_NIXL_STATS_DIR": directory}):
                hotpath._write_stats(state)
                paths = list(Path(directory).glob("nixl-hotpath-*.json"))
                self.assertEqual(len(paths), 1)
                payload = json.loads(paths[0].read_text(encoding="utf-8"))
                self.assertEqual(payload["schema"], hotpath.SCHEMA)
                with self.assertRaises(FileExistsError):
                    hotpath._write_stats(state)


if __name__ == "__main__":
    unittest.main()
