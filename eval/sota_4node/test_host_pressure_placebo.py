from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.host_pressure_placebo import PAGE_SIZE, _touch_pages, process_numa_bytes


class HostPressurePlaceboTests(unittest.TestCase):
    def test_process_numa_bytes_parses_only_the_requested_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numa_maps"
            path.write_text(
                "7f000000 rw-p 00000000 00:00 0 anon=4 N0=2 N3=6\n"
                "7f100000 rw-p 00000000 00:00 0 anon=2 N3=1\n",
                encoding="utf-8",
            )
            self.assertEqual(process_numa_bytes(3, proc_maps=path), 7 * PAGE_SIZE)
            self.assertEqual(process_numa_bytes(0, proc_maps=path), 2 * PAGE_SIZE)

    def test_process_numa_bytes_rejects_missing_node_or_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numa_maps"
            path.write_text("anon=1 N0=1\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "absent"):
                process_numa_bytes(2, proc_maps=path)
            with self.assertRaisesRegex(RuntimeError, "cannot read"):
                process_numa_bytes(0, proc_maps=path.with_name("missing"))

    def test_touch_pages_reports_declared_bytes_and_positive_busy(self) -> None:
        touched, busy = _touch_pages(bytearray(PAGE_SIZE + 17))
        self.assertEqual(touched, PAGE_SIZE + 17)
        self.assertGreaterEqual(busy, 0)


if __name__ == "__main__":
    unittest.main()
