from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
import unittest

from eval.sota_4node.cojob_phase_gate import (
    wait_for_start_file,
)


class StartFileGateTest(unittest.TestCase):
    def test_existing_marker_releases_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw).resolve() / "measured-control-complete"
            marker.write_text("ready\n", encoding="utf-8")
            wait_for_start_file(marker, timeout_s=1.0, poll_interval_s=0.05)

    def test_late_marker_releases_bounded_wait(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw).resolve() / "measured-control-complete"

            def publish() -> None:
                time.sleep(0.05)
                marker.write_text("ready\n", encoding="utf-8")

            worker = threading.Thread(target=publish)
            worker.start()
            wait_for_start_file(marker, timeout_s=1.0, poll_interval_s=0.05)
            worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())

    def test_stop_before_marker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            stop = root / "stop"
            stop.write_text("stop\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "before start-file"):
                wait_for_start_file(
                    root / "missing",
                    timeout_s=1.0,
                    stop_file=stop,
                    poll_interval_s=0.05,
                )

    def test_missing_marker_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(TimeoutError, "did not appear"):
                wait_for_start_file(
                    Path(raw).resolve() / "missing",
                    timeout_s=0.1,
                    poll_interval_s=0.05,
                )

    def test_relative_marker_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            wait_for_start_file(
                Path("relative-marker"),
                timeout_s=1.0,
                poll_interval_s=0.05,
            )


if __name__ == "__main__":
    unittest.main()
