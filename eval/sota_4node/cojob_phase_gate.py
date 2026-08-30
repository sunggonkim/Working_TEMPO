"""Bounded shared-file phase gate for native co-job experiments."""

from __future__ import annotations

import math
from pathlib import Path
import time


def wait_for_start_file(
    start_file: Path,
    *,
    timeout_s: float,
    stop_file: Path | None = None,
    poll_interval_s: float = 1.0,
) -> None:
    """Wait for a bounded, shared experiment-phase marker."""

    if not start_file.is_absolute():
        raise ValueError("start-file must be absolute")
    if not math.isfinite(timeout_s) or not 0.05 <= timeout_s <= 3600.0:
        raise ValueError("start-file timeout must be in [0.05, 3600] seconds")
    if (
        not math.isfinite(poll_interval_s)
        or not 0.05 <= poll_interval_s <= 10.0
    ):
        raise ValueError("start-file poll interval must be in [0.05, 10] seconds")

    deadline = time.monotonic() + timeout_s
    while True:
        if start_file.is_file():
            return
        if stop_file is not None and stop_file.is_file():
            raise RuntimeError("co-job stop requested before start-file appeared")
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                f"co-job start-file did not appear within {timeout_s:.3f}s: "
                f"{start_file}"
            )
        time.sleep(min(poll_interval_s, remaining))


__all__ = ["wait_for_start_file"]
