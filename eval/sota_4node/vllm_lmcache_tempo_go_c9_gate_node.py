#!/usr/bin/env python3
"""C9-only C8 node wrapper with a shared post-health measurement gate."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from eval.sota_4node import vllm_lmcache_tempo_go_c8_dual_regime_node as c8


def _result_dir() -> Path:
    argv = sys.argv[1:]
    try:
        return Path(argv[argv.index("--result-dir") + 1]).resolve()
    except (ValueError, IndexError):
        raise RuntimeError("C9 result directory argument is missing")


def main() -> int:
    result_dir = _result_dir()
    start_file_raw = os.environ.get("TEMPO_GO_C9_INFERENCE_START_FILE", "")
    if not start_file_raw:
        raise RuntimeError("C9 inference start file is missing")
    start_file = Path(start_file_raw).resolve()
    if start_file.parent != result_dir:
        raise RuntimeError("C9 inference start file must be in result dir")
    node_index = int(os.environ["SLURM_NODEID"])

    old_wait_url = c8.base.common._wait_url

    def wait_url(url: str, processes: list[object]) -> None:
        old_wait_url(url, processes)
        (result_dir / f"node-{node_index}-vllm-ready").write_text(
            "ready\n", encoding="utf-8")
        timeout_s = float(os.environ.get(
            "TEMPO_GO_C9_INFERENCE_START_TIMEOUT_S", "1800"))
        deadline = time.monotonic() + timeout_s
        while not start_file.is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError("C9 inference start gate timed out")
            time.sleep(0.1)

    c8.base.common._wait_url = wait_url
    try:
        return c8.main()
    finally:
        c8.base.common._wait_url = old_wait_url


if __name__ == "__main__":
    raise SystemExit(main())
