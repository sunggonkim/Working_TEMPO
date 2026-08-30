#!/usr/bin/env python3
"""Balanced cache-free NIXL progress: 16 yields then 100us backoff."""

from __future__ import annotations

import json
import os
from pathlib import Path

from eval.sota_4node import vllm_lmcache_nixl_polling_snapshot_node_v3 as prior


_ORIGINAL_ENVIRONMENT = prior._environment


def _environment(*args, **kwargs):
    env = _ORIGINAL_ENVIRONMENT(*args, **kwargs)
    env["TEMPO_NIXL_YIELD_POLLS"] = "16"
    return env


def main() -> int:
    prior._environment = _environment
    code = prior.main()
    if os.environ.get("SLURM_NODEID") == "0":
        result_dir = None
        arguments = os.sys.argv[1:]
        for index, value in enumerate(arguments):
            if value == "--result-dir":
                result_dir = Path(arguments[index + 1]).resolve()
                break
        if result_dir is None:
            raise RuntimeError("--result-dir is required")
        path = result_dir / "result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema"] = "lmcache-nixl-polling-ab-analysis-4"
        payload["optimization"] = {
            "prepared_handle_cache": "disabled after zero hits",
            "completion_progress": "16 cooperative yield polls, then 100us async sleep",
        }
        temporary = result_dir / ".result-v4.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
