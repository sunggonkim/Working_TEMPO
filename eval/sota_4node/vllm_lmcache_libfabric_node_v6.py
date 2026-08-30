#!/usr/bin/env python3
"""Launch-safe LIBFABRIC node wrapper with an explicit vLLM environment root."""

from __future__ import annotations

import os
from pathlib import Path

from eval.sota_4node import vllm_lmcache_libfabric_node_v5 as prior


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    environment = repo / ".vllm_venv"
    if not (environment / "bin/python").is_file():
        raise RuntimeError(f"missing vLLM environment: {environment}")
    os.environ["VIRTUAL_ENV"] = str(environment)
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
