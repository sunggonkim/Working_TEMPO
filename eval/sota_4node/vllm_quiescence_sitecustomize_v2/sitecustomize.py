"""Fail-closed, node-zero-only activation for the TP16 token31 scout."""

from __future__ import annotations

import os
import sys
import traceback


if os.environ.get("TEMPO_VLLM_QUIESCENCE_ENABLED") == "YES":
    try:
        from eval.sota_4node.vllm_decode_quiescence_gate_v2 import (
            install_from_environment,
        )

        if not install_from_environment():
            raise RuntimeError("TEMPO quiescence v2 patch was not installed")
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(78)
