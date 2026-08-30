"""Fail-closed activation of the version/source-pinned v3 hook."""

import os
import sys
import traceback


if os.environ.get("TEMPO_VLLM_QUIESCENCE_ENABLED") == "YES":
    try:
        from eval.sota_4node.vllm_decode_quiescence_gate_launch_v3_hardening import (
            install_pinned_from_environment,
        )

        if not install_pinned_from_environment():
            raise RuntimeError("pinned quiescence v3 was not installed")
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(78)
