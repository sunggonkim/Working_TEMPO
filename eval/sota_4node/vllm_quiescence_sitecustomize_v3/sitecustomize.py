"""Fail-closed activation of the launchable node-zero quiescence hook."""

import os
import sys
import traceback


if os.environ.get("TEMPO_VLLM_QUIESCENCE_ENABLED") == "YES":
    try:
        from eval.sota_4node.vllm_decode_quiescence_gate_launch_v3 import (
            install_from_environment,
        )

        if not install_from_environment():
            raise RuntimeError("quiescence launch v3 was not installed")
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(78)
