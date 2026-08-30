"""Fail-closed activation of the pinned hook for the 128 MiB hybrid."""

import os
import sys
import traceback


if os.environ.get("TEMPO_VLLM_QUIESCENCE_ENABLED") == "YES":
    try:
        from eval.sota_4node.vllm_quiescence_wave_protocol_v5 import (
            install_generic_release_protocol,
        )
        from eval.sota_4node.vllm_decode_quiescence_gate_launch_v3_hardening import (
            install_pinned_from_environment,
        )

        install_generic_release_protocol()
        if not install_pinned_from_environment():
            raise RuntimeError("pinned hybrid quiescence hook was not installed")
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(78)
