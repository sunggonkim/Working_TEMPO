#!/usr/bin/env python3
"""128 MiB prelaunch/boost release mode layered on the v4 codec."""

from eval.sota_4node import vllm_quiescence_wave_protocol_v4 as _v4


HYBRID_MODE = "tempo_prelaunch_quiescent_boost"
_v4.DATA_MODES = frozenset((*_v4.DATA_MODES, HYBRID_MODE))

ReleaseFrame = _v4.ReleaseFrame
NOOP_MODE = _v4.NOOP_MODE


def install_generic_release_protocol() -> None:
    _v4.DATA_MODES = frozenset((*_v4.DATA_MODES, HYBRID_MODE))
    _v4.install_generic_release_protocol()


__all__ = [
    "HYBRID_MODE",
    "NOOP_MODE",
    "ReleaseFrame",
    "install_generic_release_protocol",
]
