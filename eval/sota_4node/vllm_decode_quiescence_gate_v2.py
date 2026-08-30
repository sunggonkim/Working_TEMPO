#!/usr/bin/env python3
"""Falsification-safe token31 quiescence protocol for the TP16 scout.

Version 2 keeps the add-only v1 engine seam and framing, but separates
protocol correctness from the research performance gate.  An exact
8-descriptor/4-MiB wave is always released and recorded, even when a source
call takes more than 4 ms or the wave takes more than 5 ms.  Those limits are
reported through ``service_gate_met`` rather than converted into a harness
failure.

The hook is explicitly node-zero-only.  A launcher enabling it must set
``TEMPO_VLLM_QUIESCENCE_NODE_RANK=0`` only in node zero's vLLM environment,
prepend the dedicated v2 sitecustomize directory only there, and force
``--no-async-scheduling``.  The sidecar must enter ``GateListener`` before it
starts the HTTP request; a missing listener fails closed at token31.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from eval.sota_4node import vllm_decode_quiescence_gate_v1 as v1


PROTOCOL = v1.PROTOCOL
NODE_RANK_ENV = "TEMPO_VLLM_QUIESCENCE_NODE_RANK"


class ReleaseFrame(v1.ReleaseFrame):
    @property
    def service_gate_met(self) -> bool:
        if self.mode == "quiescent_noop":
            return True
        return (
            self.mode == "quiescent_512k"
            and len(self.source_elapsed_ns) == v1.SOURCE_COUNT
            and all(
                0 <= value <= v1.MAX_SOURCE_ELAPSED_NS
                for value in self.source_elapsed_ns
            )
            and 0 <= self.wave_elapsed_ns <= v1.MAX_WAVE_ELAPSED_NS
        )

    def validate(self, event: v1.ReadyEvent | None = None) -> None:
        if self.event_id < 0:
            raise ValueError("release event_id must be nonnegative")
        if not self.request_id:
            raise ValueError("release request_id is required")
        if event is not None and (
            self.event_id != event.event_id or self.request_id != event.request_id
        ):
            raise ValueError("release does not match ready event")
        if self.released_ns <= 0:
            raise ValueError("released_ns must be positive")
        if self.mode == "quiescent_noop":
            if (
                self.completed_sources != 0
                or self.physical_descriptors != 0
                or self.completed_bytes != 0
                or self.source_elapsed_ns
                or self.wave_elapsed_ns != 0
            ):
                raise ValueError("quiescent_noop must have zero transfer geometry")
            return
        if self.mode != "quiescent_512k":
            raise ValueError(f"unknown quiescence release mode: {self.mode}")
        if self.completed_sources != v1.SOURCE_COUNT:
            raise ValueError("512 KiB wave must complete all eight sources")
        if self.physical_descriptors != v1.SOURCE_COUNT:
            raise ValueError("512 KiB wave must complete exactly eight descriptors")
        if self.completed_bytes != v1.GLOBAL_BYTES:
            raise ValueError("512 KiB wave must complete exactly 4 MiB globally")
        if len(self.source_elapsed_ns) != v1.SOURCE_COUNT:
            raise ValueError("512 KiB wave requires eight source durations")
        if any(value < 0 for value in self.source_elapsed_ns):
            raise ValueError("512 KiB source durations must be nonnegative")
        if self.wave_elapsed_ns < 0:
            raise ValueError("512 KiB wave duration must be nonnegative")

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload["service_gate_met"] = self.service_gate_met
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, event: v1.ReadyEvent
    ) -> "ReleaseFrame":
        frame = super().from_payload(payload, event=event)
        if not isinstance(frame, cls):
            raise TypeError("release decoder did not preserve the v2 frame type")
        reported = payload.get("service_gate_met")
        if not isinstance(reported, bool):
            raise ValueError("service_gate_met must be a boolean")
        if reported != frame.service_gate_met:
            raise ValueError("service_gate_met disagrees with recorded durations")
        return frame


# The inherited client and connection resolve ReleaseFrame from the v1 module
# at call time.  Replace that one symbol so every v2 user, including the
# inherited EngineCore patch, decodes and emits the falsification-safe frame.
v1.ReleaseFrame = ReleaseFrame


def config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> v1.GateConfig | None:
    values = os.environ if environ is None else environ
    config = v1.config_from_environment(values)
    if config is None:
        return None
    if values.get(NODE_RANK_ENV) != "0":
        raise ValueError(
            f"{NODE_RANK_ENV} must be exactly 0; the hook is node-zero-only"
        )
    return config


def install_from_environment(
    environ: Mapping[str, str] | None = None,
) -> bool:
    config = config_from_environment(environ)
    if config is None:
        return False
    from vllm.v1.engine.core import EngineCoreProc

    return v1.patch_engine_core_class(EngineCoreProc, config)


DESCRIPTOR_BYTES = v1.DESCRIPTOR_BYTES
GLOBAL_BYTES = v1.GLOBAL_BYTES
GateClient = v1.GateClient
GateConfig = v1.GateConfig
GateConnection = v1.GateConnection
GateListener = v1.GateListener
MAX_SOURCE_ELAPSED_NS = v1.MAX_SOURCE_ELAPSED_NS
MAX_WAVE_ELAPSED_NS = v1.MAX_WAVE_ELAPSED_NS
OutputBoundaryDetector = v1.OutputBoundaryDetector
ReadyEvent = v1.ReadyEvent
SOURCE_COUNT = v1.SOURCE_COUNT
TARGET_OUTPUT_TOKEN_COUNT = v1.TARGET_OUTPUT_TOKEN_COUNT
TARGET_OUTPUT_TOKEN_INDEX = v1.TARGET_OUTPUT_TOKEN_INDEX
patch_engine_core_class = v1.patch_engine_core_class


__all__ = [
    "DESCRIPTOR_BYTES",
    "GLOBAL_BYTES",
    "GateClient",
    "GateConfig",
    "GateConnection",
    "GateListener",
    "MAX_SOURCE_ELAPSED_NS",
    "MAX_WAVE_ELAPSED_NS",
    "NODE_RANK_ENV",
    "OutputBoundaryDetector",
    "PROTOCOL",
    "ReadyEvent",
    "ReleaseFrame",
    "SOURCE_COUNT",
    "TARGET_OUTPUT_TOKEN_COUNT",
    "TARGET_OUTPUT_TOKEN_INDEX",
    "config_from_environment",
    "install_from_environment",
    "patch_engine_core_class",
]
