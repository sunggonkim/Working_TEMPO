#!/usr/bin/env python3
"""Truthful nonblocking token-31 promotion release codec."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

from eval.sota_4node import vllm_decode_quiescence_gate_launch_v3 as hook


SOURCE_COUNT = 8
NOOP_MODE = "quiescent_noop"
ASYNC_MODE = "tempo_async_promotion"


def _integer(payload: Mapping[str, Any], name: str, minimum: int) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ReleaseFrame:
    event_id: int
    request_id: str
    mode: str
    completed_sources: int
    physical_descriptors: int
    completed_bytes: int
    source_elapsed_ns: tuple[int, ...]
    wave_elapsed_ns: int
    released_ns: int
    promotion_armed_sources: int = 0

    @classmethod
    def noop(cls, event: hook.ReadyEvent) -> "ReleaseFrame":
        return cls(event.event_id, event.request_id, NOOP_MODE, 0, 0, 0, (), 0,
                   time.perf_counter_ns(), 0)

    @classmethod
    def promotion(
        cls, event: hook.ReadyEvent, *, promotion_armed_sources: int
    ) -> "ReleaseFrame":
        return cls(
            event.event_id,
            event.request_id,
            ASYNC_MODE,
            0,
            0,
            0,
            (),
            0,
            time.perf_counter_ns(),
            promotion_armed_sources,
        )

    def validate(self, event: hook.ReadyEvent | None = None) -> None:
        if self.event_id < 0 or not self.request_id or self.released_ns <= 0:
            raise ValueError("release identity/timestamp is invalid")
        if event is not None and (
            self.event_id != event.event_id or self.request_id != event.request_id
        ):
            raise ValueError("release does not match ready event")
        if self.mode == NOOP_MODE:
            if any((self.completed_sources, self.physical_descriptors,
                    self.completed_bytes, len(self.source_elapsed_ns),
                    self.wave_elapsed_ns, self.promotion_armed_sources)):
                raise ValueError("noop release must contain only zero geometry")
            return
        if self.mode != ASYNC_MODE:
            raise ValueError(f"unknown async release mode: {self.mode}")
        if any((self.completed_sources, self.physical_descriptors,
                self.completed_bytes, len(self.source_elapsed_ns),
                self.wave_elapsed_ns)):
            raise ValueError("async promotion cannot claim transfer completion")
        if self.promotion_armed_sources != SOURCE_COUNT:
            raise ValueError("async promotion must arm exactly eight sources")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol": hook.PROTOCOL,
            "kind": "release",
            "event_id": self.event_id,
            "request_id": self.request_id,
            "mode": self.mode,
            "completed_sources": self.completed_sources,
            "physical_descriptors": self.physical_descriptors,
            "completed_bytes": self.completed_bytes,
            "source_elapsed_ns": list(self.source_elapsed_ns),
            "wave_elapsed_ns": self.wave_elapsed_ns,
            "promotion_armed_sources": self.promotion_armed_sources,
            "released_ns": self.released_ns,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, event: hook.ReadyEvent
    ) -> "ReleaseFrame":
        durations = payload.get("source_elapsed_ns")
        if (payload.get("protocol") != hook.PROTOCOL
                or payload.get("kind") != "release"
                or not isinstance(durations, list)):
            raise ValueError("invalid async promotion release frame")
        frame = cls(
            _integer(payload, "event_id", 0),
            str(payload.get("request_id", "")),
            str(payload.get("mode", "")),
            _integer(payload, "completed_sources", 0),
            _integer(payload, "physical_descriptors", 0),
            _integer(payload, "completed_bytes", 0),
            tuple(_integer({"v": value}, "v", 0) for value in durations),
            _integer(payload, "wave_elapsed_ns", 0),
            _integer(payload, "released_ns", 1),
            _integer(payload, "promotion_armed_sources", 0),
        )
        frame.validate(event)
        return frame


def install_async_release_protocol() -> None:
    hook.ReleaseFrame = ReleaseFrame
    hook.wire.ReleaseFrame = ReleaseFrame


__all__ = [
    "ASYNC_MODE", "NOOP_MODE", "ReleaseFrame", "install_async_release_protocol"
]
