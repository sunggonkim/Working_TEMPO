#!/usr/bin/env python3
"""Research-only vLLM output-boundary quiescence gate.

This module provides the small control-plane seam needed by the four-node
TP16 scout.  A dedicated ``sitecustomize`` entrypoint patches
``EngineCoreProc`` so that, after zero-based output token 31 has been put on
the engine output queue, the engine waits for one bounded Unix-socket
release.  The next engine step therefore cannot start before the sidecar has
either completed the declared 8 x 512 KiB wave or explicitly selected the
no-op control.

The seam deliberately refuses vLLM asynchronous scheduling.  vLLM 0.26 uses
two concurrent batches when async scheduling is enabled, which means a later
decode can already be in flight when ``_process_engine_step`` returns.  The
corresponding launcher must pass ``--no-async-scheduling``.

This is an experiment hook, not a production extension API.  It does not
touch vLLM's installed files and is inactive unless the explicit environment
gate is set to ``YES``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Mapping


PROTOCOL = "tempo-vllm-output-quiescence-1"
ENABLE_ENV = "TEMPO_VLLM_QUIESCENCE_ENABLED"
SOCKET_ENV = "TEMPO_VLLM_QUIESCENCE_SOCKET"
TOKEN_INDEX_ENV = "TEMPO_VLLM_QUIESCENCE_TOKEN_INDEX"
TIMEOUT_ENV = "TEMPO_VLLM_QUIESCENCE_TIMEOUT_S"

TARGET_OUTPUT_TOKEN_INDEX = 31
TARGET_OUTPUT_TOKEN_COUNT = TARGET_OUTPUT_TOKEN_INDEX + 1
SOURCE_COUNT = 8
DESCRIPTOR_BYTES = 512 << 10
GLOBAL_BYTES = SOURCE_COUNT * DESCRIPTOR_BYTES
MAX_SOURCE_ELAPSED_NS = 4_000_000
MAX_WAVE_ELAPSED_NS = 5_000_000
MAX_FRAME_BYTES = 4096


@dataclass(frozen=True)
class GateConfig:
    socket_path: Path
    token_index: int = TARGET_OUTPUT_TOKEN_INDEX
    timeout_s: float = 10.0


@dataclass(frozen=True)
class ReadyEvent:
    event_id: int
    request_id: str
    output_token_index: int
    output_token_count: int
    ready_ns: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "kind": "ready",
            "event_id": self.event_id,
            "request_id": self.request_id,
            "output_token_index": self.output_token_index,
            "output_token_count": self.output_token_count,
            "ready_ns": self.ready_ns,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReadyEvent":
        if payload.get("protocol") != PROTOCOL or payload.get("kind") != "ready":
            raise ValueError("invalid quiescence ready frame")
        event = cls(
            event_id=_exact_int(payload.get("event_id"), "event_id", minimum=0),
            request_id=str(payload.get("request_id", "")),
            output_token_index=_exact_int(
                payload.get("output_token_index"),
                "output_token_index",
                minimum=0,
            ),
            output_token_count=_exact_int(
                payload.get("output_token_count"),
                "output_token_count",
                minimum=1,
            ),
            ready_ns=_exact_int(payload.get("ready_ns"), "ready_ns", minimum=1),
        )
        if not event.request_id or len(event.request_id) > 512:
            raise ValueError("request_id must contain 1..512 characters")
        if event.output_token_index != TARGET_OUTPUT_TOKEN_INDEX:
            raise ValueError("quiescence ready frame changed token index")
        if event.output_token_count != TARGET_OUTPUT_TOKEN_COUNT:
            raise ValueError("quiescence ready frame changed token count")
        return event


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

    @classmethod
    def noop(cls, event: ReadyEvent) -> "ReleaseFrame":
        return cls(
            event_id=event.event_id,
            request_id=event.request_id,
            mode="quiescent_noop",
            completed_sources=0,
            physical_descriptors=0,
            completed_bytes=0,
            source_elapsed_ns=(),
            wave_elapsed_ns=0,
            released_ns=time.perf_counter_ns(),
        )

    @classmethod
    def wave512k(
        cls,
        event: ReadyEvent,
        *,
        source_elapsed_ns: tuple[int, ...],
        wave_elapsed_ns: int,
    ) -> "ReleaseFrame":
        return cls(
            event_id=event.event_id,
            request_id=event.request_id,
            mode="quiescent_512k",
            completed_sources=SOURCE_COUNT,
            physical_descriptors=SOURCE_COUNT,
            completed_bytes=GLOBAL_BYTES,
            source_elapsed_ns=source_elapsed_ns,
            wave_elapsed_ns=wave_elapsed_ns,
            released_ns=time.perf_counter_ns(),
        )

    def validate(self, event: ReadyEvent | None = None) -> None:
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
        if self.completed_sources != SOURCE_COUNT:
            raise ValueError("512 KiB wave must complete all eight sources")
        if self.physical_descriptors != SOURCE_COUNT:
            raise ValueError("512 KiB wave must complete exactly eight descriptors")
        if self.completed_bytes != GLOBAL_BYTES:
            raise ValueError("512 KiB wave must complete exactly 4 MiB globally")
        if len(self.source_elapsed_ns) != SOURCE_COUNT:
            raise ValueError("512 KiB wave requires eight source durations")
        if any(
            value < 0 or value > MAX_SOURCE_ELAPSED_NS
            for value in self.source_elapsed_ns
        ):
            raise ValueError("512 KiB source call exceeded the 4 ms gate")
        if not 0 <= self.wave_elapsed_ns <= MAX_WAVE_ELAPSED_NS:
            raise ValueError("512 KiB wave exceeded the 5 ms gate")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol": PROTOCOL,
            "kind": "release",
            "event_id": self.event_id,
            "request_id": self.request_id,
            "mode": self.mode,
            "completed_sources": self.completed_sources,
            "physical_descriptors": self.physical_descriptors,
            "completed_bytes": self.completed_bytes,
            "source_elapsed_ns": list(self.source_elapsed_ns),
            "wave_elapsed_ns": self.wave_elapsed_ns,
            "released_ns": self.released_ns,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, event: ReadyEvent
    ) -> "ReleaseFrame":
        if payload.get("protocol") != PROTOCOL or payload.get("kind") != "release":
            raise ValueError("invalid quiescence release frame")
        durations = payload.get("source_elapsed_ns")
        if not isinstance(durations, list):
            raise ValueError("source_elapsed_ns must be a list")
        frame = cls(
            event_id=_exact_int(payload.get("event_id"), "event_id", minimum=0),
            request_id=str(payload.get("request_id", "")),
            mode=str(payload.get("mode", "")),
            completed_sources=_exact_int(
                payload.get("completed_sources"), "completed_sources", minimum=0
            ),
            physical_descriptors=_exact_int(
                payload.get("physical_descriptors"),
                "physical_descriptors",
                minimum=0,
            ),
            completed_bytes=_exact_int(
                payload.get("completed_bytes"), "completed_bytes", minimum=0
            ),
            source_elapsed_ns=tuple(
                _exact_int(value, "source_elapsed_ns item", minimum=0)
                for value in durations
            ),
            wave_elapsed_ns=_exact_int(
                payload.get("wave_elapsed_ns"), "wave_elapsed_ns", minimum=0
            ),
            released_ns=_exact_int(
                payload.get("released_ns"), "released_ns", minimum=1
            ),
        )
        frame.validate(event)
        return frame


def _exact_int(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def validate_socket_path(path: Path) -> Path:
    if not path.is_absolute() or path.parent != Path("/tmp"):
        raise ValueError("quiescence socket must be an immediate child of /tmp")
    if not path.name.startswith("tempo-vllm-quiescence-") or not path.name.endswith(
        ".sock"
    ):
        raise ValueError("quiescence socket name must use the frozen prefix/suffix")
    if len(os.fsencode(path)) > 100:
        raise ValueError("quiescence Unix-socket path is too long")
    return path


def config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> GateConfig | None:
    values = os.environ if environ is None else environ
    enabled = values.get(ENABLE_ENV)
    if enabled is None:
        return None
    if enabled != "YES":
        raise ValueError(f"{ENABLE_ENV} must be exactly YES when set")
    raw_path = values.get(SOCKET_ENV)
    if not raw_path:
        raise ValueError(f"{SOCKET_ENV} is required")
    raw_token = values.get(TOKEN_INDEX_ENV, str(TARGET_OUTPUT_TOKEN_INDEX))
    try:
        token_index = int(raw_token)
    except ValueError as exc:
        raise ValueError(f"{TOKEN_INDEX_ENV} must be an integer") from exc
    if token_index != TARGET_OUTPUT_TOKEN_INDEX:
        raise ValueError("v1 scout is frozen to zero-based output token index 31")
    raw_timeout = values.get(TIMEOUT_ENV, "10")
    try:
        timeout_s = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"{TIMEOUT_ENV} must be a number") from exc
    if not 0.1 <= timeout_s <= 30.0:
        raise ValueError(f"{TIMEOUT_ENV} must be in 0.1..30 seconds")
    return GateConfig(
        socket_path=validate_socket_path(Path(raw_path)),
        token_index=token_index,
        timeout_s=timeout_s,
    )


def _send_payload(stream: socket.socket, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(payload), separators=(",", ":"), sort_keys=True
    ).encode("utf-8") + b"\n"
    if len(encoded) > MAX_FRAME_BYTES:
        raise ValueError("quiescence frame exceeds 4096 bytes")
    stream.sendall(encoded)


def _receive_payload(stream: socket.socket) -> dict[str, Any]:
    chunks = bytearray()
    while b"\n" not in chunks:
        piece = stream.recv(min(1024, MAX_FRAME_BYTES + 1 - len(chunks)))
        if not piece:
            raise RuntimeError("quiescence peer closed before a complete frame")
        chunks.extend(piece)
        if len(chunks) > MAX_FRAME_BYTES:
            raise ValueError("quiescence frame exceeds 4096 bytes")
    line, trailing = bytes(chunks).split(b"\n", 1)
    if trailing:
        raise ValueError("quiescence connection must contain exactly one frame")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid quiescence JSON frame") from exc
    if not isinstance(payload, dict):
        raise ValueError("quiescence frame must contain a JSON object")
    return payload


class GateClient:
    def __init__(self, config: GateConfig) -> None:
        self.config = config

    def wait_release(self, event: ReadyEvent) -> ReleaseFrame:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(self.config.timeout_s)
            stream.connect(os.fspath(self.config.socket_path))
            _send_payload(stream, event.to_payload())
            return ReleaseFrame.from_payload(_receive_payload(stream), event=event)


class GateConnection:
    def __init__(self, stream: socket.socket, event: ReadyEvent) -> None:
        self._stream = stream
        self.event = event
        self._released = False

    def release(self, frame: ReleaseFrame) -> None:
        if self._released:
            raise RuntimeError("quiescence connection was already released")
        frame.validate(self.event)
        _send_payload(self._stream, frame.to_payload())
        self._released = True

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "GateConnection":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


class GateListener:
    """Synchronous, one-event-at-a-time sidecar endpoint.

    No watcher thread is created.  The sidecar rank-zero main loop calls
    ``accept`` exactly once for each 64-token scout request.
    """

    def __init__(self, config: GateConfig) -> None:
        self.config = config
        self._listener: socket.socket | None = None
        self._owns_path = False

    def open(self) -> None:
        if self._listener is not None:
            raise RuntimeError("quiescence listener is already open")
        if self.config.socket_path.exists():
            raise RuntimeError(
                f"refusing to replace existing quiescence socket: "
                f"{self.config.socket_path}"
            )
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.settimeout(self.config.timeout_s)
        try:
            listener.bind(os.fspath(self.config.socket_path))
            self._owns_path = True
            listener.listen(1)
        except BaseException:
            listener.close()
            if self._owns_path:
                self.config.socket_path.unlink(missing_ok=True)
                self._owns_path = False
            raise
        self._listener = listener

    def accept(self) -> GateConnection:
        if self._listener is None:
            raise RuntimeError("quiescence listener is not open")
        stream, _ = self._listener.accept()
        stream.settimeout(self.config.timeout_s)
        try:
            event = ReadyEvent.from_payload(_receive_payload(stream))
        except BaseException:
            stream.close()
            raise
        return GateConnection(stream, event)

    def close(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.close()
        if self._owns_path:
            self.config.socket_path.unlink(missing_ok=True)
            self._owns_path = False

    def __enter__(self) -> "GateListener":
        self.open()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


class OutputBoundaryDetector:
    def __init__(self, token_index: int = TARGET_OUTPUT_TOKEN_INDEX) -> None:
        if token_index != TARGET_OUTPUT_TOKEN_INDEX:
            raise ValueError("v1 detector is frozen to output token index 31")
        self.token_index = token_index
        self._counts: dict[str, int] = {}
        self._pending: list[ReadyEvent] = []
        self._next_event_id = 0

    def observe_queue_item(self, item: Any) -> None:
        if not isinstance(item, tuple) or len(item) != 2:
            return
        engine_outputs = item[1]
        outputs = getattr(engine_outputs, "outputs", ())
        if not isinstance(outputs, (list, tuple)):
            return
        finished: list[str] = []
        for output in outputs:
            request_id = str(getattr(output, "request_id", ""))
            if not request_id:
                continue
            token_ids = getattr(output, "new_token_ids", ())
            if not isinstance(token_ids, (list, tuple)):
                raise RuntimeError("EngineCoreOutput.new_token_ids changed type")
            before = self._counts.get(request_id, 0)
            after = before + len(token_ids)
            if before <= self.token_index < after:
                if len(token_ids) != 1 or before != self.token_index:
                    raise RuntimeError(
                        "token31 boundary was crossed by a multi-token output; "
                        "the scout requires ordinary one-token decoding"
                    )
                self._pending.append(
                    ReadyEvent(
                        event_id=self._next_event_id,
                        request_id=request_id,
                        output_token_index=self.token_index,
                        output_token_count=after,
                        ready_ns=time.perf_counter_ns(),
                    )
                )
                self._next_event_id += 1
            self._counts[request_id] = after
            if bool(getattr(output, "finished", False)):
                finished.append(request_id)
        for request_id in finished:
            self._counts.pop(request_id, None)

    def pop_pending(self) -> ReadyEvent | None:
        if len(self._pending) > 1:
            raise RuntimeError(
                "multiple requests hit token31 in one engine step; "
                "the scout requires a single foreground request"
            )
        return self._pending.pop() if self._pending else None


class _OutputQueueTap:
    def __init__(self, target: Any, detector: OutputBoundaryDetector) -> None:
        self._target = target
        self._detector = detector

    def put_nowait(self, item: Any) -> None:
        self._target.put_nowait(item)
        self._detector.observe_queue_item(item)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def patch_engine_core_class(
    engine_core_class: type[Any],
    config: GateConfig,
    *,
    client_factory: Callable[[GateConfig], Any] = GateClient,
) -> bool:
    """Patch one EngineCoreProc class; returns False if already installed."""

    marker = "_tempo_output_quiescence_v1_installed"
    if bool(getattr(engine_core_class, marker, False)):
        return False
    original_init = engine_core_class.__init__
    original_process_step = engine_core_class._process_engine_step

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if bool(getattr(self, "async_scheduling", False)):
            raise RuntimeError(
                "TEMPO token31 quiescence requires --no-async-scheduling"
            )
        if getattr(self, "batch_queue", None) is not None:
            raise RuntimeError("TEMPO token31 quiescence requires no batch queue")
        if int(getattr(self, "batch_queue_size", -1)) != 1:
            raise RuntimeError("TEMPO token31 quiescence requires one batch in flight")
        detector = OutputBoundaryDetector(config.token_index)
        self.output_queue = _OutputQueueTap(self.output_queue, detector)
        self._tempo_quiescence_detector_v1 = detector
        self._tempo_quiescence_client_v1 = client_factory(config)
        self._tempo_quiescence_trace_v1 = []

    def patched_process_step(self: Any) -> bool:
        entered_ns = time.perf_counter_ns()
        trace = self._tempo_quiescence_trace_v1
        if trace and "next_engine_step_enter_ns" not in trace[-1]:
            trace[-1]["next_engine_step_enter_ns"] = entered_ns
            if entered_ns < int(trace[-1]["released_ns"]):
                raise RuntimeError("next engine step entered before quiescence release")
        model_executed = original_process_step(self)
        event = self._tempo_quiescence_detector_v1.pop_pending()
        if event is not None:
            release = self._tempo_quiescence_client_v1.wait_release(event)
            returned_ns = time.perf_counter_ns()
            if returned_ns < release.released_ns:
                raise RuntimeError("local gate returned before declared release")
            trace.append(
                {
                    "event_id": event.event_id,
                    "request_id": event.request_id,
                    "output_token_index": event.output_token_index,
                    "output_token_count": event.output_token_count,
                    "ready_ns": event.ready_ns,
                    "mode": release.mode,
                    "completed_sources": release.completed_sources,
                    "physical_descriptors": release.physical_descriptors,
                    "completed_bytes": release.completed_bytes,
                    "source_elapsed_ns": list(release.source_elapsed_ns),
                    "wave_elapsed_ns": release.wave_elapsed_ns,
                    "released_ns": release.released_ns,
                    "gate_returned_ns": returned_ns,
                }
            )
        return model_executed

    engine_core_class.__init__ = patched_init
    engine_core_class._process_engine_step = patched_process_step
    setattr(engine_core_class, marker, True)
    return True


def install_from_environment(
    environ: Mapping[str, str] | None = None,
) -> bool:
    config = config_from_environment(environ)
    if config is None:
        return False
    from vllm.v1.engine.core import EngineCoreProc

    return patch_engine_core_class(EngineCoreProc, config)


__all__ = [
    "DESCRIPTOR_BYTES",
    "GLOBAL_BYTES",
    "GateClient",
    "GateConfig",
    "GateConnection",
    "GateListener",
    "MAX_SOURCE_ELAPSED_NS",
    "MAX_WAVE_ELAPSED_NS",
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
