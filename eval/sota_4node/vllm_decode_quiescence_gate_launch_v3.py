#!/usr/bin/env python3
"""Launchable TP16 quiescence seam for the bounded token-31 scout.

The hook is installed only in node zero's ``EngineCoreProc``.  After the
31st generated token (one-based count 31, zero-based client index 30) has
already been enqueued for output, it executes a callable RPC on all sixteen
workers.  Each worker synchronizes its current accelerator before replying.
Only after all replies arrive does the hook publish READY and wait for the
sidecar RELEASE.  vLLM async scheduling and speculative decoding are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Mapping

from eval.sota_4node import vllm_decode_quiescence_gate_v1 as wire


PROTOCOL = "tempo-vllm-output-quiescence-3"
ENABLE_ENV = "TEMPO_VLLM_QUIESCENCE_ENABLED"
SOCKET_ENV = "TEMPO_VLLM_QUIESCENCE_SOCKET"
TRACE_ENV = "TEMPO_VLLM_QUIESCENCE_TRACE"
NODE_RANK_ENV = "TEMPO_VLLM_QUIESCENCE_NODE_RANK"
TOKEN_INDEX_ENV = "TEMPO_VLLM_QUIESCENCE_TOKEN_INDEX"
TIMEOUT_ENV = "TEMPO_VLLM_QUIESCENCE_TIMEOUT_S"
REQUEST_ID_MARKER = "tempo-scout-"

TARGET_OUTPUT_TOKEN_INDEX = 30
TARGET_GENERATED_TOKEN_COUNT = 31
WORLD_SIZE = 16
SOURCE_COUNT = 8
DESCRIPTOR_BYTES = 512 << 10
GLOBAL_BYTES = SOURCE_COUNT * DESCRIPTOR_BYTES
MAX_SOURCE_ELAPSED_NS = 4_000_000
MAX_WAVE_ELAPSED_NS = 5_000_000


@dataclass(frozen=True)
class GateConfig:
    socket_path: Path
    trace_path: Path
    token_index: int = TARGET_OUTPUT_TOKEN_INDEX
    timeout_s: float = 10.0


@dataclass(frozen=True)
class ReadyEvent:
    event_id: int
    request_id: str
    output_token_index: int
    generated_token_count: int
    output_enqueued_ns: int
    fence_started_ns: int = 0
    fence_finished_ns: int = 0
    ready_ns: int = 0

    def to_payload(self) -> dict[str, Any]:
        if REQUEST_ID_MARKER not in self.request_id:
            raise ValueError("ready request_id lacks the scout marker")
        if self.output_token_index != TARGET_OUTPUT_TOKEN_INDEX:
            raise ValueError("ready output token index changed")
        if self.generated_token_count != TARGET_GENERATED_TOKEN_COUNT:
            raise ValueError("ready generated-token count changed")
        if not (
            0 < self.output_enqueued_ns <= self.fence_started_ns
            <= self.fence_finished_ns <= self.ready_ns
        ):
            raise ValueError("ready timeline is not monotonic")
        return {
            "protocol": PROTOCOL,
            "kind": "ready",
            "event_id": self.event_id,
            "request_id": self.request_id,
            "output_token_index_zero_based": self.output_token_index,
            "generated_token_count_one_based": self.generated_token_count,
            "output_enqueued_ns": self.output_enqueued_ns,
            "fence_started_ns": self.fence_started_ns,
            "fence_finished_ns": self.fence_finished_ns,
            "ready_ns": self.ready_ns,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReadyEvent":
        if payload.get("protocol") != PROTOCOL or payload.get("kind") != "ready":
            raise ValueError("invalid v3 ready frame")
        event = cls(
            event_id=_integer(payload, "event_id", 0),
            request_id=str(payload.get("request_id", "")),
            output_token_index=_integer(
                payload, "output_token_index_zero_based", 0
            ),
            generated_token_count=_integer(
                payload, "generated_token_count_one_based", 1
            ),
            output_enqueued_ns=_integer(payload, "output_enqueued_ns", 1),
            fence_started_ns=_integer(payload, "fence_started_ns", 1),
            fence_finished_ns=_integer(payload, "fence_finished_ns", 1),
            ready_ns=_integer(payload, "ready_ns", 1),
        )
        event.to_payload()
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

    @property
    def service_gate_met(self) -> bool:
        if self.mode == "quiescent_noop":
            return True
        return (
            self.mode == "quiescent_512k"
            and len(self.source_elapsed_ns) == SOURCE_COUNT
            and all(value <= MAX_SOURCE_ELAPSED_NS for value in self.source_elapsed_ns)
            and self.wave_elapsed_ns <= MAX_WAVE_ELAPSED_NS
        )

    @classmethod
    def noop(cls, event: ReadyEvent) -> "ReleaseFrame":
        return cls(
            event.event_id, event.request_id, "quiescent_noop", 0, 0, 0, (), 0,
            time.perf_counter_ns(),
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
            event.event_id,
            event.request_id,
            "quiescent_512k",
            SOURCE_COUNT,
            SOURCE_COUNT,
            GLOBAL_BYTES,
            source_elapsed_ns,
            wave_elapsed_ns,
            time.perf_counter_ns(),
        )

    def validate(self, event: ReadyEvent | None = None) -> None:
        if event is not None and (
            self.event_id != event.event_id or self.request_id != event.request_id
        ):
            raise ValueError("release does not match ready event")
        if self.released_ns <= 0:
            raise ValueError("released_ns must be positive")
        if self.mode == "quiescent_noop":
            if any(
                (
                    self.completed_sources,
                    self.physical_descriptors,
                    self.completed_bytes,
                    len(self.source_elapsed_ns),
                    self.wave_elapsed_ns,
                )
            ):
                raise ValueError("noop release must have zero transfer geometry")
            return
        if self.mode != "quiescent_512k":
            raise ValueError("unknown release mode")
        if (
            self.completed_sources != SOURCE_COUNT
            or self.physical_descriptors != SOURCE_COUNT
            or self.completed_bytes != GLOBAL_BYTES
            or len(self.source_elapsed_ns) != SOURCE_COUNT
        ):
            raise ValueError("wave release must be exact 8 descriptors / 4 MiB")
        if any(value < 0 for value in self.source_elapsed_ns) or self.wave_elapsed_ns < 0:
            raise ValueError("wave durations must be nonnegative")

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
            "service_gate_met": self.service_gate_met,
            "released_ns": self.released_ns,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, event: ReadyEvent
    ) -> "ReleaseFrame":
        if payload.get("protocol") != PROTOCOL or payload.get("kind") != "release":
            raise ValueError("invalid v3 release frame")
        durations = payload.get("source_elapsed_ns")
        if not isinstance(durations, list):
            raise ValueError("source_elapsed_ns must be a list")
        frame = cls(
            _integer(payload, "event_id", 0),
            str(payload.get("request_id", "")),
            str(payload.get("mode", "")),
            _integer(payload, "completed_sources", 0),
            _integer(payload, "physical_descriptors", 0),
            _integer(payload, "completed_bytes", 0),
            tuple(_plain_integer(value, "source duration", 0) for value in durations),
            _integer(payload, "wave_elapsed_ns", 0),
            _integer(payload, "released_ns", 1),
        )
        frame.validate(event)
        reported = payload.get("service_gate_met")
        if not isinstance(reported, bool) or reported != frame.service_gate_met:
            raise ValueError("service_gate_met disagrees with durations")
        return frame


def _plain_integer(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _integer(payload: Mapping[str, Any], name: str, minimum: int) -> int:
    return _plain_integer(payload.get(name), name, minimum)


def _tmp_path(raw: str, *, prefix: str, suffix: str) -> Path:
    path = Path(raw)
    if path.parent != Path("/tmp") or not path.is_absolute():
        raise ValueError("gate paths must be immediate children of /tmp")
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError("gate path prefix/suffix changed")
    if len(os.fsencode(path)) > 100:
        raise ValueError("gate path is too long")
    return path


def config_from_environment(
    environ: Mapping[str, str] | None = None,
) -> GateConfig | None:
    values = os.environ if environ is None else environ
    enabled = values.get(ENABLE_ENV)
    if enabled is None:
        return None
    if enabled != "YES" or values.get(NODE_RANK_ENV) != "0":
        raise ValueError("quiescence v3 must be explicitly enabled on node rank 0")
    if values.get(TOKEN_INDEX_ENV, "30") != "30":
        raise ValueError("v3 is frozen to index30 / generated count31")
    try:
        timeout_s = float(values.get(TIMEOUT_ENV, "10"))
    except ValueError as exc:
        raise ValueError("gate timeout must be numeric") from exc
    if not 0.1 <= timeout_s <= 30:
        raise ValueError("gate timeout must be in 0.1..30 seconds")
    return GateConfig(
        socket_path=_tmp_path(
            values.get(SOCKET_ENV, ""),
            prefix="tempo-vllm-quiescence-",
            suffix=".sock",
        ),
        trace_path=_tmp_path(
            values.get(TRACE_ENV, ""), prefix="tempo-step-gate-", suffix=".jsonl"
        ),
        timeout_s=timeout_s,
    )


# Reuse the audited one-frame Unix-socket transport with v3 payload classes.
wire.ReadyEvent = ReadyEvent
wire.ReleaseFrame = ReleaseFrame
GateClient = wire.GateClient
GateConnection = wire.GateConnection
GateListener = wire.GateListener


class OutputBoundaryDetector:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._pending: list[ReadyEvent] = []
        self._next_event_id = 0

    def observe_queue_item(self, item: Any) -> None:
        if not isinstance(item, tuple) or len(item) != 2:
            return
        outputs = getattr(item[1], "outputs", ())
        if not isinstance(outputs, (list, tuple)):
            return
        for output in outputs:
            request_id = str(getattr(output, "request_id", ""))
            if REQUEST_ID_MARKER not in request_id:
                continue
            token_ids = getattr(output, "new_token_ids", ())
            if not isinstance(token_ids, (list, tuple)):
                raise RuntimeError("new_token_ids changed type")
            before = self._counts.get(request_id, 0)
            after = before + len(token_ids)
            if before <= TARGET_OUTPUT_TOKEN_INDEX < after:
                if before != TARGET_OUTPUT_TOKEN_INDEX or len(token_ids) != 1:
                    raise RuntimeError("token31 was crossed by multi-token decoding")
                self._pending.append(
                    ReadyEvent(
                        self._next_event_id,
                        request_id,
                        TARGET_OUTPUT_TOKEN_INDEX,
                        after,
                        time.perf_counter_ns(),
                    )
                )
                self._next_event_id += 1
            self._counts[request_id] = after
            if bool(getattr(output, "finished", False)):
                self._counts.pop(request_id, None)

    def pop_pending(self) -> ReadyEvent | None:
        if len(self._pending) > 1:
            raise RuntimeError("multiple scout requests reached token31 together")
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


def _tempo_accelerator_fence(worker: Any) -> tuple[int, int, int]:
    import torch

    started_ns = time.perf_counter_ns()
    torch.accelerator.synchronize()
    finished_ns = time.perf_counter_ns()
    return int(worker.global_rank), started_ns, finished_ns


def _write_trace(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    encoded = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    ) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC
    flags |= os.O_EXCL if exclusive else os.O_APPEND
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("short quiescence trace write")
            offset += written
    finally:
        os.close(descriptor)


def _validate_fence(rows: Any) -> list[tuple[int, int, int]]:
    if not isinstance(rows, list) or len(rows) != WORLD_SIZE:
        raise RuntimeError("accelerator fence did not return sixteen replies")
    normalized = [tuple(int(value) for value in row) for row in rows]
    if any(len(row) != 3 for row in normalized):
        raise RuntimeError("accelerator fence reply shape changed")
    if sorted(row[0] for row in normalized) != list(range(WORLD_SIZE)):
        raise RuntimeError("accelerator fence rank coverage changed")
    if any(row[1] <= 0 or row[2] < row[1] for row in normalized):
        raise RuntimeError("accelerator fence timeline changed")
    return normalized


def patch_engine_core_class(
    engine_core_class: type[Any],
    config: GateConfig,
    *,
    client_factory: Callable[[GateConfig], Any] = GateClient,
) -> bool:
    marker = "_tempo_output_quiescence_launch_v3_installed"
    if bool(getattr(engine_core_class, marker, False)):
        return False
    original_init = engine_core_class.__init__
    original_step = engine_core_class._process_engine_step

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if bool(getattr(self, "async_scheduling", False)):
            raise RuntimeError("quiescence v3 requires --no-async-scheduling")
        if getattr(self, "batch_queue", None) is not None or int(
            getattr(self, "batch_queue_size", -1)
        ) != 1:
            raise RuntimeError("quiescence v3 requires one batch in flight")
        cfg = self.vllm_config
        parallel = cfg.parallel_config
        if cfg.speculative_config is not None:
            raise RuntimeError("quiescence v3 forbids speculative decoding")
        if (
            int(parallel.world_size) != WORLD_SIZE
            or int(parallel.tensor_parallel_size) != WORLD_SIZE
            or int(parallel.pipeline_parallel_size) != 1
            or int(parallel.data_parallel_size) != 1
        ):
            raise RuntimeError("quiescence v3 requires DP1/PP1/TP16")
        detector = OutputBoundaryDetector()
        self.output_queue = _OutputQueueTap(self.output_queue, detector)
        self._tempo_gate_detector_v3 = detector
        self._tempo_gate_client_v3 = client_factory(config)
        self._tempo_gate_last_release_v3 = None
        _write_trace(
            config.trace_path,
            {
                "kind": "provenance",
                "protocol": PROTOCOL,
                "node_rank": 0,
                "world_size": WORLD_SIZE,
                "tensor_parallel_size": WORLD_SIZE,
                "async_scheduling": False,
                "speculative_decoding": False,
                "output_token_index_zero_based": TARGET_OUTPUT_TOKEN_INDEX,
                "generated_token_count_one_based": TARGET_GENERATED_TOKEN_COUNT,
                "request_id_marker": REQUEST_ID_MARKER,
            },
            exclusive=True,
        )

    def patched_step(self: Any) -> bool:
        entered_ns = time.perf_counter_ns()
        previous = self._tempo_gate_last_release_v3
        if previous is not None:
            if entered_ns < int(previous[1]):
                raise RuntimeError("next engine step entered before release")
            _write_trace(
                config.trace_path,
                {
                    "kind": "next_engine_step_enter",
                    "event_id": int(previous[0]),
                    "entered_ns": entered_ns,
                    "released_ns": int(previous[1]),
                },
            )
            self._tempo_gate_last_release_v3 = None
        executed = original_step(self)
        event = self._tempo_gate_detector_v3.pop_pending()
        if event is None:
            return executed
        counts = tuple(int(value) for value in self.scheduler.get_request_counts())
        if counts != (1, 0):
            raise RuntimeError("token31 gate requires one running and zero waiting requests")
        fence_started_ns = time.perf_counter_ns()
        rows = _validate_fence(
            self.model_executor.collective_rpc(
                _tempo_accelerator_fence, timeout=config.timeout_s
            )
        )
        fence_finished_ns = time.perf_counter_ns()
        event = replace(
            event,
            fence_started_ns=fence_started_ns,
            fence_finished_ns=fence_finished_ns,
            ready_ns=fence_finished_ns,
        )
        _write_trace(
            config.trace_path,
            {
                "kind": "ready",
                **event.to_payload(),
                "fence_rows": [list(row) for row in rows],
                "fence_elapsed_ns": fence_finished_ns - fence_started_ns,
            },
        )
        try:
            release = self._tempo_gate_client_v3.wait_release(event)
        except BaseException as exc:
            _write_trace(
                config.trace_path,
                {
                    "kind": "error",
                    "event_id": event.event_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "observed_ns": time.perf_counter_ns(),
                },
            )
            raise
        gate_returned_ns = time.perf_counter_ns()
        if gate_returned_ns < release.released_ns:
            raise RuntimeError("gate returned before declared release")
        _write_trace(
            config.trace_path,
            {
                "kind": "release",
                **release.to_payload(),
                "gate_returned_ns": gate_returned_ns,
            },
        )
        self._tempo_gate_last_release_v3 = (release.event_id, release.released_ns)
        return executed

    engine_core_class.__init__ = patched_init
    engine_core_class._process_engine_step = patched_step
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
    "GateClient",
    "GateConfig",
    "GateConnection",
    "GateListener",
    "PROTOCOL",
    "ReadyEvent",
    "ReleaseFrame",
    "config_from_environment",
    "install_from_environment",
    "patch_engine_core_class",
]
