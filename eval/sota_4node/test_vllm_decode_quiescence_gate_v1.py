from __future__ import annotations

from queue import Queue
from types import SimpleNamespace
import os
from pathlib import Path
import threading
import uuid

import pytest

from eval.sota_4node import vllm_decode_quiescence_gate_v1 as hook


def _event(event_id: int = 0) -> hook.ReadyEvent:
    return hook.ReadyEvent(
        event_id=event_id,
        request_id="request-0",
        output_token_index=31,
        output_token_count=32,
        ready_ns=1,
    )


def _engine_item(request_id: str, *, finished: bool = False):
    output = SimpleNamespace(
        request_id=request_id,
        new_token_ids=[17],
        finished=finished,
    )
    return (0, SimpleNamespace(outputs=[output]))


def test_environment_is_explicit_and_frozen() -> None:
    assert hook.config_from_environment({}) is None
    values = {
        hook.ENABLE_ENV: "YES",
        hook.SOCKET_ENV: "/tmp/tempo-vllm-quiescence-test.sock",
        hook.TOKEN_INDEX_ENV: "31",
        hook.TIMEOUT_ENV: "2.5",
    }
    config = hook.config_from_environment(values)
    assert config is not None
    assert config.token_index == 31
    assert config.timeout_s == 2.5
    with pytest.raises(ValueError, match="frozen"):
        hook.config_from_environment({**values, hook.TOKEN_INDEX_ENV: "30"})
    with pytest.raises(ValueError, match="immediate child"):
        hook.config_from_environment(
            {**values, hook.SOCKET_ENV: "/tmp/nested/tempo-vllm-quiescence-x.sock"}
        )


def test_detector_fires_once_after_zero_based_token_31_is_enqueued() -> None:
    detector = hook.OutputBoundaryDetector()
    for _ in range(31):
        detector.observe_queue_item(_engine_item("request-0"))
        assert detector.pop_pending() is None
    detector.observe_queue_item(_engine_item("request-0"))
    event = detector.pop_pending()
    assert event is not None
    assert event.output_token_index == 31
    assert event.output_token_count == 32
    detector.observe_queue_item(_engine_item("request-0", finished=True))
    assert detector.pop_pending() is None


def test_detector_rejects_multi_token_boundary_crossing() -> None:
    detector = hook.OutputBoundaryDetector()
    for _ in range(31):
        detector.observe_queue_item(_engine_item("request-0"))
    item = (
        0,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    request_id="request-0",
                    new_token_ids=[1, 2],
                    finished=False,
                )
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="multi-token"):
        detector.observe_queue_item(item)


def test_release_geometry_and_latency_gates() -> None:
    event = _event()
    noop = hook.ReleaseFrame.noop(event)
    noop.validate(event)
    wave = hook.ReleaseFrame.wave512k(
        event,
        source_elapsed_ns=(1_000_000,) * 8,
        wave_elapsed_ns=2_000_000,
    )
    wave.validate(event)
    assert wave.physical_descriptors == 8
    assert wave.completed_bytes == 4 << 20
    with pytest.raises(ValueError, match="4 ms"):
        hook.ReleaseFrame.wave512k(
            event,
            source_elapsed_ns=(4_000_001,) + (1_000_000,) * 7,
            wave_elapsed_ns=2_000_000,
        ).validate(event)
    with pytest.raises(ValueError, match="5 ms"):
        hook.ReleaseFrame.wave512k(
            event,
            source_elapsed_ns=(1_000_000,) * 8,
            wave_elapsed_ns=5_000_001,
        ).validate(event)


def test_synchronous_unix_gate_round_trip() -> None:
    socket_path = Path(
        f"/tmp/tempo-vllm-quiescence-test-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    )
    config = hook.GateConfig(socket_path=socket_path, timeout_s=2.0)
    event = _event(7)
    server_error: list[BaseException] = []
    with hook.GateListener(config) as listener:
        def server() -> None:
            try:
                with listener.accept() as connection:
                    assert connection.event == event
                    connection.release(hook.ReleaseFrame.noop(connection.event))
            except BaseException as exc:
                server_error.append(exc)

        thread = threading.Thread(target=server)
        thread.start()
        release = hook.GateClient(config).wait_release(event)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert not server_error
        assert release.mode == "quiescent_noop"
    assert not socket_path.exists()


def test_engine_patch_enqueues_then_waits_and_blocks_async_mode() -> None:
    order: list[str] = []

    class FakeClient:
        def __init__(self, _config: hook.GateConfig) -> None:
            pass

        def wait_release(self, event: hook.ReadyEvent) -> hook.ReleaseFrame:
            order.append("gate")
            return hook.ReleaseFrame.noop(event)

    class FakeEngine:
        def __init__(self, *, async_scheduling: bool = False) -> None:
            self.async_scheduling = async_scheduling
            self.batch_queue = [] if async_scheduling else None
            self.batch_queue_size = 2 if async_scheduling else 1
            self.output_queue = Queue()
            self.calls = 0

        def _process_engine_step(self) -> bool:
            self.calls += 1
            self.output_queue.put_nowait(_engine_item("request-0"))
            order.append("post")
            return True

    config = hook.GateConfig(
        Path("/tmp/tempo-vllm-quiescence-patch-test.sock"), timeout_s=1.0
    )
    assert hook.patch_engine_core_class(
        FakeEngine, config, client_factory=FakeClient
    )
    engine = FakeEngine()
    for _ in range(32):
        assert engine._process_engine_step()
    assert order[-2:] == ["post", "gate"]
    assert len(engine._tempo_quiescence_trace_v1) == 1
    engine._process_engine_step()
    trace = engine._tempo_quiescence_trace_v1[0]
    assert trace["next_engine_step_enter_ns"] >= trace["released_ns"]
    with pytest.raises(RuntimeError, match="--no-async-scheduling"):
        FakeEngine(async_scheduling=True)
