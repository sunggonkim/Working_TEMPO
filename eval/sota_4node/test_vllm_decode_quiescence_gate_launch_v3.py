from __future__ import annotations

from queue import Queue
from types import SimpleNamespace
import os
from pathlib import Path
import time
import uuid

import pytest

from eval.sota_4node import vllm_decode_quiescence_gate_launch_v3 as hook


def _item(index: int):
    return (
        0,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    request_id="cmpl-tempo-scout-c0-b0-0-deadbeef",
                    new_token_ids=[index],
                    finished=False,
                )
            ]
        ),
    )


def test_detector_uses_one_based_count31_and_index30() -> None:
    detector = hook.OutputBoundaryDetector()
    for index in range(30):
        detector.observe_queue_item(_item(index))
        assert detector.pop_pending() is None
    detector.observe_queue_item(_item(30))
    event = detector.pop_pending()
    assert event is not None
    assert event.output_token_index == 30
    assert event.generated_token_count == 31


def test_slow_wave_releases_but_falsifies_service_gate() -> None:
    event = hook.ReadyEvent(
        0,
        "cmpl-tempo-scout-c0-b0-0-deadbeef",
        30,
        31,
        1,
        2,
        3,
        3,
    )
    frame = hook.ReleaseFrame.wave512k(
        event,
        source_elapsed_ns=(6_000_000,) * 8,
        wave_elapsed_ns=8_000_000,
    )
    frame.validate(event)
    assert frame.completed_bytes == 4 << 20
    assert not frame.service_gate_met
    decoded = hook.ReleaseFrame.from_payload(frame.to_payload(), event=event)
    assert decoded == frame


def test_engine_patch_fences_all16_and_persists_order() -> None:
    suffix = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    trace = Path(f"/tmp/tempo-step-gate-{suffix}.jsonl")
    config = hook.GateConfig(
        Path(f"/tmp/tempo-vllm-quiescence-{suffix}.sock"), trace, timeout_s=1
    )
    calls: list[str] = []

    class Client:
        def __init__(self, _config):
            pass

        def wait_release(self, event):
            calls.append("release")
            return hook.ReleaseFrame.noop(event)

    class Executor:
        def collective_rpc(self, function, timeout):
            calls.append("fence")
            now = time.perf_counter_ns()
            return [(rank, now, now + 1) for rank in range(16)]

    parallel = SimpleNamespace(
        world_size=16,
        tensor_parallel_size=16,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )

    class Engine:
        def __init__(self):
            self.async_scheduling = False
            self.batch_queue = None
            self.batch_queue_size = 1
            self.output_queue = Queue()
            self.vllm_config = SimpleNamespace(
                speculative_config=None, parallel_config=parallel
            )
            self.scheduler = SimpleNamespace(get_request_counts=lambda: (1, 0))
            self.model_executor = Executor()
            self.index = 0

        def _process_engine_step(self):
            self.output_queue.put_nowait(_item(self.index))
            self.index += 1
            calls.append("post")
            return True

    try:
        assert hook.patch_engine_core_class(Engine, config, client_factory=Client)
        engine = Engine()
        for _ in range(31):
            engine._process_engine_step()
        assert calls[-3:] == ["post", "fence", "release"]
        engine._process_engine_step()
        records = [json_line for json_line in trace.read_text().splitlines()]
        assert any('"kind":"ready"' in line for line in records)
        assert any('"kind":"release"' in line for line in records)
        assert any('"kind":"next_engine_step_enter"' in line for line in records)
    finally:
        trace.unlink(missing_ok=True)


def test_config_requires_node0_and_short_tmp_paths() -> None:
    values = {
        hook.ENABLE_ENV: "YES",
        hook.NODE_RANK_ENV: "0",
        hook.SOCKET_ENV: "/tmp/tempo-vllm-quiescence-test.sock",
        hook.TRACE_ENV: "/tmp/tempo-step-gate-test.jsonl",
        hook.TOKEN_INDEX_ENV: "30",
    }
    assert hook.config_from_environment(values) is not None
    with pytest.raises(ValueError, match="node rank 0"):
        hook.config_from_environment({**values, hook.NODE_RANK_ENV: "1"})
