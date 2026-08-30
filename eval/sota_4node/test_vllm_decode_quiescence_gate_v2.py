from __future__ import annotations

import os
from pathlib import Path
import threading
import uuid

import pytest

from eval.sota_4node import vllm_decode_quiescence_gate_v2 as hook


def _event() -> hook.ReadyEvent:
    return hook.ReadyEvent(
        event_id=3,
        request_id="request-3",
        output_token_index=31,
        output_token_count=32,
        ready_ns=1,
    )


def test_hook_is_explicitly_node_zero_only() -> None:
    values = {
        "TEMPO_VLLM_QUIESCENCE_ENABLED": "YES",
        "TEMPO_VLLM_QUIESCENCE_SOCKET": (
            "/tmp/tempo-vllm-quiescence-v2-config.sock"
        ),
        "TEMPO_VLLM_QUIESCENCE_TOKEN_INDEX": "31",
        hook.NODE_RANK_ENV: "0",
    }
    assert hook.config_from_environment(values) is not None
    with pytest.raises(ValueError, match="node-zero-only"):
        hook.config_from_environment({**values, hook.NODE_RANK_ENV: "1"})


def test_slow_exact_wave_is_valid_but_falsifies_service_gate() -> None:
    event = _event()
    slow_call = hook.ReleaseFrame.wave512k(
        event,
        source_elapsed_ns=(4_000_001,) + (1_000_000,) * 7,
        wave_elapsed_ns=2_000_000,
    )
    slow_call.validate(event)
    assert not slow_call.service_gate_met
    slow_wave = hook.ReleaseFrame.wave512k(
        event,
        source_elapsed_ns=(1_000_000,) * 8,
        wave_elapsed_ns=5_000_001,
    )
    slow_wave.validate(event)
    assert not slow_wave.service_gate_met


def test_slow_wave_round_trip_still_releases_client() -> None:
    socket_path = Path(
        f"/tmp/tempo-vllm-quiescence-v2-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    )
    config = hook.GateConfig(socket_path=socket_path, timeout_s=2.0)
    event = _event()
    errors: list[BaseException] = []
    with hook.GateListener(config) as listener:
        def server() -> None:
            try:
                with listener.accept() as connection:
                    connection.release(
                        hook.ReleaseFrame.wave512k(
                            connection.event,
                            source_elapsed_ns=(6_000_000,) * 8,
                            wave_elapsed_ns=8_000_000,
                        )
                    )
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=server)
        thread.start()
        release = hook.GateClient(config).wait_release(event)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert not errors
        assert release.completed_bytes == 4 << 20
        assert release.physical_descriptors == 8
        assert not release.service_gate_met
    assert not socket_path.exists()
