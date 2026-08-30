from __future__ import annotations

import pytest

from eval.sota_4node import (
    vllm_decode_quiescence_gate_launch_v3_hardening as hardening,
)


def test_exact_installed_vllm_and_process_step_source_are_pinned() -> None:
    from vllm.v1.engine.core import EngineCoreProc

    assert hardening.validate_engine_core_compatibility(EngineCoreProc) == {
        "vllm_version": "0.26.0+cu129",
        "engine_core_process_step_sha256": (
            "41295db73bb85ebda9cee7c4f32d944e5f973b6bcc0433ff6b152a9368b175b9"
        ),
    }


def test_mismatched_source_fails_before_patch(monkeypatch) -> None:
    class WrongEngine:
        def _process_engine_step(self):
            return False

    monkeypatch.setattr(
        hardening.importlib.metadata,
        "version",
        lambda _name: "0.26.0+cu129",
    )
    with pytest.raises(RuntimeError, match="source hash changed"):
        hardening.validate_engine_core_compatibility(WrongEngine)


def test_wrong_vllm_version_fails_before_patch(monkeypatch) -> None:
    from vllm.v1.engine.core import EngineCoreProc

    monkeypatch.setattr(
        hardening.importlib.metadata, "version", lambda _name: "0.26.1"
    )
    with pytest.raises(
        RuntimeError, match=r"requires exact vLLM 0\.26\.0\+cu129"
    ):
        hardening.validate_engine_core_compatibility(EngineCoreProc)


def test_identity_is_added_only_to_provenance() -> None:
    identity = {
        "vllm_version": hardening.EXPECTED_VLLM_VERSION,
        "engine_core_process_step_sha256": (
            hardening.EXPECTED_PROCESS_STEP_SHA256
        ),
    }
    provenance = hardening.provenance_with_identity(
        {"kind": "provenance", "protocol": "test"}, identity
    )
    assert provenance["vllm_version"] == "0.26.0+cu129"
    assert (
        provenance["engine_core_process_step_sha256"]
        == hardening.EXPECTED_PROCESS_STEP_SHA256
    )
    assert hardening.provenance_with_identity(
        {"kind": "ready", "event_id": 0}, identity
    ) == {"kind": "ready", "event_id": 0}
