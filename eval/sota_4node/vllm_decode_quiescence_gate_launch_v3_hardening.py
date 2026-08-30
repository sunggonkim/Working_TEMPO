#!/usr/bin/env python3
"""Fail-closed compatibility pin for the launchable quiescence v3 hook.

This add-only wrapper exists because the workspace patch helper cannot update
the untracked launchable v3 module in place. It preserves that module's gate
API, validates the exact installed vLLM distribution before patching, and
injects the validated identity into the existing provenance record.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from typing import Any, Callable, Mapping

from eval.sota_4node import vllm_decode_quiescence_gate_launch_v3 as hook


EXPECTED_VLLM_VERSION = "0.26.0+cu129"
EXPECTED_PROCESS_STEP_SHA256 = (
    "41295db73bb85ebda9cee7c4f32d944e5f973b6bcc0433ff6b152a9368b175b9"
)


def validate_engine_core_compatibility(
    engine_core_class: type[Any],
) -> dict[str, str]:
    """Return audited identity or raise before any method replacement."""

    installed_version = importlib.metadata.version("vllm")
    try:
        source = inspect.getsource(engine_core_class._process_engine_step)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "cannot inspect EngineCoreProc._process_engine_step before patch"
        ) from exc
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if installed_version != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            "quiescence v3 requires exact vLLM "
            f"{EXPECTED_VLLM_VERSION}, found {installed_version}"
        )
    if source_sha256 != EXPECTED_PROCESS_STEP_SHA256:
        raise RuntimeError(
            "EngineCoreProc._process_engine_step source hash changed: "
            f"{source_sha256}"
        )
    return {
        "vllm_version": installed_version,
        "engine_core_process_step_sha256": source_sha256,
    }


def provenance_with_identity(
    payload: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, Any]:
    result = dict(payload)
    if result.get("kind") == "provenance":
        result.update(identity)
    return result


def install_pinned_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    identity_validator: Callable[[type[Any]], dict[str, str]] = (
        validate_engine_core_compatibility
    ),
) -> bool:
    config = hook.config_from_environment(environ)
    if config is None:
        return False
    from vllm.v1.engine.core import EngineCoreProc

    identity = identity_validator(EngineCoreProc)
    original_writer = hook._write_trace

    def pinned_writer(path, payload, *, exclusive=False):
        original_writer(
            path,
            provenance_with_identity(payload, identity),
            exclusive=exclusive,
        )

    hook._write_trace = pinned_writer
    try:
        installed = hook.patch_engine_core_class(EngineCoreProc, config)
    except BaseException:
        hook._write_trace = original_writer
        raise
    if not installed:
        hook._write_trace = original_writer
        raise RuntimeError(
            "quiescence v3 was already patched before compatibility validation"
        )
    return True


__all__ = [
    "EXPECTED_PROCESS_STEP_SHA256",
    "EXPECTED_VLLM_VERSION",
    "install_pinned_from_environment",
    "provenance_with_identity",
    "validate_engine_core_compatibility",
]
