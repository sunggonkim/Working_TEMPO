#!/usr/bin/env python3
"""Keep TEMPO's decoder APC control off the P/D producer request."""

from __future__ import annotations

from typing import Any


PROXY_DECODER_CONTROL_FIELD = (
    "tempo_decoder_skip_local_prefix_cache_read"
)
VLLM_XARGS_FIELD = "vllm_xargs"
VLLM_SKIP_LOCAL_PREFIX_READ_XARG = (
    "tempo_skip_local_prefix_cache_read"
)


def extract_decoder_cache_read_control(payload: dict[str, Any]) -> int | None:
    """Consume proxy-only control before forwarding a request to producer P."""

    if not isinstance(payload, dict):
        raise ValueError("P/D proxy request payload must be an object")
    raw_xargs = payload.get(VLLM_XARGS_FIELD)
    if raw_xargs is not None and not isinstance(raw_xargs, dict):
        raise ValueError("vllm_xargs must be an object")
    if (
        isinstance(raw_xargs, dict)
        and VLLM_SKIP_LOCAL_PREFIX_READ_XARG in raw_xargs
    ):
        raise ValueError(
            "decoder APC control must use the proxy-only request field"
        )
    if PROXY_DECODER_CONTROL_FIELD not in payload:
        return None
    raw = payload.pop(PROXY_DECODER_CONTROL_FIELD)
    if type(raw) is not int or raw not in (0, 1):
        raise ValueError(
            f"{PROXY_DECODER_CONTROL_FIELD} must be the integer 0 or 1"
        )
    return raw


def apply_decoder_cache_read_control(
    payload: dict[str, Any], control: int | None,
) -> dict[str, Any]:
    """Inject the consumed control only into the downstream decoder request."""

    if not isinstance(payload, dict):
        raise ValueError("P/D proxy request payload must be an object")
    if PROXY_DECODER_CONTROL_FIELD in payload:
        raise ValueError("proxy-only decoder APC control was not consumed")
    if control is None:
        return payload
    if type(control) is not int or control not in (0, 1):
        raise ValueError("decoder APC control must be the integer 0 or 1")
    raw_xargs = payload.get(VLLM_XARGS_FIELD)
    if raw_xargs is not None and not isinstance(raw_xargs, dict):
        raise ValueError("vllm_xargs must be an object")
    xargs = dict(raw_xargs or {})
    if VLLM_SKIP_LOCAL_PREFIX_READ_XARG in xargs:
        raise ValueError("decoder APC control xarg already exists")
    xargs[VLLM_SKIP_LOCAL_PREFIX_READ_XARG] = control
    payload[VLLM_XARGS_FIELD] = xargs
    return payload
