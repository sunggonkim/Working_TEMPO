#!/usr/bin/env python3
"""Recursion-safe process entrypoint for the hybrid v6 implementation."""

from __future__ import annotations

from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed


_ORIGINAL_VALIDATE_TRACE = old._validate_trace
_ORIGINAL_AGGREGATE = old._aggregate
_V6_VALIDATE_TRACE = fixed._validate_trace
_V6_AGGREGATE = fixed._aggregate


def _validate_trace(*args: Any, **kwargs: Any) -> dict[str, Any]:
    installed = old._validate_trace
    old._validate_trace = _ORIGINAL_VALIDATE_TRACE
    try:
        return _V6_VALIDATE_TRACE(*args, **kwargs)
    finally:
        old._validate_trace = installed


def _aggregate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    installed = old._aggregate
    old._aggregate = _ORIGINAL_AGGREGATE
    try:
        return _V6_AGGREGATE(*args, **kwargs)
    finally:
        old._aggregate = installed


def main() -> None:
    fixed._validate_trace = _validate_trace
    fixed._aggregate = _aggregate
    fixed.main()


if __name__ == "__main__":
    main()
