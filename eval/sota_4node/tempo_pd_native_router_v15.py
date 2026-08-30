#!/usr/bin/env python3
"""Fixed-remote router with truthful native-Nixl route provenance."""

from __future__ import annotations

from dataclasses import replace

from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_admission import PDRoute


class NativeCore(base.TempoPDRouterCore):
    def decide(self, **kwargs):
        record = super().decide(**kwargs)
        if record.route is PDRoute.REMOTE_PREFILL:
            record = replace(record, reason="fixed_native_vllm_nixl_remote_candidate")
            with self._lock:
                self._records[record.request_id] = record
        return record


def main() -> int:
    original = base.TempoPDRouterCore
    base.TempoPDRouterCore = NativeCore
    try:
        return base.main()
    finally:
        base.TempoPDRouterCore = original


if __name__ == "__main__":
    raise SystemExit(main())
