#!/usr/bin/env python3
"""Frontend entry point for the NetKV/Kairos paper-policy reproductions."""

from __future__ import annotations

import asyncio
import time

from eval.sota_4node import tempo_pd_elastic_frontend as frontend
from tempo.pd_paper_baselines import (
    PaperBaselineCoordinator,
    PaperBaselineOrchestrator,
)


class _NoTempoBusinessAdmissionGate:
    """Emit lifecycle evidence without throttling or priority protection.

    The frozen C8 workload validator requires a held/released decoder receipt
    for every global victim and local background request.  Returning ``None``
    removes both policy and evidence, which makes the workload unverifiable.
    This adapter preserves the receipt schema but never waits, caps, drains,
    or orders a request.  ``policy_effect`` makes that distinction explicit.
    """

    def __init__(
        self, *, background_limits, background_max_wait_ns,
        protected_tenants,
    ) -> None:
        self.background_limits = tuple(int(value) for value in background_limits)
        self.background_max_wait_ns = int(background_max_wait_ns)
        self.protected_tenants = frozenset(protected_tenants)
        self._leases: dict[str, dict[str, object]] = {}
        self._lock = asyncio.Lock()
        self._admitted = 0

    async def acquire(
        self, *, request_id, pair_index, tenant_id, globally_committed,
    ):
        foreground = bool(
            globally_committed and tenant_id in self.protected_tenants)
        background = bool(
            isinstance(tenant_id, str)
            and tenant_id not in self.protected_tenants)
        if not (foreground or background):
            return None
        admitted_ns = time.perf_counter_ns()
        receipt = {
            "schema": frontend.DECODER_BUSINESS_ADMISSION_SCHEMA,
            "request_id": request_id,
            "tenant_id": tenant_id,
            "pair_index": pair_index,
            "admission_class": "protected" if foreground else "background",
            "status": "held",
            "arrived_ns": admitted_ns,
            "admitted_ns": admitted_ns,
            "wait_ns": 0,
            "background_limit": self.background_limits[pair_index],
            "background_max_wait_ns": self.background_max_wait_ns,
            "starvation_escape": False,
            "foreground_active_before": 0,
            "background_active_before": 0,
            "foreground_active_after": 0,
            "background_active_after": 0,
            "released_ns": None,
            "mode": "evidence_only_no_throttle",
            "policy_effect": "none",
        }
        async with self._lock:
            if request_id in self._leases:
                raise ValueError("duplicate paper-baseline evidence lease")
            self._leases[request_id] = dict(receipt)
            self._admitted += 1
        return receipt

    async def release(self, request_id):
        async with self._lock:
            receipt = self._leases.pop(request_id, None)
        if receipt is None:
            raise ValueError("paper-baseline evidence lease is absent")
        return {
            **receipt,
            "status": "released",
            "released_ns": time.perf_counter_ns(),
            "foreground_active_after_release": 0,
            "background_active_after_release": 0,
        }

    async def snapshot(self):
        async with self._lock:
            return {
                "schema": "tempo-paper-baseline-no-business-gate-v1",
                "mode": "evidence_only_no_throttle",
                "policy_effect": "none",
                "background_limits_not_enforced": list(
                    self.background_limits),
                "background_max_wait_ns_not_enforced": (
                    self.background_max_wait_ns),
                "leases": len(self._leases),
                "receipts_admitted": self._admitted,
            }


# ``build_app`` resolves these module globals at runtime.  Patch only this new
# process; the canonical C9 module and its source hash remain untouched.
frontend.GlobalOrchestrator = PaperBaselineOrchestrator
frontend.GlobalAdmissionCoordinator = PaperBaselineCoordinator
frontend.DecoderBusinessAdmissionGate = _NoTempoBusinessAdmissionGate

build_app = frontend.build_app
main = frontend.main


if __name__ == "__main__":
    raise SystemExit(main())
