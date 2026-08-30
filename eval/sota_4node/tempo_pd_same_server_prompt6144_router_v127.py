#!/usr/bin/env python3
"""Diagnostic local routing for validated outputs at prompt lengths <=6144."""

from __future__ import annotations

import time

from eval.sota_4node import tempo_pd_router_v1 as base
from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node.tempo_pd_same_server_production_interleaved_router_v108 import (
    ProductionInterleavedCore,
)
from tempo.pd_admission import PDRequestPhase, PDRoute


class Prompt6144Core(ProductionInterleavedCore):
    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        arm, phase_name = self._arm(request_id)
        if arm != "tempo" or output_tokens not in (16, 128):
            return super().decide(
                request_id=request_id, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_deadline_ms=remaining_deadline_ms)
        del remaining_deadline_ms
        if not 4096 < prompt_tokens <= 6144:
            raise ValueError("prompt6144 diagnostic requires 4096 < prompt_tokens <= 6144")
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        now_ns = time.perf_counter_ns()
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            record = base.RouterDecision(
                request_id=request_id, mode=base.RouterMode.TEMPO_AUTO,
                route=PDRoute.DECODER_LOCAL,
                reason=(f"same_server_tempo_{phase_name}:"
                        f"output{output_tokens}_prompt6144_local_diagnostic"),
                workload=workload, profile_id="prompt6144-local-v127",
                manifest_id="prompt6144-local-v127", policy_epoch=0,
                remote_advantage_lower_bound_ms=None,
                prompt_tokens=prompt_tokens, potential_kv_bytes=kv_bytes,
                decided_ns=now_ns, phase=PDRequestPhase.LOCAL_SELECTED.value,
            )
            self._records[request_id] = record
            return record


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = Prompt6144Core
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
