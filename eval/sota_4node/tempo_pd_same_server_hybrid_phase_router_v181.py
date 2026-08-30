#!/usr/bin/env python3
"""Live HybridPDController adapter covering MISS, WARM_SEED, and WARM_HIT."""

from __future__ import annotations

import time

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as router_base
from eval.sota_4node import tempo_pd_same_server_hybrid_controller_router_v150 as warm
from tempo.pd_admission import PDRequestPhase, PDRoute
from tempo.pd_hybrid_controller import CachePhase


class FullPhaseHybridCore(warm.ProductionHybridCore):
    @staticmethod
    def _arm(request_id: str) -> tuple[str, str]:
        if request_id.startswith("ssb-tempo-r0-cold-"):
            return "tempo", "cold"
        return warm.ProductionHybridCore._arm(request_id)

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        arm, phase_name = self._arm(request_id)
        if arm != "tempo" or phase_name != "cold":
            return super().decide(
                request_id=request_id, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_deadline_ms=remaining_deadline_ms)
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        now_ns = time.perf_counter_ns()
        decision = self._hybrid.decide(
            request_id=request_id, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, now_ns=now_ns,
            cache_phase=CachePhase.MISS)
        with self._lock:
            router_base._require(request_id not in self._records, "duplicate request_id")
            record = router_base.RouterDecision(
                request_id=request_id, mode=router_base.RouterMode.TEMPO_AUTO,
                route=decision.route,
                reason=f"same_server_tempo_cold:hybrid_cold:{decision.reason}",
                workload=workload, profile_id=decision.policy_id,
                manifest_id=decision.policy_id, policy_epoch=0,
                remote_advantage_lower_bound_ms=(
                    0.0 if decision.route is PDRoute.REMOTE_PREFILL else None),
                prompt_tokens=prompt_tokens, potential_kv_bytes=kv_bytes,
                decided_ns=now_ns,
                phase=(PDRequestPhase.REMOTE_SELECTED.value
                       if decision.route is PDRoute.REMOTE_PREFILL
                       else PDRequestPhase.LOCAL_SELECTED.value))
            self._records[request_id] = record
            self._hybrid_owned.add(request_id)
            return record


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = FullPhaseHybridCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
