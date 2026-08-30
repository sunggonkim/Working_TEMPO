#!/usr/bin/env python3
"""Adapter from the production hybrid controller to the live HTTP router."""

from __future__ import annotations

import time

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as base
from eval.sota_4node import tempo_pd_same_server_balanced_router_v70 as balanced
from tempo.pd_admission import PDRequestPhase, PDRoute
from tempo.pd_hybrid_controller import CachePhase, HybridPDController


def _cache_item(request_id: str) -> str:
    marker = "-cache-item-"
    if marker not in request_id:
        raise ValueError("stable cache item identity missing")
    suffix = request_id.rsplit(marker, 1)[1]
    if len(suffix) != 2 or not suffix.isdigit():
        raise ValueError("stable cache item identity malformed")
    return f"cache-item-{suffix}"


class ProductionHybridCore(balanced.BalancedSameServerCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._hybrid = HybridPDController()
        self._hybrid_owned: set[str] = set()

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        del remaining_deadline_ms
        arm, phase_name = self._arm(request_id)
        if arm != "tempo":
            return super().decide(
                request_id=request_id, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens)
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        now_ns = time.perf_counter_ns()
        cache_phase = (CachePhase.WARM_SEED if phase_name == "warm"
                       else CachePhase.WARM_HIT)
        decision = self._hybrid.decide(
            request_id=request_id, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, now_ns=now_ns,
            cache_phase=cache_phase, cache_item=_cache_item(request_id))
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            record = base.RouterDecision(
                request_id=request_id, mode=base.RouterMode.TEMPO_AUTO,
                route=decision.route,
                reason=f"same_server_tempo_{phase_name}:{decision.reason}",
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

    def _release(self, request_id: str) -> None:
        with self._lock:
            owned = request_id in self._hybrid_owned
            if owned:
                self._hybrid_owned.remove(request_id)
        if owned:
            self._hybrid.complete(request_id)
        else:
            super()._release(request_id)


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = ProductionHybridCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__": raise SystemExit(main())
