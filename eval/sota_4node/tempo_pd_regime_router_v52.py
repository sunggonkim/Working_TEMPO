#!/usr/bin/env python3
"""Actual P/D router adapter for the production arrival-regime controller."""

from __future__ import annotations
import time
from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_admission import PDRequestPhase, PDRoute
from tempo.pd_regime_controller import AdmissionRoute, PairArrivalRegimeController


class RegimeCore(credit.CreditCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._regime_controller = PairArrivalRegimeController()

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        del remaining_deadline_ms
        base._require(isinstance(request_id, str) and request_id.strip(),
                      "request_id must be nonempty")
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        now_ns = time.perf_counter_ns()
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            admission = self._regime_controller.decide(request_id, now_ns)
            route = (PDRoute.REMOTE_PREFILL
                     if admission.route is AdmissionRoute.REMOTE
                     else PDRoute.DECODER_LOCAL)
            record = base.RouterDecision(
                request_id=request_id, mode=base.RouterMode.TEMPO_AUTO,
                route=route,
                reason=f"{admission.reason}:mean_pair_interval_ns={admission.observed_mean_pair_interval_ns}",
                workload=workload, profile_id="pair-arrival-regime-controller-v52",
                manifest_id="pair-arrival-regime-controller-v52", policy_epoch=0,
                remote_advantage_lower_bound_ms=(37.0 if route is PDRoute.REMOTE_PREFILL else None),
                prompt_tokens=prompt_tokens, potential_kv_bytes=kv_bytes,
                decided_ns=now_ns,
                phase=(PDRequestPhase.REMOTE_SELECTED.value
                       if route is PDRoute.REMOTE_PREFILL
                       else PDRequestPhase.LOCAL_SELECTED.value),
            )
            self._records[request_id] = record
            return record

    def _release(self, request_id: str) -> None:
        with self._lock:
            self._regime_controller.release(request_id)


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = RegimeCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
