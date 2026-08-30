#!/usr/bin/env python3
"""Online arrival-regime controller: local, remote-dominant, or threshold-eight mix."""

from __future__ import annotations
from collections import deque
import time
from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_admission import PDRequestPhase, PDRoute

HIGH_PAIR_INTERVAL_NS = 74_000_000
MID_PAIR_INTERVAL_NS = 110_000_000
CALIBRATION_REQUESTS = 4
HIGH_LOCAL_INFLIGHT_CROSSOVER = 8


class RegimeCore(credit.CreditCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._local_owned: set[str] = set()
        self._remote_owned: set[str] = set()
        self._last_arrival_ns: int | None = None
        self._arrival_intervals_ns: deque[int] = deque(maxlen=CALIBRATION_REQUESTS - 1)
        self._arrival_regime: str | None = None

    def _observe_arrival(self, now_ns: int) -> str:
        if self._last_arrival_ns is not None:
            interval = now_ns - self._last_arrival_ns
            base._require(interval > 0, "arrival clock must advance")
            self._arrival_intervals_ns.append(interval)
        self._last_arrival_ns = now_ns
        if self._arrival_regime is None and len(self._arrival_intervals_ns) == CALIBRATION_REQUESTS - 1:
            mean_ns = sum(self._arrival_intervals_ns) / len(self._arrival_intervals_ns)
            if mean_ns <= HIGH_PAIR_INTERVAL_NS:
                self._arrival_regime = "high"
            elif mean_ns <= MID_PAIR_INTERVAL_NS:
                self._arrival_regime = "mid"
            else:
                self._arrival_regime = "low"
        return self._arrival_regime or "calibrating"

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
            regime = self._observe_arrival(now_ns)
            if regime == "mid":
                route = PDRoute.REMOTE_PREFILL
                reason = "remote_mid_arrival_regime"
            elif regime == "high" and len(self._local_owned) >= HIGH_LOCAL_INFLIGHT_CROSSOVER:
                route = PDRoute.REMOTE_PREFILL
                reason = "remote_high_regime_local_inflight_ge_8"
            else:
                route = PDRoute.DECODER_LOCAL
                reason = f"local_{regime}_arrival_regime"
            if route is PDRoute.REMOTE_PREFILL:
                self._remote_owned.add(request_id)
            else:
                self._local_owned.add(request_id)
            record = base.RouterDecision(
                request_id=request_id, mode=base.RouterMode.TEMPO_AUTO,
                route=route, reason=reason, workload=workload,
                profile_id="online-arrival-regime-v50",
                manifest_id="online-arrival-regime-v50", policy_epoch=0,
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
            self._local_owned.discard(request_id)
            self._remote_owned.discard(request_id)


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = RegimeCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
