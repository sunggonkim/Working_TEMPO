#!/usr/bin/env python3
"""Pair-local queue-pressure admission: one remote only after two local inflight."""

from __future__ import annotations

import threading
import time

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_admission import PDRequestPhase, PDRoute


class PressureCore(credit.CreditCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._local_owned: set[str] = set()
        self._remote_ever_used = False

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        del remaining_deadline_ms
        base._require(isinstance(request_id, str) and request_id.strip(),
                      "request_id must be nonempty")
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens
        )
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            base._require(len(self._records) < self.config.decision_capacity,
                          "decision capacity exhausted")
            if (len(self._local_owned) >= 2 and self._remote_owner is None
                    and not self._remote_ever_used):
                route = PDRoute.REMOTE_PREFILL
                reason = "remote_once_local_inflight_ge_2"
                self._remote_owner = request_id
                self._reserved.add(request_id)
                self._remote_ever_used = True
            else:
                route = PDRoute.DECODER_LOCAL
                reason = "local_below_pressure_or_remote_already_used"
                self._local_owned.add(request_id)
            record = base.RouterDecision(
                request_id=request_id,
                mode=base.RouterMode.TEMPO_AUTO,
                route=route,
                reason=reason,
                workload=workload,
                profile_id=None,
                manifest_id=None,
                policy_epoch=None,
                remote_advantage_lower_bound_ms=None,
                prompt_tokens=prompt_tokens,
                potential_kv_bytes=kv_bytes,
                decided_ns=time.perf_counter_ns(),
                phase=(PDRequestPhase.REMOTE_SELECTED.value
                       if route is PDRoute.REMOTE_PREFILL
                       else PDRequestPhase.LOCAL_SELECTED.value),
            )
            self._records[request_id] = record
        return record

    def _release(self, request_id: str) -> None:
        with self._lock:
            self._local_owned.discard(request_id)
            if request_id in self._reserved:
                base._require(self._remote_owner == request_id,
                              "remote credit owner mismatch")
                self._reserved.remove(request_id)
                self._remote_owner = None


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = PressureCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
