#!/usr/bin/env python3
"""Router for monotone local-bypass opportunity detection."""

from dataclasses import replace
import time

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as router_base
from eval.sota_4node import tempo_pd_same_server_hybrid_phase_router_v181 as phase
from tempo.pd_admission import PDRequestPhase, PDRoute
from tempo.pd_online_regime_latched_v380 import (
    LatchedOnlineRegimeClassifier, OnlineRegime,
)


POLICY_ID = "tempo-pd-online-regime-latched-router-381"


class LatchedOnlineRegimeCore(phase.FullPhaseHybridCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._online_regime = LatchedOnlineRegimeClassifier()

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        arm, phase_name = self._arm(request_id)
        snapshot = (self._online_regime.observe(time.perf_counter_ns())
                    if phase_name == "measured" else None)
        record = super().decide(
            request_id=request_id, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            remaining_deadline_ms=remaining_deadline_ms,
        )
        if arm != "tempo" or phase_name != "measured":
            return record
        assert snapshot is not None
        median = ("none" if snapshot.median_gap_ns is None
                  else str(snapshot.median_gap_ns))
        high = snapshot.regime is OnlineRegime.HIGH_LOAD_LOCAL_BYPASS
        annotated = replace(
            record,
            route=PDRoute.DECODER_LOCAL if high else record.route,
            reason=(f"{record.reason}:online_regime={snapshot.regime.value}:"
                    f"observations={snapshot.observations}:"
                    f"median_gap_ns={median}:threshold_ns={snapshot.threshold_ns}:"
                    f"raw_regime={snapshot.raw_regime.value}:"
                    f"high_load_latched={str(snapshot.high_load_latched).lower()}"),
            profile_id=POLICY_ID,
            manifest_id=POLICY_ID,
            remote_advantage_lower_bound_ms=(
                None if high else record.remote_advantage_lower_bound_ms
            ),
            phase=(PDRequestPhase.LOCAL_SELECTED.value if high else record.phase),
        )
        with self._lock:
            router_base._require(self._records.get(request_id) == record,
                                 "latched regime record changed concurrently")
            self._records[request_id] = annotated
        return annotated


def main(argv=None):
    original = credit.CreditCore
    credit.CreditCore = LatchedOnlineRegimeCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
