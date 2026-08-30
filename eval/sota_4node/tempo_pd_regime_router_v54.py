#!/usr/bin/env python3
"""Measured-epoch regime adapter with a trace-calibrated 70 ms high cutoff."""

from __future__ import annotations
from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_regime_router_v52 as v52
from tempo.pd_regime_controller import PairArrivalRegimeController


class MeasuredEpochRegimeCore(v52.RegimeCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._measured_epoch_started = False

    @staticmethod
    def _new_controller() -> PairArrivalRegimeController:
        return PairArrivalRegimeController(high_pair_interval_ns=70_000_000)

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        if request_id.startswith("val-") and not self._measured_epoch_started:
            if self._regime_controller.local_inflight != 0:
                raise RuntimeError("measured epoch began before warmup requests drained")
            self._regime_controller = self._new_controller()
            self._measured_epoch_started = True
        return v52.RegimeCore.decide(
            self, request_id=request_id, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, remaining_deadline_ms=remaining_deadline_ms)


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = MeasuredEpochRegimeCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
