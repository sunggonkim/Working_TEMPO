#!/usr/bin/env python3
"""Production regime adapter with an explicit measured admission epoch."""

from __future__ import annotations
from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_regime_router_v52 as v52
from tempo.pd_regime_controller import PairArrivalRegimeController


class MeasuredEpochRegimeCore(v52.RegimeCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._measured_epoch_started = False

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        if request_id.startswith("val-") and not self._measured_epoch_started:
            if self._regime_controller.local_inflight != 0:
                raise RuntimeError("measured epoch began before warmup requests drained")
            self._regime_controller = PairArrivalRegimeController()
            self._measured_epoch_started = True
        return super().decide(
            request_id=request_id, prompt_tokens=prompt_tokens,
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
