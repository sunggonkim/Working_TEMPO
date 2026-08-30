#!/usr/bin/env python3
"""Measured regime controller with two calibration intervals per pair."""

from __future__ import annotations
from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_regime_router_v54 as v54
from tempo.pd_regime_controller import PairArrivalRegimeController


class FastCalibrationRegimeCore(v54.MeasuredEpochRegimeCore):
    @staticmethod
    def _new_controller() -> PairArrivalRegimeController:
        return PairArrivalRegimeController(
            high_pair_interval_ns=70_000_000,
            calibration_requests=3,
        )


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = FastCalibrationRegimeCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
