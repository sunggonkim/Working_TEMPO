#!/usr/bin/env python3
"""Balanced diagnostic using a 58 ms pair-local high-pressure boundary."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_same_server_balanced_router_v70 as balanced
from eval.sota_4node import tempo_pd_same_server_router_v61 as prior
from tempo.pd_regime_controller import PairArrivalRegimeController
from tempo.pd_workload_policy import FrozenPDPolicy


class Threshold58Policy(FrozenPDPolicy):
    def controller(self, output_tokens: int) -> PairArrivalRegimeController:
        return PairArrivalRegimeController(
            high_pair_interval_ns=58_000_000,
            mid_pair_interval_ns=self.mid_pair_interval_ns,
            calibration_requests=self.calibration_requests,
            high_local_inflight_cap=self.high_local_credit(output_tokens),
        )


def main(argv=None) -> int:
    original = prior.FrozenPDPolicy
    prior.FrozenPDPolicy = Threshold58Policy
    try:
        return balanced.main(argv)
    finally:
        prior.FrozenPDPolicy = original


if __name__ == "__main__":
    raise SystemExit(main())
