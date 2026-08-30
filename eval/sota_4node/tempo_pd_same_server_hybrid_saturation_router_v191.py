#!/usr/bin/env python3
"""Full-phase router accepting extra fixed-local saturation replicates."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_hybrid_phase_router_v181 as phase


class SaturationHybridCore(phase.FullPhaseHybridCore):
    @staticmethod
    def _arm(request_id: str) -> tuple[str, str]:
        for replicate in (2, 3):
            for name in ("warm", "measured"):
                if request_id.startswith(f"ssb-local-r{replicate}-{name}-"):
                    return "local", name
        return phase.FullPhaseHybridCore._arm(request_id)


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = SaturationHybridCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
