#!/usr/bin/env python3
"""Same-server router accepting explicit balanced-crossover request IDs."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_router_v61 as prior


class BalancedSameServerCore(prior.SameServerCore):
    @staticmethod
    def _arm(request_id: str) -> tuple[str, str]:
        for arm in ("local", "tempo", "remote"):
            for replicate in (0, 1):
                for phase in ("warm", "measured"):
                    if request_id.startswith(f"ssb-{arm}-r{replicate}-{phase}-"):
                        return arm, phase
        raise ValueError("balanced same-server request ID has no explicit arm/phase prefix")


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = BalancedSameServerCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
