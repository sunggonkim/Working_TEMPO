#!/usr/bin/env python3
"""Production same-server controller with request-interleaved identities."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_balanced_router_v70 as balanced


class ProductionInterleavedCore(balanced.BalancedSameServerCore):
    """Keep production decisions; extend only the measurement ID grammar."""

    @staticmethod
    def _arm(request_id: str) -> tuple[str, str]:
        for arm in ("local", "tempo", "remote"):
            for replicate in (0, 1):
                if request_id.startswith(f"ssi-{arm}-r{replicate}-measured-"):
                    return arm, "measured"
        return balanced.BalancedSameServerCore._arm(request_id)


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = ProductionInterleavedCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
